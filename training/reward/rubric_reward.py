#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Asynchronous instance-aware rubric rewards for SVG generation.

Renders each candidate, filters invalid or blank SVG images, and requests
continuous rubric scores from OpenAI-compatible vision-language endpoints.
Per-item scoring guides and weights define the final reward.
"""
import os
import sys
import io
import re
import json
import time
import base64
import asyncio
import atexit
import signal
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import copy
import aiohttp
import cairosvg
import numpy as np
from PIL import Image
from datetime import datetime


if __package__:
    from .rubric_prompt import get_rubric_judging_prompt, format_rubric_items_for_judging
else:
    # Also support direct-file imports by VERL's custom reward loader.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rubric_prompt import get_rubric_judging_prompt, format_rubric_items_for_judging


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

_VLM_ENDPOINTS_STR = os.getenv("VLM_ENDPOINTS", "")
VLM_ENDPOINTS = [ep.strip().rstrip("/") for ep in _VLM_ENDPOINTS_STR.split(",") if ep.strip()]
VLM_API_KEY = os.getenv("VLM_API_KEY", "EMPTY")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen3-vl-8b-judge").strip()
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "768"))
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "120"))
VLM_MAX_RETRIES = int(os.getenv("VLM_MAX_RETRIES", "3"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "768"))

# Start conservative. Raise to 3/4 only if server GPU still has headroom.
PER_ENDPOINT_CONCURRENCY = int(os.getenv("PER_ENDPOINT_CONCURRENCY", "32"))

BLANK_MSE_THRESHOLD = float(os.getenv("BLANK_MSE_THRESHOLD", "0.0001"))
BLANK_CHECK_SIZE = (336, 336)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "training" / "reward"))
DEBUG_BLANK_DIR = os.getenv("DEBUG_BLANK_DIR", os.path.join(OUTPUT_DIR, "blank_images"))
# Set sampling rate to prevent disk bloat (1.0 = all, 0.1 = 10%)
DEBUG_BLANK_SAMPLING = float(os.getenv("DEBUG_BLANK_SAMPLING", "1.0"))

REWARD_LOG_PATH = os.getenv(
    "REWARD_LOG_PATH",
    os.path.join(OUTPUT_DIR, "reward_scores.jsonl"),
)

_RR_COUNTER = 0


DEBUG_PROMPT_LOG_PATH = os.getenv(
    "DEBUG_PROMPT_LOG_PATH",
    os.path.join(OUTPUT_DIR, "debug_api_payloads.jsonl"),
)
_DEBUG_PROMPT_COUNTER = 0
_DEBUG_PROMPT_LOCK = threading.Lock()

# ----------------------------------------------------------------------
# Render config: process pool + execution-only timeout
# ----------------------------------------------------------------------

# Only limits actual CairoSVG execution time (not queue time)
RENDER_EXEC_TIMEOUT = float(os.getenv("RENDER_EXEC_TIMEOUT", "3.0"))
RENDER_MAX_WORKERS = int(os.getenv("RENDER_MAX_WORKERS", "8"))
RENDER_SUBMIT_CONCURRENCY = int(
    os.getenv("RENDER_SUBMIT_CONCURRENCY", str(RENDER_MAX_WORKERS))
)

RENDER_MP_START_METHOD = os.getenv("RENDER_MP_START_METHOD", "forkserver").strip()
RENDER_OUTPUT_W = int(os.getenv("RENDER_OUTPUT_W", "336"))
RENDER_OUTPUT_H = int(os.getenv("RENDER_OUTPUT_H", "336"))
RENDER_BACKGROUND_COLOR = os.getenv("RENDER_BACKGROUND_COLOR", "white")

# ----------------------------------------------------------------------
# Global runtime state
# ----------------------------------------------------------------------

_WHITE_IMAGE_CACHE = None
_PENDING = [0 for _ in VLM_ENDPOINTS]
_SEMAPHORES = [asyncio.Semaphore(PER_ENDPOINT_CONCURRENCY) for _ in VLM_ENDPOINTS]
_PICK_LOCK: Optional[asyncio.Lock] = None
_ASYNC_SESSIONS: Dict[str, aiohttp.ClientSession] = {}

_RENDER_POOL: Optional[ProcessPoolExecutor] = None
_RENDER_POOL_INIT_LOCK = threading.Lock()
_RENDER_SUBMIT_SEM: Optional[asyncio.Semaphore] = None
_RENDER_RUNTIME_LOCK: Optional[asyncio.Lock] = None

_SVG_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)

def _fix_svg(svg_content: str) -> str:
    svg_content = re.sub(r"<[^>]*$", "", svg_content)

    open_tags = re.findall(r"<([a-zA-Z][a-zA-Z0-9:_-]*)\b[^>/]*?>", svg_content)
    close_tags = re.findall(r"</([a-zA-Z][a-zA-Z0-9:_-]*)>", svg_content)

    tmp_close_tags = close_tags.copy()
    stack = []
    for tag in open_tags:
        if tag not in tmp_close_tags:
            stack.append(tag)
        else:
            tmp_close_tags.remove(tag)

    while stack:
        svg_content += f"</{stack.pop()}>"

    if "</svg>" not in svg_content:
        svg_content += "</svg>"

    return svg_content

def extract_svg_block(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _SVG_RE.search(text)
    if m:
        # regex matches only if closing </svg> is present → complete, no fix needed
        return m.group(0)
    text = text.strip()
    if "<svg" in text.lower():
        # <svg> exists but regex failed → code is incomplete/truncated, fix it
        return _fix_svg(text)
    return None

def _short_exc(e: Exception, max_len: int = 500) -> str:
    s = f"{type(e).__name__}: {str(e)}"
    return s.replace("\n", " ")[:max_len]

def _short_text(s: str, max_len: int = 500) -> str:
    return s.replace("\n", "\\n")[:max_len]

# ----------------------------------------------------------------------
# Log
# ----------------------------------------------------------------------

_REWARD_LOG_LOCK = threading.Lock()


def _extract_case_id(ground_truth: dict | None, extra_info: dict | None) -> str:
    ground_truth = ground_truth or {}
    extra_info = extra_info or {}

    for key in ["id", "sample_id", "uuid", "case_id"]:
        if key in extra_info and extra_info[key] is not None:
            return str(extra_info[key])
        if key in ground_truth and ground_truth[key] is not None:
            return str(ground_truth[key])

    gt_img = ground_truth.get("gt_img")
    if gt_img:
        return os.path.splitext(os.path.basename(str(gt_img)))[0]

    return "unknown"


def _log_debug_payload_sync(payload: dict):
    """Save first 3 payloads for debugging, truncating base64 strings."""
    global _DEBUG_PROMPT_COUNTER

    with _DEBUG_PROMPT_LOCK:
        if _DEBUG_PROMPT_COUNTER >= 3:
            return

        _DEBUG_PROMPT_COUNTER += 1
        current_idx = _DEBUG_PROMPT_COUNTER

    try:
        debug_payload = copy.deepcopy(payload)

        try:
            messages = debug_payload.get("messages", [])
            for msg in messages:
                for content_item in msg.get("content", []):
                    if content_item.get("type") == "image_url":
                        img_url = content_item["image_url"]["url"]
                        if img_url.startswith("data:image"):
                            content_item["image_url"]["url"] = img_url[:40] + "...[TRUNCATED_BASE64]..."
        except Exception:
            pass

        record = {
            "debug_idx": current_idx,
            "ts": datetime.now().isoformat(),
            "payload": debug_payload
        }

        os.makedirs(os.path.dirname(DEBUG_PROMPT_LOG_PATH), exist_ok=True)

        with open(DEBUG_PROMPT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")

    except Exception as e:
        print(f"Failed to log debug payload: {e}", file=sys.stderr)


def _log_reward_event(
    case_id: str,
    score: float,
    status: str,
    latency_sec: float | None = None,
    extra: dict | None = None,
):
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "case_id": case_id,
        "score": float(score),
        "status": status,
    }
    if latency_sec is not None:
        record["latency_sec"] = round(float(latency_sec), 4)
    if extra:
        record.update(extra)

    line = json.dumps(record, ensure_ascii=False)
    with _REWARD_LOG_LOCK:
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------

def _get_white_image() -> np.ndarray:
    global _WHITE_IMAGE_CACHE
    if _WHITE_IMAGE_CACHE is None:
        _WHITE_IMAGE_CACHE = np.full((*BLANK_CHECK_SIZE, 3), 255, dtype=np.float32)
    return _WHITE_IMAGE_CACHE


def _compute_mse_vs_white(image_bytes: bytes) -> float:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(BLANK_CHECK_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        white_array = _get_white_image() / 255.0
        return float(np.mean((img_array - white_array) ** 2))
    except Exception:
        return float("inf")


def is_blank_image(image_bytes: bytes, threshold: float = BLANK_MSE_THRESHOLD) -> bool:
    return _compute_mse_vs_white(image_bytes) < threshold


def quick_reject_check(svg_code: Optional[str]) -> Tuple[bool, float, str]:
    if not svg_code:
        return True, 0.0, "empty_svg"

    svg_clean = svg_code.strip()

    if "<svg" not in svg_clean.lower():
        return True, 0.0, "no_svg_tag"

    if len(svg_clean) < 100:
        return True, 0.0, "svg_too_short"

    return False, 0.0, "needs_vlm"


def _b64_image_bytes(image_bytes: bytes, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


@lru_cache(maxsize=4096)
def _read_image_as_png_bytes_cached(image_path: str) -> bytes:
    img = Image.open(image_path).convert("RGBA")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


@lru_cache(maxsize=4096)
def _read_image_as_data_url_cached(image_path: str) -> str:
    return _b64_image_bytes(_read_image_as_png_bytes_cached(image_path))


# ----------------------------------------------------------------------
# Render in dedicated process pool
# ----------------------------------------------------------------------

class _RenderExecTimeout(Exception):
    pass


def _render_timeout_handler(signum, frame):
    raise _RenderExecTimeout("CairoSVG execution timeout")


def _get_render_pool() -> ProcessPoolExecutor:
    global _RENDER_POOL

    if _RENDER_POOL is not None:
        return _RENDER_POOL

    with _RENDER_POOL_INIT_LOCK:
        if _RENDER_POOL is None:
            mp_ctx = mp.get_context(RENDER_MP_START_METHOD)
            _RENDER_POOL = ProcessPoolExecutor(
                max_workers=RENDER_MAX_WORKERS,
                mp_context=mp_ctx,
            )

    return _RENDER_POOL


async def _ensure_render_runtime() -> None:
    global _RENDER_SUBMIT_SEM, _RENDER_RUNTIME_LOCK

    if _RENDER_RUNTIME_LOCK is None:
        _RENDER_RUNTIME_LOCK = asyncio.Lock()

    async with _RENDER_RUNTIME_LOCK:
        if _RENDER_SUBMIT_SEM is None:
            _RENDER_SUBMIT_SEM = asyncio.Semaphore(RENDER_SUBMIT_CONCURRENCY)
            _get_render_pool()


def _shutdown_render_pool_sync():
    global _RENDER_POOL
    if _RENDER_POOL is not None:
        try:
            _RENDER_POOL.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _RENDER_POOL = None


atexit.register(_shutdown_render_pool_sync)


def _render_svg_in_subprocess(
    svg_code: str,
    submit_ts_wall: float,
    output_width: int,
    output_height: int,
    background_color: str,
    exec_timeout_sec: float,
) -> Dict[str, Any]:
    start_ts_wall = time.time()
    start_ts_perf = time.perf_counter()

    old_handler = None
    timer_armed = False

    try:
        old_handler = signal.signal(signal.SIGALRM, _render_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, exec_timeout_sec)
        timer_armed = True

        png_bytes = cairosvg.svg2png(
            bytestring=svg_code.encode("utf-8"),
            output_width=output_width,
            output_height=output_height,
            background_color=background_color,
        )

        end_ts_perf = time.perf_counter()
        end_ts_wall = time.time()

        return {
            "status": "ok",
            "png_bytes": png_bytes,
            "error": "",
            "render_submit_wait_sec": 0.0,
            "render_queue_wait_sec": max(0.0, start_ts_wall - submit_ts_wall),
            "render_pre_exec_wait_sec": 0.0,
            "render_exec_sec": max(0.0, end_ts_perf - start_ts_perf),
            "render_total_wall_sec": max(0.0, end_ts_wall - submit_ts_wall),
            "backend": "processpool_exec_timeout",
        }

    except _RenderExecTimeout:
        end_ts_perf = time.perf_counter()
        end_ts_wall = time.time()

        return {
            "status": "render_exec_timeout",
            "png_bytes": None,
            "error": f"render_exec_timeout>{exec_timeout_sec}s",
            "render_submit_wait_sec": 0.0,
            "render_queue_wait_sec": max(0.0, start_ts_wall - submit_ts_wall),
            "render_pre_exec_wait_sec": 0.0,
            "render_exec_sec": max(0.0, end_ts_perf - start_ts_perf),
            "render_total_wall_sec": max(0.0, end_ts_wall - submit_ts_wall),
            "backend": "processpool_exec_timeout",
        }

    except Exception as e:
        end_ts_perf = time.perf_counter()
        end_ts_wall = time.time()

        return {
            "status": "render_fail",
            "png_bytes": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "render_submit_wait_sec": 0.0,
            "render_queue_wait_sec": max(0.0, start_ts_wall - submit_ts_wall),
            "render_pre_exec_wait_sec": 0.0,
            "render_exec_sec": max(0.0, end_ts_perf - start_ts_perf),
            "render_total_wall_sec": max(0.0, end_ts_wall - submit_ts_wall),
            "backend": "processpool_exec_timeout",
        }

    finally:
        if timer_armed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
        if old_handler is not None:
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass


async def render_svg_async(svg_code: str) -> Dict[str, Any]:
    await _ensure_render_runtime()

    req_ts_wall = time.time()
    loop = asyncio.get_running_loop()

    async with _RENDER_SUBMIT_SEM:
        submit_ts_wall = time.time()

        result = await loop.run_in_executor(
            _get_render_pool(),
            _render_svg_in_subprocess,
            svg_code,
            submit_ts_wall,
            RENDER_OUTPUT_W,
            RENDER_OUTPUT_H,
            RENDER_BACKGROUND_COLOR,
            RENDER_EXEC_TIMEOUT,
        )

    submit_wait_sec = max(0.0, submit_ts_wall - req_ts_wall)
    queue_wait_sec = float(result.get("render_queue_wait_sec", 0.0))

    result["render_submit_wait_sec"] = submit_wait_sec
    result["render_pre_exec_wait_sec"] = submit_wait_sec + queue_wait_sec
    return result


def _save_success_svg_sync(case_id: str, svg_code: str):
    try:
        date_str = datetime.now().strftime("%Y%m%d_%H")
        save_dir = os.path.join(DEBUG_BLANK_DIR, "success", date_str)
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.now().strftime("%M%S_%f")
        file_path = os.path.join(save_dir, f"{ts}_{case_id}.svg")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
    except Exception as e:
        print(f"Failed to save success SVG: {e}", file=sys.stderr)


def _save_svg_only_sync(case_id: str, svg_code: str, status: str):
    try:
        date_str = datetime.now().strftime("%Y%m%d_%H")
        save_dir = os.path.join(DEBUG_BLANK_DIR, "fails", status, date_str)
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.now().strftime("%M%S_%f")
        file_path = os.path.join(save_dir, f"{ts}_{case_id}.svg")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
    except Exception as e:
        print(f"Failed to save debug SVG: {e}", file=sys.stderr)


def _save_blank_debug_data_sync(case_id: str, svg_code: str, png_bytes: bytes):
    try:
        date_str = datetime.now().strftime("%Y%m%d_%H")
        save_dir = os.path.join(DEBUG_BLANK_DIR, date_str)
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.now().strftime("%M%S")
        base_name = f"{ts}_{case_id}"

        with open(os.path.join(save_dir, f"{base_name}.svg"), "w", encoding="utf-8") as f:
            f.write(svg_code)

        if png_bytes:
            with open(os.path.join(save_dir, f"{base_name}.png"), "wb") as f:
                f.write(png_bytes)
    except Exception as e:
        print(f"Failed to save blank debug data: {e}", file=sys.stderr)

# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------

def _parse_json_object(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None

    text = content.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            obj = json.loads(text[left : right + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def _extract_satisfaction(item_result: Dict[str, Any]) -> float:
    if "satisfaction" in item_result:
        try:
            return max(0.0, min(1.0, float(item_result["satisfaction"])))
        except (ValueError, TypeError):
            pass

    for key in ["score", "satisfaction_score", "value"]:
        if key in item_result:
            try:
                return max(0.0, min(1.0, float(item_result[key])))
            except (ValueError, TypeError):
                pass

    if "satisfied" in item_result:
        val = item_result["satisfied"]
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, str):
            return 1.0 if val.strip().lower() in {"true", "1", "yes", "y", "pass"} else 0.0

    return 0.0


def normalize_rubric_items(rubric_items: Any) -> List[Dict[str, Any]]:
    if not isinstance(rubric_items, list):
        return []

    normalized = []
    for idx, item in enumerate(rubric_items):
        if isinstance(item, dict):
            title = item.get("title") or item.get("name") or item.get("criterion") or f"item_{idx}"
            desc = item.get("description") or item.get("desc") or item.get("requirement") or ""
            weight = float(item.get("weight", 3.0))
            # Preserve scoring_guide if present
            sg = item.get("scoring_guide", "")
        else:
            title = f"item_{idx}"
            desc = str(item).strip()
            weight = 3.0
            sg = ""

        entry = {
            "title": title,
            "description": desc,
            "weight": weight,
            "index": idx,
        }
        if sg:
            entry["scoring_guide"] = sg
        normalized.append(entry)
    return normalized


def parse_judge_result(
    judge_result: Dict[str, Any],
    rubric_items: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    items_list = judge_result.get("items", [])
    item_results = []
    total_weight = 0.0
    weighted_satisfaction = 0.0

    for idx, rubric_item in enumerate(rubric_items):
        judge_item = None
        for ji in items_list:
            if ji.get("item_index") == (idx + 1) or ji.get("title") == rubric_item["title"]:
                judge_item = ji
                break

        if judge_item:
            satisfaction = _extract_satisfaction(judge_item)
            reason = judge_item.get("reason", "")
        else:
            satisfaction = 0.0
            reason = "No evaluation provided"

        weight = rubric_item["weight"]

        if weight > 0:
            total_weight += weight
        weighted_satisfaction += satisfaction * weight

        item_results.append(
            {
                "item_index": idx + 1,
                "title": rubric_item["title"],
                "description": rubric_item["description"],
                "weight": weight,
                "satisfaction": round(satisfaction, 4),
                "reason": reason,
            }
        )

    final_score = round(
        max(0.0, min(1.0, (weighted_satisfaction / total_weight) if total_weight > 0 else 0.0)),
        4,
    )
    return final_score, item_results


# ----------------------------------------------------------------------
# Async HTTP helpers
# ----------------------------------------------------------------------

def _request_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json",
    }


async def _ensure_runtime_objects() -> None:
    global _PICK_LOCK
    if _PICK_LOCK is None:
        _PICK_LOCK = asyncio.Lock()


async def _get_session(endpoint: str) -> aiohttp.ClientSession:
    await _ensure_runtime_objects()

    sess = _ASYNC_SESSIONS.get(endpoint)
    if sess is not None and not sess.closed:
        return sess

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=VLM_TIMEOUT)
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=100, ttl_dns_cache=300)
    sess = aiohttp.ClientSession(
        headers=_request_headers(),
        timeout=timeout,
        connector=connector,
    )
    _ASYNC_SESSIONS[endpoint] = sess
    return sess


async def _pick_endpoint_idx() -> int:
    global _RR_COUNTER
    await _ensure_runtime_objects()
    async with _PICK_LOCK:
        min_pending = min(_PENDING)
        candidates = [i for i, p in enumerate(_PENDING) if p == min_pending]
        idx = candidates[_RR_COUNTER % len(candidates)]
        _RR_COUNTER += 1
        return idx



async def _post_json(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    session = await _get_session(endpoint)
    async with session.post(f"{endpoint}/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def close_async_sessions() -> None:
    for sess in list(_ASYNC_SESSIONS.values()):
        try:
            if not sess.closed:
                await sess.close()
        except Exception:
            pass
    _ASYNC_SESSIONS.clear()


# ----------------------------------------------------------------------
# Async VLM judge
# ----------------------------------------------------------------------

async def judge_with_vlm_async(
    candidate_png_bytes: bytes,
    rubric_items: List[Dict[str, Any]],
    user_prompt: str,
) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
    if not VLM_ENDPOINTS:
        raise ValueError(
            "Set VLM_ENDPOINTS to one or more comma-separated OpenAI-compatible VLM base URLs"
        )
    start_time = time.time()
    debug_info = {"attempts": []}

    try:
        cand_data_url = _b64_image_bytes(candidate_png_bytes)
    except Exception as e:
        debug_info["image_prepare_error"] = _short_exc(e)
        return None, time.time() - start_time, debug_info

    prompt_template = get_rubric_judging_prompt()
    rubric_formatted = format_rubric_items_for_judging(rubric_items)

    prompt_text = prompt_template.replace("{{user_prompt}}", user_prompt)
    prompt_text = prompt_text.replace("{{rubric_items_formatted}}", rubric_formatted)

    prompt_text += (
        "\n\nOutput JSON only. No extra explanation outside JSON. "
        "Keep each reason brief (1 short sentence)."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Candidate image (to evaluate):"},
                {"type": "image_url", "image_url": {"url": cand_data_url}},
                {"type": "text", "text": f"\n{prompt_text}"},
            ],
        }
    ]

    payload = {
        "model": VLM_MODEL,
        "messages": messages,
        "max_tokens": VLM_MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 3407,
    }

    if _DEBUG_PROMPT_COUNTER < 3:
        asyncio.create_task(asyncio.to_thread(_log_debug_payload_sync, payload))

    for attempt in range(max(1, VLM_MAX_RETRIES)):
        idx = await _pick_endpoint_idx()
        endpoint = VLM_ENDPOINTS[idx]

        async with _SEMAPHORES[idx]:
            _PENDING[idx] += 1
            try:
                response_data = await _post_json(endpoint, payload)

                choices = response_data.get("choices", [])
                if not choices:
                    raise ValueError(f"empty choices, response={json.dumps(response_data)[:500]}")

                message = choices[0].get("message", {})
                content = message.get("content", "")

                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    content = "\n".join(parts).strip()
                elif content is None:
                    content = ""

                parsed = _parse_json_object(content)
                if parsed is None:
                    debug_info["attempts"].append({
                        "attempt": attempt + 1,
                        "endpoint": endpoint,
                        "error": "parse_json_fail",
                        "raw_content_head": _short_text(content, 800),
                    })
                    await asyncio.sleep(0.2)
                    continue

                return parsed, time.time() - start_time, debug_info

            except Exception as e:
                debug_info["attempts"].append({
                    "attempt": attempt + 1,
                    "endpoint": endpoint,
                    "error": _short_exc(e, 800),
                })
                await asyncio.sleep(0.2)
            finally:
                _PENDING[idx] -= 1

    return None, time.time() - start_time, debug_info


def _score_result(score: float, status: str, error: Optional[str] = None, return_status: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {"score": float(score)}
    if return_status:
        result["status"] = status
        if error:
            result["error"] = error
    return result


async def compute_score_fast(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
    return_status: bool = False,
) -> Dict[str, Any]:
    extra_info = extra_info or {}
    case_id = _extract_case_id(ground_truth, extra_info)

    prompt_text = extra_info.get("prompt_text")

    gt_img_path = ground_truth.get("gt_img") if isinstance(ground_truth, dict) else None
    if gt_img_path and not os.path.exists(gt_img_path):
        _log_reward_event(case_id, 0.0, "missing_gt")
        return _score_result(0.0, "missing_gt", return_status=return_status)

    rubric_items = normalize_rubric_items(extra_info.get("rubric_items", []))
    if not rubric_items:
        _log_reward_event(case_id, 0.0, "missing_rubric")
        return _score_result(0.0, "missing_rubric", return_status=return_status)

    svg_code = extract_svg_block(solution_str)
    should_reject, reject_score, reject_reason = quick_reject_check(svg_code)
    if should_reject:
        _log_reward_event(
            case_id,
            float(reject_score),
            "quick_reject",
            extra={"reason": reject_reason},
        )
        return _score_result(float(reject_score), "quick_reject", reject_reason, return_status=return_status)

    render_result = await render_svg_async(svg_code)
    render_status = render_result.get("status", "render_fail")

    render_extra = {
        "render_backend": render_result.get("backend", ""),
        "render_submit_wait_sec": round(float(render_result.get("render_submit_wait_sec", 0.0)), 4),
        "render_queue_wait_sec": round(float(render_result.get("render_queue_wait_sec", 0.0)), 4),
        "render_pre_exec_wait_sec": round(float(render_result.get("render_pre_exec_wait_sec", 0.0)), 4),
        "render_exec_sec": round(float(render_result.get("render_exec_sec", 0.0)), 4),
        "render_total_wall_sec": round(float(render_result.get("render_total_wall_sec", 0.0)), 4),
    }

    if render_status == "render_exec_timeout":
        asyncio.create_task(
            asyncio.to_thread(_save_svg_only_sync, case_id, svg_code or "", "render_exec_timeout")
        )
        _log_reward_event(
            case_id,
            0.0,
            "render_exec_timeout",
            latency_sec=float(render_result.get("render_exec_sec", 0.0)),
            extra={**render_extra, "error": render_result.get("error", "")},
        )
        return _score_result(0.0, "render_exec_timeout", render_result.get("error", ""), return_status=return_status)

    if render_status != "ok":
        _log_reward_event(
            case_id,
            0.0,
            "render_fail",
            latency_sec=float(render_result.get("render_exec_sec", 0.0)),
            extra={**render_extra, "error": render_result.get("error", "")},
        )
        return _score_result(0.0, "render_fail", render_result.get("error", ""), return_status=return_status)

    candidate_png_bytes = render_result["png_bytes"]

    is_blank = await asyncio.to_thread(is_blank_image, candidate_png_bytes)
    if is_blank:
        import random
        if random.random() < DEBUG_BLANK_SAMPLING:
            asyncio.create_task(asyncio.to_thread(
                _save_blank_debug_data_sync,
                case_id,
                svg_code or "",
                candidate_png_bytes
            ))

        _log_reward_event(case_id, 0.0, "blank_image", extra=render_extra)
        return _score_result(0.0, "blank_image", return_status=return_status)

    judge_result, latency, debug_info = await judge_with_vlm_async(
        user_prompt=prompt_text,
        candidate_png_bytes=candidate_png_bytes,
        rubric_items=rubric_items,
    )

    if judge_result is None:
        last_error = "unknown_error"
        if debug_info.get("attempts"):
            last_error = debug_info["attempts"][-1].get("error", "unknown_error")
        _log_reward_event(
            case_id,
            0.0,
            "judge_fail",
            latency_sec=latency,
            extra={**render_extra, **debug_info},
        )
        return _score_result(0.0, "judge_fail", last_error, return_status=return_status)

    final_score, _ = parse_judge_result(judge_result, rubric_items)
    _log_reward_event(
        case_id,
        float(final_score),
        "judged",
        latency_sec=latency,
        extra=render_extra,
    )

    if float(final_score) >= 1.0:
        asyncio.create_task(asyncio.to_thread(_save_success_svg_sync, case_id, svg_code or ""))

    return _score_result(float(final_score), "judged", return_status=return_status)


async def compute_score_fast_safe(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wrapper that returns string '0.0' for abnormal zero scores so callers can distinguish them."""
    extra_info = extra_info or {}

    # Detect incomplete SVG that would have been silently fixed before
    svg_code = extract_svg_block(solution_str)
    if svg_code is None and solution_str and "<svg" in solution_str.lower():
        return {"score": "0.0", "status": "incomplete_svg", "error": "SVG tag present but regex did not match closing tag"}

    try:
        result = await compute_score_fast(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            return_status=True,
        )
        score = result.get("score")
        status = result.get("status", "")
        error = result.get("error", "")

        # Mark abnormal zero scores with string "0.0"
        abnormal_statuses = ("judge_fail", "render_fail", "render_exec_timeout")
        if score == 0.0 and status in abnormal_statuses:
            return {"score": "0.0", "status": status, "error": error}

        return result
    except Exception as e:
        return {"score": "0.0", "status": "exception", "error": str(e)}


# Optional alias
compute_score_auto = compute_score_fast
