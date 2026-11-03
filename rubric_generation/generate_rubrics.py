#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import shutil
import random
import signal
import argparse
import cairosvg
import pandas as pd
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
if __package__:
    from .prompt import get_svg_rubric_generation_prompt
else:
    from prompt import get_svg_rubric_generation_prompt


# ----------------------------------------------------------------------
# OpenAI-compatible model client
# ----------------------------------------------------------------------

class RubricClient:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY")

        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=240.0
        )

    @staticmethod
    def _encode_image(image_path):
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _normalize_response_content(content):
        if isinstance(content, str):
            return content

        # 兼容某些 OpenAI-compatible endpoint 返回 list content 的情况
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif item.get("type") == "text" and "content" in item:
                        parts.append(str(item["content"]))
            return "".join(parts)

        return str(content) if content is not None else ""

    def multimodal_conversation(self, inputs, model_name, gen_kwargs=None):
        prompt_text = ""
        image_path = None

        for item in inputs:
            if item.get("data_type") == "text":
                prompt_text = item.get("data_value", "")
            elif item.get("data_type") == "image":
                image_path = item.get("data_value")

        if not prompt_text:
            return [""]

        gen_kwargs = gen_kwargs or {}
        temperature = gen_kwargs.get("temperature", 0.6)
        max_tokens = gen_kwargs.get("max_new_tokens", 32768)

        # ✅ 根据模型名选择正确的token参数
        if 'gemini' in model_name:
            token_param = {'max_completion_tokens': max_tokens}
        else:
            token_param = {'max_tokens': max_tokens}


        content = [{"type": "text", "text": prompt_text}]

        # 当前 prompt-only rubric generation 不建议传图；
        # 但这里保留兼容能力：如果 inputs 里真的传了 image，就会发给模型。
        image_base64 = self._encode_image(image_path)
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })

        max_retry = 5
        for attempt in range(max_retry):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": content
                    }],
                    **token_param,
                    temperature=temperature,
                )

                if not response.choices:
                    raise ValueError("Empty response choices")

                response_text = self._normalize_response_content(
                    response.choices[0].message.content
                )

                if not response_text.strip():
                    raise ValueError("Empty response content")

                return [response_text]

            except Exception as e:
                if attempt == max_retry - 1:
                    print(f"Final Failure: {e}")
                    return [""]

                wait_time = min(2 ** attempt, 60)
                if attempt > 1:
                    print(
                        f"Retry {attempt + 1}/{max_retry} error: "
                        f"{str(e)[:150]}. Waiting {wait_time}s..."
                    )
                time.sleep(wait_time)

        return [""]


# ----------------------------------------------------------------------
# IO utils
# ----------------------------------------------------------------------

def append_jsonl(jsonl_path, record: dict):
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_save_json(obj, save_path, compact=True):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        if compact:
            json.dump(obj, f, ensure_ascii=False)
        else:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, save_path)


def atomic_save_text(text: str, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, save_path)


def save_records_jsonl(records, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_filename(sample_id: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(sample_id)).strip("_") or "unknown_id"


def remove_if_exists(path):
    try:
        if path and Path(path).exists():
            Path(path).unlink()
    except Exception:
        pass


# ----------------------------------------------------------------------
# Parse utils
# ----------------------------------------------------------------------

def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```svg"):
        text = text[6:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def normalize_svg_code(svg_code: str):
    if not isinstance(svg_code, str):
        return None
    svg_code = strip_code_fence(svg_code).strip()
    left = svg_code.find("<svg")
    right = svg_code.rfind("</svg>")
    if left != -1 and right != -1 and right > left:
        svg_code = svg_code[left:right + len("</svg>")]
    return svg_code if svg_code.startswith("<svg") and svg_code.endswith("</svg>") else None


def extract_first_json_array(text: str):
    """
    从文本中提取第一个合法 JSON array。
    用于 fallback，避免简单 find("[") / rfind("]") 误伤。
    """
    start = text.find("[")
    while start != -1:
        depth = 0
        in_str = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, list):
                            return data
                    except Exception:
                        break

        start = text.find("[", start + 1)

    return None


def parse_generation_result(content: str):
    """
    解析模型输出：

    ```xml
    <svg ...>
    ...
    </svg>
    ```

    ```json
    [
      {...},
      ...
    ]
    ```

    返回：
    {
        "ideal_svg": str,
        "rubric_items": list
    }
    """
    if not content:
        return None

    text = content.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text

    ideal_svg = None
    rubric_items = None

    # 解析所有 markdown fenced code blocks
    fenced_blocks = re.findall(
        r"```([^\n`]*)\n(.*?)```",
        text,
        flags=re.DOTALL
    )

    for lang, body in fenced_blocks:
        lang = lang.strip().lower().split()[0] if lang.strip() else ""
        body = body.strip()

        # 1) XML / SVG block
        if ideal_svg is None and (lang in {"xml", "svg"} or "<svg" in body):
            maybe_svg = normalize_svg_code(body)
            if maybe_svg is not None:
                ideal_svg = maybe_svg
                continue

        # 2) JSON block
        if rubric_items is None and (lang == "json" or body.startswith("[") or body.startswith("{")):
            try:
                data = json.loads(body)
                if isinstance(data, list):
                    rubric_items = data
                    continue
                if isinstance(data, dict) and isinstance(data.get("rubric_items"), list):
                    rubric_items = data["rubric_items"]
                    if ideal_svg is None:
                        maybe_svg = normalize_svg_code(data.get("ideal_svg"))
                        if maybe_svg is not None:
                            ideal_svg = maybe_svg
                    continue
            except Exception:
                pass

    # fallback 1: 如果没有成功解析 xml block，就从全文抽取 <svg>...</svg>
    if ideal_svg is None:
        ideal_svg = normalize_svg_code(text)

    # fallback 2: 如果没有成功解析 json block，就从全文里抽第一个 JSON array
    if rubric_items is None:
        rubric_items = extract_first_json_array(text)

    if ideal_svg is None or rubric_items is None:
        return None

    return {
        "ideal_svg": ideal_svg,
        "rubric_items": rubric_items
    }


# ----------------------------------------------------------------------
# Render utils
# ----------------------------------------------------------------------

class RenderTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise RenderTimeoutError("SVG render timeout")


def render_svg_to_png_with_timeout(svg_code: str, png_path, timeout_sec=3, background_color="white"):
    """
    在当前进程中使用 signal 超时控制。
    """
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = png_path.with_suffix(png_path.suffix + ".tmp")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)

        cairosvg.svg2png(
            bytestring=svg_code.encode("utf-8"),
            write_to=str(tmp_png),
            background_color=background_color
        )

        signal.setitimer(signal.ITIMER_REAL, 0)
        os.replace(tmp_png, png_path)

    except Exception:
        signal.setitimer(signal.ITIMER_REAL, 0)
        remove_if_exists(tmp_png)
        raise
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ----------------------------------------------------------------------
# Existing-result loading
# ----------------------------------------------------------------------

def load_existing_success(final_json_path):
    final_json_path = Path(final_json_path)
    if not final_json_path.exists():
        return {}
    try:
        with open(final_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            if "results" in obj and isinstance(obj["results"], dict):
                return obj["results"]
            return obj
    except Exception as e:
        print(f"Warning: failed to load existing success file {final_json_path}: {e}")
    return {}


def load_existing_failed(failed_json_path):
    failed_json_path = Path(failed_json_path)
    if not failed_json_path.exists():
        return {}
    try:
        with open(failed_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        print(f"Warning: failed to load existing failed file {failed_json_path}: {e}")
    return {}


# ----------------------------------------------------------------------
# Sample / load
# ----------------------------------------------------------------------

def maybe_sample_records(records, sample_size=None, sample_ratio=None, sample_seed=42):
    if sample_size is not None and sample_ratio is not None:
        raise ValueError("Only one of --sample_size and --sample_ratio can be set.")

    n = len(records)
    if n == 0:
        return [], []

    if sample_ratio is not None:
        if not (0 < sample_ratio <= 1):
            raise ValueError("--sample_ratio must be in (0, 1].")
        sample_size = max(1, int(round(n * sample_ratio)))

    if sample_size is None:
        return records, list(range(n))

    if sample_size <= 0:
        raise ValueError("--sample_size must be > 0.")
    if sample_size > n:
        print(f"[WARN] sample_size={sample_size} > total={n}, will use full dataset.")
        sample_size = n

    rng = random.Random(sample_seed)
    sampled_indices = rng.sample(range(n), sample_size)
    sampled_indices.sort()
    sampled_records = [records[i] for i in sampled_indices]
    print(f"[INFO] Random sampled {len(sampled_records)} / {n} samples with seed={sample_seed}")
    return sampled_records, sampled_indices


def load_rollout_aligned_records_from_parquet(input_parquet):
    df = pd.read_parquet(input_parquet)

    records = []
    skipped_missing_id_or_prompt = 0
    missing_gt_fields = 0

    for row_idx, row in enumerate(df.itertuples()):
        extra_info = getattr(row, "extra_info", None) or {}
        case_id = extra_info.get("id", "")
        prompt_text = extra_info.get("prompt_text", "")

        if not case_id or not prompt_text:
            skipped_missing_id_or_prompt += 1
            continue

        record = {
            "id": str(case_id),
            "description": prompt_text,
            "source_index": len(records),
            "parquet_row_index": row_idx,
            "gt_code": None,
            "gt_img": None,
            "is_rubric_ready": False,
            "malformed_reason": None,
        }

        try:
            reward_model = getattr(row, "reward_model", None) or {}
            ground_truth = reward_model.get("ground_truth", {}) if isinstance(reward_model, dict) else {}
            gt_code = ground_truth.get("gt_code", None)
            gt_img = ground_truth.get("gt_img", None)

            if gt_code and gt_img:
                record["gt_code"] = gt_code
                record["gt_img"] = gt_img
                record["is_rubric_ready"] = True
            else:
                record["malformed_reason"] = "missing_gt_code_or_gt_img"
                missing_gt_fields += 1

        except Exception as e:
            record["malformed_reason"] = f"gt_extract_error: {e}"
            missing_gt_fields += 1

        records.append(record)

    return records, {
        "total_candidates_after_rollout_filter": len(records),
        "skipped_missing_id_or_prompt": skipped_missing_id_or_prompt,
        "missing_gt_fields_within_candidates": missing_gt_fields,
    }


def load_or_create_sample_manifest(
    manifest_path,
    candidate_records,
    sample_size,
    sample_ratio,
    sample_seed,
    input_parquet,
    force_resample=False,
):
    manifest_path = Path(manifest_path)

    if manifest_path.exists() and not force_resample:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        sampled_indices = manifest.get("sampled_indices", None)
        sampled_ids = [str(x) for x in manifest.get("sampled_ids", [])]

        print(f"Loaded existing sample manifest from: {manifest_path}")
        print(f"Sampled ids in manifest: {len(sampled_ids)}")

        if sampled_indices is not None:
            return manifest, sampled_indices, sampled_ids

        id_to_first_idx = {}
        for i, rec in enumerate(candidate_records):
            sid = str(rec["id"])
            if sid not in id_to_first_idx:
                id_to_first_idx[sid] = i

        recovered_indices = [id_to_first_idx[sid] for sid in sampled_ids if sid in id_to_first_idx]
        return manifest, recovered_indices, sampled_ids

    sampled_records, sampled_indices = maybe_sample_records(
        candidate_records,
        sample_size=sample_size,
        sample_ratio=sample_ratio,
        sample_seed=sample_seed,
    )
    sampled_ids = [str(x["id"]) for x in sampled_records]

    manifest = {
        "input_parquet": str(input_parquet),
        "sample_seed": sample_seed,
        "requested_sample_size": sample_size,
        "requested_sample_ratio": sample_ratio,
        "actual_sample_size": len(sampled_records),
        "candidate_count": len(candidate_records),
        "sampled_indices": sampled_indices,
        "sampled_ids": sampled_ids,
    }

    atomic_save_json(manifest, manifest_path)
    print(f"Created new sample manifest: {manifest_path}")
    print(f"Actual sampled size: {len(sampled_records)}")
    return manifest, sampled_indices, sampled_ids


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------

def merge_shards_and_save(
    final_json_path,
    failed_json_path,
    summary_json_path,
    shard_dir,
    input_parquet,
    temperature,
    total_sampled_count,
    total_rubric_ready_count,
    sampled_id_set,
    sample_manifest_path,
    candidate_stats,
):
    success_map = {
        str(k): v
        for k, v in load_existing_success(final_json_path).items()
        if str(k) in sampled_id_set
    }
    failed_map = {
        str(k): v
        for k, v in load_existing_failed(failed_json_path).items()
        if str(k) in sampled_id_set
    }

    shard_dir = Path(shard_dir)
    shard_files = sorted(shard_dir.glob("rank_*.jsonl"))

    for shard_file in shard_files:
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                sid = str(obj.get("id"))
                if not sid or sid not in sampled_id_set:
                    continue

                if obj.get("status") == "ok":
                    success_map[sid] = obj
                    failed_map.pop(sid, None)
                else:
                    if sid not in success_map:
                        failed_map[sid] = obj

    summary_obj = {
        "input_parquet": str(input_parquet),
        "sample_manifest_path": str(sample_manifest_path),
        "temperature": temperature,
        "total_sampled_count": total_sampled_count,
        "sampled_rubric_ready_count": total_rubric_ready_count,
        "successful_count": len(success_map),
        "failed_count": len(failed_map),
        "success_rate_over_sampled_rubric_ready": round(
            len(success_map) / total_rubric_ready_count, 6
        ) if total_rubric_ready_count > 0 else 0.0,
        "success_rate_over_total_sampled": round(
            len(success_map) / total_sampled_count, 6
        ) if total_sampled_count > 0 else 0.0,
        "candidate_stats": candidate_stats,
    }

    atomic_save_json(success_map, final_json_path)
    atomic_save_json(failed_map, failed_json_path)
    atomic_save_json(summary_obj, summary_json_path)
    return success_map, failed_map, summary_obj


# ----------------------------------------------------------------------
# Failed-debug saving
# ----------------------------------------------------------------------

def save_failed_debug(failed_dir, sample_id, raw_response, error_msg, metadata):
    failed_dir = Path(failed_dir)
    raw_dir = failed_dir / "raw_response"
    rec_dir = failed_dir / "records"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    fname = safe_filename(sample_id)
    raw_path = raw_dir / f"{fname}.txt"
    rec_path = rec_dir / f"{fname}.json"

    if raw_response is not None:
        atomic_save_text(raw_response, raw_path)

    record = {
        "id": sample_id,
        "error": error_msg,
        "raw_response_path": str(raw_path) if raw_response is not None else None,
        "metadata": metadata,
    }
    atomic_save_json(record, rec_path)
    return str(raw_path) if raw_response is not None else None, str(rec_path)


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

def worker(
    rank,
    world_size,
    df_records,
    shard_dir,
    temperature,
    model_name,
    ideal_svg_dir,
    ideal_svg_rendered_dir,
    failed_dir,
    render_timeout_sec,
):
    my_tasks = [df_records[i] for i in range(rank, len(df_records), world_size)]

    model_api = RubricClient()

    shard_path = Path(shard_dir) / f"rank_{rank}.jsonl"
    pbar = tqdm(my_tasks, desc=f"Process {rank}", disable=(rank != 0))

    for record in pbar:
        sample_id = str(record["id"])
        fname = safe_filename(sample_id)

        metadata = {
            "description": record["description"],
            "gt_code": record["gt_code"],
            "gt_img": record["gt_img"],
            "source_index": record.get("source_index"),
            "parquet_row_index": record.get("parquet_row_index"),
        }

        raw_content = None

        try:
            prompt_content = get_svg_rubric_generation_prompt(
                description=record["description"],
                reference_svg=None,
            )

            multimodal_input = [
                {"data_type": "text", "data_value": prompt_content}
            ]

            gen_kwargs = {
                "max_new_tokens": 18000,
                "temperature": temperature
            }

            response = model_api.multimodal_conversation(multimodal_input, model_name, gen_kwargs=gen_kwargs)
            if not response or not response[0].strip():
                raise ValueError("Empty API response")

            raw_content = response[0]
            clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip() or raw_content.strip()

            parsed = parse_generation_result(clean_content)
            if parsed is None:
                raise ValueError("Failed to parse generation result JSON")

            ideal_svg = normalize_svg_code(parsed.get("ideal_svg"))
            rubric_items = parsed.get("rubric_items")

            if ideal_svg is None:
                raise ValueError("Missing or invalid ideal_svg in response")

            if not isinstance(rubric_items, list):
                raise ValueError("rubric_items is not a list")

            if len(rubric_items) != 6:
                raise ValueError(
                    f"Expected 6 generated rubric items, got {len(rubric_items)}"
                )

            svg_path = str(Path(ideal_svg_dir) / f"{fname}.svg")
            png_path = str(Path(ideal_svg_rendered_dir) / f"{fname}.png")

            # 先渲染；只有渲染成功才保存到 outputdir
            try:
                render_svg_to_png_with_timeout(
                    ideal_svg,
                    png_path,
                    timeout_sec=render_timeout_sec,
                    background_color="white"
                )
                atomic_save_text(ideal_svg, svg_path)
            except Exception as e:
                remove_if_exists(svg_path)
                remove_if_exists(png_path)
                raise ValueError(f"SVG render failed: {e}")

            append_jsonl(shard_path, {
                "id": sample_id,
                "status": "ok",
                "ideal_svg_path": svg_path,
                "ideal_rendered_png_path": png_path,
                "rubric_items": rubric_items,
                "metadata": metadata
            })

        except Exception as e:
            raw_response_path, failed_record_path = save_failed_debug(
                failed_dir=failed_dir,
                sample_id=sample_id,
                raw_response=raw_content,
                error_msg=str(e),
                metadata=metadata
            )

            append_jsonl(shard_path, {
                "id": sample_id,
                "status": "error",
                "error": str(e),
                "raw_response_path": raw_response_path,
                "failed_record_path": failed_record_path,
                "metadata": metadata
            })


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate an ideal SVG and six-item rubric for each prompt.")

    # 输入输出
    parser.add_argument("--input_parquet", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/rubrics")

    # 抽样参数
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_manifest_name", type=str, default="sample_manifest.json")
    parser.add_argument("--sample_records_jsonl_name", type=str, default="sampled_records.jsonl")
    parser.add_argument("--sample_records_all_jsonl_name", type=str, default="sampled_records_all.jsonl")
    parser.add_argument("--sampled_but_not_rubric_ready_jsonl_name", type=str, default="sampled_but_not_rubric_ready.jsonl")
    parser.add_argument("--force_resample", action="store_true")

    # 结果文件
    parser.add_argument("--final_json_name", type=str, default="rubrics_sample_final.json")
    parser.add_argument("--failed_json_name", type=str, default="rubrics_sample_failed.json")
    parser.add_argument("--summary_json_name", type=str, default="rubrics_sample_summary.json")

    # 附加输出目录
    parser.add_argument("--ideal_svg_dirname", type=str, default="ideal_svg")
    parser.add_argument("--ideal_svg_rendered_dirname", type=str, default="ideal_svg_rendered")
    parser.add_argument("--failed_dirname", type=str, default="failed_cases")

    # 运行配置
    parser.add_argument("--proc_num", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--render_timeout_sec", type=int, default=3)
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.environ.get("RUBRIC_MODEL"),
        help="Model served by the configured endpoint (or set RUBRIC_MODEL).",
    )

    args = parser.parse_args()
    if not args.model_name:
        parser.error("Set RUBRIC_MODEL or pass --model_name.")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY for the model endpoint.")
    if args.proc_num < 1:
        parser.error("--proc_num must be at least 1.")

    if args.sample_size is not None and args.sample_ratio is not None:
        raise ValueError("Only one of --sample_size and --sample_ratio can be set.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_json_path = output_dir / args.final_json_name
    failed_json_path = output_dir / args.failed_json_name
    summary_json_path = output_dir / args.summary_json_name
    sample_manifest_path = output_dir / args.sample_manifest_name
    sample_records_jsonl_path = output_dir / args.sample_records_jsonl_name
    sample_records_all_jsonl_path = output_dir / args.sample_records_all_jsonl_name
    sampled_but_not_ready_jsonl_path = output_dir / args.sampled_but_not_rubric_ready_jsonl_name
    shard_dir = output_dir / f".tmp_{Path(args.final_json_name).stem}"

    ideal_svg_dir = output_dir / args.ideal_svg_dirname
    ideal_svg_rendered_dir = output_dir / args.ideal_svg_rendered_dirname
    failed_dir = output_dir / args.failed_dirname

    shard_dir.mkdir(parents=True, exist_ok=True)
    ideal_svg_dir.mkdir(parents=True, exist_ok=True)
    ideal_svg_rendered_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.input_parquet} ...")
    candidate_records, candidate_stats = load_rollout_aligned_records_from_parquet(args.input_parquet)

    print(f"Rollout-aligned candidate count   : {candidate_stats['total_candidates_after_rollout_filter']}")
    print(f"Skipped missing id/prompt         : {candidate_stats['skipped_missing_id_or_prompt']}")
    print(f"Missing gt fields in candidates   : {candidate_stats['missing_gt_fields_within_candidates']}")

    if len(candidate_records) == 0:
        raise ValueError("No rollout-aligned candidate records found.")

    seen = set()
    duplicate_id_count = 0
    for rec in candidate_records:
        sid = str(rec["id"])
        if sid in seen:
            duplicate_id_count += 1
        else:
            seen.add(sid)

    if duplicate_id_count > 0:
        print(f"[WARN] Found {duplicate_id_count} duplicate ids in rollout-aligned candidates.")
        print("[WARN] This script does NOT deduplicate, to preserve exact sampling alignment with rollout.")

    manifest, sampled_indices, sampled_ids = load_or_create_sample_manifest(
        manifest_path=sample_manifest_path,
        candidate_records=candidate_records,
        sample_size=args.sample_size,
        sample_ratio=args.sample_ratio,
        sample_seed=args.sample_seed,
        input_parquet=args.input_parquet,
        force_resample=args.force_resample,
    )

    sampled_records_all = []
    valid_index_max = len(candidate_records) - 1
    bad_indices = []

    for idx in sampled_indices:
        if 0 <= idx <= valid_index_max:
            sampled_records_all.append(candidate_records[idx])
        else:
            bad_indices.append(idx)

    if bad_indices:
        print(f"[WARN] {len(bad_indices)} sampled indices are out of range and will be ignored.")

    total_sampled_count = len(sampled_records_all)
    print(f"Total sampled records (strict rollout-aligned): {total_sampled_count}")

    save_records_jsonl(sampled_records_all, sample_records_all_jsonl_path)
    print(f"Saved all sampled records to: {sample_records_all_jsonl_path}")

    sampled_records = [x for x in sampled_records_all if x.get("is_rubric_ready", False)]
    sampled_not_ready = [x for x in sampled_records_all if not x.get("is_rubric_ready", False)]

    save_records_jsonl(sampled_records, sample_records_jsonl_path)
    save_records_jsonl(sampled_not_ready, sampled_but_not_ready_jsonl_path)

    print(f"Sampled rubric-ready records      : {len(sampled_records)}")
    print(f"Sampled but not rubric-ready      : {len(sampled_not_ready)}")
    print(f"Saved rubric-ready records to     : {sample_records_jsonl_path}")
    print(f"Saved non-ready records to        : {sampled_but_not_ready_jsonl_path}")

    sampled_id_set = set(str(x["id"]) for x in sampled_records)

    existing_success_map = {
        str(k): v
        for k, v in load_existing_success(final_json_path).items()
        if str(k) in sampled_id_set
    }

    # If final JSON is missing/empty but shards exist, recover from shards
    # (handles cases where workers were interrupted before merge step)
    if not existing_success_map:
        shard_dir_path = Path(shard_dir)
        if shard_dir_path.exists():
            print(f"Final JSON empty/missing; recovering completed IDs from shards in {shard_dir_path} ...")
            for shard_file in sorted(shard_dir_path.glob("rank_*.jsonl")):
                with open(shard_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        sid = str(obj.get("id", ""))
                        if sid and sid in sampled_id_set and obj.get("status") == "ok":
                            existing_success_map[sid] = obj
            if existing_success_map:
                print(f"Recovered {len(existing_success_map)} successful records from shards.")

    done_ids = set(existing_success_map.keys())
    pending_records = [x for x in sampled_records if str(x["id"]) not in done_ids]

    print(f"Already completed within sampled rubric-ready subset: {len(done_ids)}")
    print(f"Pending sampled rubric-ready records                : {len(pending_records)}")

    if len(pending_records) > 0:
        actual_proc_num = min(args.proc_num, len(pending_records))
        print(f"Starting {actual_proc_num} processes ...")

        mp.set_start_method("spawn", force=True)
        mp.spawn(
            worker,
            args=(
                actual_proc_num,
                pending_records,
                str(shard_dir),
                args.temperature,
                args.model_name,
                str(ideal_svg_dir),
                str(ideal_svg_rendered_dir),
                str(failed_dir),
                args.render_timeout_sec,
            ),
            nprocs=actual_proc_num
        )
    else:
        print("No pending sampled rubric-ready records. Skip generation.")

    print("\nMerging shard files and saving final outputs ...")
    success_map, failed_map, summary_obj = merge_shards_and_save(
        final_json_path=final_json_path,
        failed_json_path=failed_json_path,
        summary_json_path=summary_json_path,
        shard_dir=shard_dir,
        input_parquet=args.input_parquet,
        temperature=args.temperature,
        total_sampled_count=total_sampled_count,
        total_rubric_ready_count=len(sampled_records),
        sampled_id_set=sampled_id_set,
        sample_manifest_path=sample_manifest_path,
        candidate_stats=candidate_stats,
    )

    if not args.keep_tmp:
        shutil.rmtree(shard_dir, ignore_errors=True)

    print("Done!")
    print(f"Sample manifest                    : {sample_manifest_path}")
    print(f"Sampled records all jsonl          : {sample_records_all_jsonl_path}")
    print(f"Sampled rubric-ready records jsonl : {sample_records_jsonl_path}")
    print(f"Sampled not-ready records jsonl    : {sampled_but_not_ready_jsonl_path}")
    print(f"Ideal SVG dir                      : {ideal_svg_dir}")
    print(f"Ideal rendered PNG dir             : {ideal_svg_rendered_dir}")
    print(f"Failed debug dir                   : {failed_dir}")
    print(f"Final success file                 : {final_json_path}")
    print(f"Final failed file                  : {failed_json_path}")
    print(f"Final summary file                 : {summary_json_path}")
    print(f"Successful rubrics                 : {len(success_map)}")
    print(f"Failed samples                     : {len(failed_map)}")
    print(f"Success rate (rubric-ready)        : {summary_obj['success_rate_over_sampled_rubric_ready']:.4f}")
    print(f"Success rate (total sampled)       : {summary_obj['success_rate_over_total_sampled']:.4f}")


if __name__ == "__main__":
    main()
