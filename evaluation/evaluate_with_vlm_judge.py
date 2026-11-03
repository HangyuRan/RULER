#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate generated images with General Rubric Score using a VLM judge.

Usage:
    python evaluation/evaluate_with_vlm_judge.py \
        --results-json /path/to/results.json \
        --output-json /path/to/rubric_metrics.json \
        --num-workers 8

Configure OPENAI_API_KEY, OPENAI_BASE_URL, and VLM_JUDGE_MODEL, or pass their
corresponding command-line options. Sampling uses the API defaults.
"""

import os
import json
import base64
import argparse
import mimetypes
import time
import re
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__:
    from .judge_prompt import get_judge_prompt
else:
    from judge_prompt import get_judge_prompt


# ----------------------------------------------------------------------
# OpenAI-compatible VLM judge client
# ----------------------------------------------------------------------

class VLMJudgeClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass --api-key")
        if not base_url:
            raise ValueError("Set OPENAI_BASE_URL or pass --base-url")

        import requests
        self.requests = requests
        self.url = base_url.rstrip("/") + "/chat/completions"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _encode_image(self, image_path: str) -> Optional[str]:
        if not image_path or not os.path.exists(image_path):
            return None
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def multimodal_conversation(self, inputs: List[Dict[str, Any]], model_name: str, gen_kwargs: Optional[Dict] = None) -> List[str]:
        # parse input
        image_path = None
        prompt_text = ""

        for item in inputs:
            if item.get("data_type") == "image":
                image_path = item.get("data_value")
            elif item.get("data_type") == "text":
                prompt_text = item.get("data_value")

        if not prompt_text:
            return [""]

        image_base64 = None
        if image_path:
            image_base64 = self._encode_image(image_path)

        if gen_kwargs is None:
            gen_kwargs = {}

        max_tokens = gen_kwargs.get("max_new_tokens", 32768)

        content = []

        if prompt_text:
            content.append({
                "type": "text",
                "text": prompt_text
            })

        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mimetypes.guess_type(image_path)[0] or 'image/png'};base64,{image_base64}"
                }
            })

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_completion_tokens": max_tokens,
        }

        # request + retry
        max_retry = 5
        for attempt in range(max_retry):
            try:
                resp = self.requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=300,
                )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}"
                    )

                data = resp.json()

                # parse response
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("Empty choices")

                msg = choices[0].get("message", {})
                resp_content = msg.get("content", "")

                if not resp_content:
                    raise ValueError("Empty content")

                if isinstance(resp_content, list):
                    text = "".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in resp_content
                    )
                else:
                    text = str(resp_content)

                if not text.strip():
                    raise ValueError("Empty decoded text")

                return [text]

            except Exception as e:
                wait = min(2 ** attempt, 60)
                if attempt == max_retry - 1:
                    print(f"  Final Failure: {e}")
                    return [""]
                print(f"  Retry {attempt + 1}/{max_retry}: {str(e)[:100]}, wait {wait}s")
                time.sleep(wait)

        return [""]


# ----------------------------------------------------------------------
# JSON parse helpers
# ----------------------------------------------------------------------

def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_score_json(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None

    text = content.strip()
    # Remove think tags if present
    text = re.sub(r"\n\s*think\s*\n.*?\n\s*/think\s*\n", "\n", text, flags=re.DOTALL)
    text = strip_code_fence(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to extract JSON from text
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            obj = json.loads(text[left:right + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def extract_overall_score(parsed: Dict[str, Any]) -> Optional[float]:
    overall_score = parsed.get("overall_score")
    if overall_score is None:
        return None
    try:
        score = float(overall_score)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------

def load_results(results_json: str) -> Dict[str, Dict[str, List[str]]]:
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = {
        "icon": {"paths": [], "prompts": [], "ids": []},
        "illustration": {"paths": [], "prompts": [], "ids": []},
    }

    for sample in data.get("samples", []):
        if sample.get("status") != "success":
            continue
        sample_type = sample.get("type")
        png_path = sample.get("png_path")
        prompt = sample.get("prompt")
        sample_id = sample.get("id")

        if sample_type not in grouped or not png_path or not os.path.exists(png_path):
            continue

        grouped[sample_type]["paths"].append(png_path)
        grouped[sample_type]["prompts"].append(prompt)
        grouped[sample_type]["ids"].append(sample_id)

    return grouped


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

def _score_single_worker(args_tuple) -> Dict[str, Any]:
    idx, image_path, prompt_text, sample_id, model_name, api_key, base_url = args_tuple

    try:
        client = VLMJudgeClient(api_key=api_key, base_url=base_url)
        prompt_template = get_judge_prompt()
        full_prompt = prompt_template.replace("{user_prompt}", prompt_text)
        full_prompt += (
            "\n\nOutput JSON only. No extra explanation outside JSON. "
            "Keep the reason brief (1-2 short sentences)."
        )

        inputs = [
            {"data_type": "text", "data_value": full_prompt},
            {"data_type": "image", "data_value": image_path},
        ]

        # Keep the scoring response concise.
        gen_kwargs = {"max_new_tokens": 2048}
        result = client.multimodal_conversation(inputs, model_name, gen_kwargs)

        if not result or result[0] == "":
            return {
                "id": sample_id,
                "score": float("nan"),
                "status": "api_fail",
                "error": "API returned empty after retries",
            }

        raw_content = result[0]
        parsed = parse_score_json(raw_content)

        if parsed is None:
            return {
                "id": sample_id,
                "score": float("nan"),
                "status": "parse_fail",
                "error": f"Cannot parse JSON from response: {raw_content[:200]}",
            }

        score = extract_overall_score(parsed)
        if score is None:
            return {
                "id": sample_id,
                "score": float("nan"),
                "status": "no_score",
                "error": f"No overall_score in parsed JSON: {json.dumps(parsed)[:200]}",
            }

        return {
            "id": sample_id,
            "score": score,
            "status": "success",
            "error": None,
            "reason": parsed.get("reason", ""),
            "raw_response": raw_content,
        }

    except Exception as e:
        import traceback
        return {
            "id": sample_id,
            "score": float("nan"),
            "status": "exception",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "traceback": traceback.format_exc(),
        }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def evaluate_group(paths: List[str], prompts: List[str], ids: List[str], model_name: str, num_workers: int,
                   api_key=None, base_url=None) -> Dict[str, Any]:
    import numpy as np
    from tqdm import tqdm

    args_list = [
        (i, paths[i], prompts[i], ids[i], model_name, api_key, base_url)
        for i in range(len(paths))
    ]

    results = []
    with mp.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_score_single_worker, args_list),
            total=len(args_list),
            desc="VLM Rubric",
        ):
            results.append(result)

    valid_scores = [r["score"] for r in results if r["status"] == "success" and not np.isnan(r["score"])]
    mean_score = float(np.mean(valid_scores)) if valid_scores else float("nan")

    return {
        "GeneralRubricScore": mean_score,
        "num_scored": len(valid_scores),
        "num_total": len(paths),
        "per_sample": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved outputs with General Rubric using an OpenAI-compatible VLM judge"
    )
    parser.add_argument("--results-json", type=str, required=True,
                        help="Path to results.json containing samples with png_path")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Path to output JSON file for metrics")
    parser.add_argument("--model-name", type=str, default=os.environ.get("VLM_JUDGE_MODEL"),
                        help="Judge model name (default: VLM_JUDGE_MODEL)")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL"),
                        help="OpenAI-compatible API base URL, including /v1 (default: OPENAI_BASE_URL)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="API key (prefer setting OPENAI_API_KEY)")
    parser.add_argument("--metrics", type=str, default="rubric",
                        help="Comma-separated metrics to compute (default: rubric)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel API processes (default: 8)")
    parser.add_argument("--per-sample-json", type=str, default=None,
                        help="Optional: save per-sample details to this path prefix (will create _icon.jsonl and _illustration.jsonl)")

    args = parser.parse_args()
    for option, value, environment in [
        ("--model-name", args.model_name, "VLM_JUDGE_MODEL"),
        ("--base-url", args.base_url, "OPENAI_BASE_URL"),
        ("--api-key", args.api_key, "OPENAI_API_KEY"),
    ]:
        if not value:
            parser.error(f"{option} is required unless {environment} is set")

    requested_metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    grouped = load_results(args.results_json)

    output = {
        "results_json": args.results_json,
        "metrics_requested": requested_metrics,
        "model_name": args.model_name,
        "metrics": {},
    }

    for sample_type in ["icon", "illustration"]:
        paths = grouped[sample_type]["paths"]
        prompts = grouped[sample_type]["prompts"]
        ids_list = grouped[sample_type]["ids"]

        if not paths:
            print(f"\nNo successful {sample_type} samples found, skipping")
            output["metrics"][sample_type] = {}
            continue

        print("\n" + "=" * 60)
        print(f"{sample_type.upper()} VLM RUBRIC SCORE EVAL")
        print(f"Num images: {len(paths)}")
        print(f"Model: {args.model_name}")
        print(f"Workers: {args.num_workers}")
        print("=" * 60)

        metrics = {}

        if "rubric" in requested_metrics:
            try:
                result = evaluate_group(
                    paths=paths,
                    prompts=prompts,
                    ids=ids_list,
                    model_name=args.model_name,
                    num_workers=args.num_workers,
                    api_key=args.api_key,
                    base_url=args.base_url,
                )
                metrics["GeneralRubricScore"] = result["GeneralRubricScore"]
                metrics["num_scored"] = result["num_scored"]
                metrics["num_total"] = result["num_total"]

                print(f"GeneralRubricScore: {result['GeneralRubricScore']:.4f} "
                      f"(scored {result['num_scored']}/{result['num_total']})")

                # Save per-sample details if requested
                if args.per_sample_json:
                    per_sample_path = str(Path(args.per_sample_json).with_suffix("")) + f"_{sample_type}.jsonl"
                    Path(per_sample_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(per_sample_path, "w", encoding="utf-8") as f:
                        for r in result["per_sample"]:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"Saved per-sample details to: {per_sample_path}")

            except Exception as e:
                import traceback
                print(f"Rubric evaluation failed for {sample_type}: {e}")
                traceback.print_exc()
                metrics["GeneralRubricScore"] = float("nan")

        output["metrics"][sample_type] = metrics

    # Save output
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved VLM rubric eval results to: {output_path}")


if __name__ == "__main__":
    main()
