#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate SVG icons with vLLM on the MMSVGBench text2svg task.

This script ONLY does generation + saving:
- vLLM batch inference via llm.generate()
- use chat template stored in SFT checkpoint
- extract SVG from response
- fix truncated SVG if missing </svg>
- render PNG
- save SVG / PNG
- save results.json

Important:
- Only runs icon task
- Adds the SVG instruction used for icon training before applying the chat template
- Uses the chat template saved in the SFT checkpoint directory
"""

import os
import io
import re
import gc
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cairosvg
from transformers import AutoTokenizer


# ============================================================================
# Constants
# ============================================================================

TASK_CONFIGS = {
    "icon": {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.03,
    },
    "illustration": {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.03,
    }
}

MAX_NEW_TOKENS = 8192
RENDER_SIZE = 512

TARGET_TASK_TYPE = "icon"
TARGET_OUTPUT_SUBDIR = "icons"


# ============================================================================
# vLLM / Tokenizer Loading
# ============================================================================

def load_model_vllm(
    model_path,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.9,
    max_model_len=16384,
):
    from vllm import LLM
    import vllm

    print(f"Loading vLLM model from: {model_path}")
    print(f"vLLM version: {vllm.__version__}")

    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    print(f"Using dtype: {dtype}")
    print(f"tensor_parallel_size: {tensor_parallel_size}")
    print(f"gpu_memory_utilization: {gpu_memory_utilization}")
    print(f"max_model_len: {max_model_len}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="mp",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    print("vLLM model loaded successfully!")
    return llm


def load_sft_tokenizer(model_path):
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise ValueError(
            f"No chat_template found in tokenizer loaded from: {model_path}\n"
            "Expected the SFT checkpoint to contain the Jinja chat template."
        )

    print("Tokenizer loaded successfully.")
    print("Found chat_template in checkpoint.")
    return tokenizer


# ============================================================================
# Prompt construction: MUST use SFT checkpoint template
# ============================================================================

def build_formatted_prompt(tokenizer, prompt: str) -> str:
    """
    Use the chat template stored in the SFT checkpoint.

    Wrap the description with the SVG instruction used for icon training
    and pass it as a single user turn.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "Generate an SVG illustration for the following description:\n\n"
                f"{prompt}\n\n"
                "Output only SVG code."
            ),
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted_prompt


def build_sampling_params(subtype):
    from vllm import SamplingParams

    config = TASK_CONFIGS.get(subtype, TASK_CONFIGS[TARGET_TASK_TYPE])

    return SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        repetition_penalty=config["repetition_penalty"],
        max_tokens=MAX_NEW_TOKENS,
    )


# ============================================================================
# SVG extraction / repair
# ============================================================================

def extract_svg_from_response(response_text):
    """Extract SVG code from model response, including truncated SVG."""
    if not response_text:
        return None

    svg_match = re.search(
        r'```(?:svg|xml)?\s*(<svg.*?</svg>)\s*```',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    if svg_match:
        return svg_match.group(1).strip()

    svg_match = re.search(
        r'<svg.*?</svg>',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    if svg_match:
        return svg_match.group(0).strip()

    if "<svg" in response_text:
        start = response_text.find("<svg")
        return response_text[start:].strip()

    return None


def fix_svg(svg_content: str) -> str:
    """
    Fix truncated SVG when response contains a starting <svg ...> but misses </svg>.
    Strategy:
    1. Remove the last incomplete tag fragment like '<path fill="red'
    2. Remove any incomplete trailing '</tag' fragment
    3. Close still-open tags in reverse order
    4. Ensure final </svg>
    """
    if not svg_content:
        return svg_content

    svg_content = svg_content.strip()

    if "<svg" in svg_content:
        svg_content = svg_content[svg_content.find("<svg"):]

    svg_content = re.sub(r"<[^>]*$", "", svg_content)
    svg_content = re.sub(r"</[a-zA-Z][a-zA-Z0-9:_-]*\s*$", "", svg_content)

    stack = []
    for match in re.finditer(r"<(/?)([a-zA-Z][a-zA-Z0-9:_-]*)\b[^<>]*?(/?)>", svg_content):
        is_close = match.group(1) == "/"
        tag = match.group(2)
        self_close = match.group(3) == "/"

        if is_close:
            for i in range(len(stack) - 1, -1, -1):
                if stack[i] == tag:
                    stack.pop(i)
                    break
        elif not self_close:
            stack.append(tag)

    while stack:
        tag = stack.pop()
        svg_content += f"</{tag}>"

    if not re.search(r"</svg>\s*$", svg_content, flags=re.IGNORECASE):
        lower = svg_content.lower()
        if "</svg>" in lower:
            end = lower.rfind("</svg>") + len("</svg>")
            svg_content = svg_content[:end]
        else:
            svg_content += "</svg>"

    return svg_content


def postprocess_response_text(response_text):
    if not hasattr(postprocess_response_text, "debug_count"):
        postprocess_response_text.debug_count = 0

    svg_code = extract_svg_from_response(response_text)
    was_fixed = False

    if svg_code is not None and "<svg" in svg_code and "</svg>" not in svg_code.lower():
        fixed_svg = fix_svg(svg_code)
        if fixed_svg != svg_code:
            was_fixed = True
        svg_code = fixed_svg

    if svg_code is None:
        postprocess_response_text.debug_count += 1
        if postprocess_response_text.debug_count <= 3:
            print(f"\n[DEBUG] Failed to extract SVG. Response (first 500 chars):")
            print(response_text[:500] if response_text else "None")
            print("---")
    else:
        postprocess_response_text.debug_count = 0

    return svg_code, response_text, was_fixed


def generate_batch_vllm(llm, tokenizer, prompts, subtypes, print_first_formatted_once=True):
    formatted_prompts = [build_formatted_prompt(tokenizer, p) for p in prompts]
    batch_sampling_params = [build_sampling_params(st) for st in subtypes]

    if print_first_formatted_once and not hasattr(generate_batch_vllm, "_printed_example"):
        generate_batch_vllm._printed_example = True
        print("\n" + "=" * 80)
        print("[DEBUG] Example formatted prompt from SFT chat template:")
        print(formatted_prompts[0][:2000])
        if len(formatted_prompts[0]) > 2000:
            print("... (truncated)")
        print("=" * 80 + "\n")

    outputs = llm.generate(
        formatted_prompts,
        sampling_params=batch_sampling_params,
        use_tqdm=False,
    )

    results = []
    for out in outputs:
        if out is None or not out.outputs:
            results.append((None, None, None, False))
            continue

        response_text = out.outputs[0].text
        svg_code, response_text, was_fixed = postprocess_response_text(response_text)
        results.append((svg_code, response_text, out.prompt, was_fixed))

    return results


# ============================================================================
# SVG Rendering
# ============================================================================

def render_svg_to_image(svg_str, size=RENDER_SIZE):
    try:
        if 'width=' not in svg_str and 'viewBox' not in svg_str:
            svg_str = svg_str.replace('<svg', f'<svg width="{size}" height="{size}"', 1)

        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=size,
            output_height=size
        )
        image_rgba = Image.open(io.BytesIO(png_data)).convert("RGBA")

        bg = Image.new("RGB", image_rgba.size, (255, 255, 255))
        bg.paste(image_rgba, mask=image_rgba.split()[3])

        return bg
    except Exception as e:
        print(f"Render error: {e}")
        return None


def is_valid_svg(svg_str, img):
    if not svg_str or len(svg_str) < 20:
        return False, "too_short"

    if "<svg" not in svg_str:
        return False, "no_svg_tag"

    if img is None:
        return False, "render_failed"

    img_array = np.array(img)
    mean_val = img_array.mean()

    if mean_val > 252:
        return False, "empty_image"

    return True, "ok"


# ============================================================================
# Main generation
# ============================================================================

def generate_svgs(
    model_path,
    data_path,
    output_dir,
    device="cuda",
    tensor_parallel_size=4,
    batch_size=16,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    resume_skip_existing=False,
    save_raw_response_txt=False,
    save_formatted_prompt_txt=False,
):
    output_dir = Path(output_dir)
    icons_dir = output_dir / TARGET_OUTPUT_SUBDIR
    raw_response_dir = output_dir / "raw_responses"
    formatted_prompt_dir = output_dir / "formatted_prompts"

    icons_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_response_txt:
        raw_response_dir.mkdir(parents=True, exist_ok=True)
    if save_formatted_prompt_txt:
        formatted_prompt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Dataset size: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    df = df[df["type"] == TARGET_TASK_TYPE].copy()
    print(f"Filtered dataset to only '{TARGET_TASK_TYPE}' type. Remaining samples: {len(df)}")

    tokenizer = load_sft_tokenizer(model_path)

    llm = load_model_vllm(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    results = {
        "config": {
            "model_path": str(model_path),
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "timestamp": datetime.now().isoformat(),
            "sampling_params": TASK_CONFIGS[TARGET_TASK_TYPE],
            "backend": "vllm_generate_with_sft_chat_template",
            "tensor_parallel_size": tensor_parallel_size,
            "batch_size": batch_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_new_tokens": MAX_NEW_TOKENS,
            "render_size": RENDER_SIZE,
            "resume_skip_existing": resume_skip_existing,
            "save_raw_response_txt": save_raw_response_txt,
            "save_formatted_prompt_txt": save_formatted_prompt_txt,
            "task_filter": "icon_only",
            "prompt_style": "raw_text_prompt_rendered_by_checkpoint_chat_template",
            "truncated_svg_fix_enabled": True,
        },
        "samples": [],
        "summary": {
            "success_count": 0,
            "fail_count": 0,
            "skipped_count": 0,
            "fixed_svg_count": 0,
        }
    }

    print("\n" + "=" * 60)
    print("Generating SVGs for icon task only...")
    print("=" * 60)

    pbar = tqdm(total=len(df), desc="Generating")

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]

        batch_prompts = []
        batch_subtypes = []
        batch_meta = []

        for local_idx, (_, row) in enumerate(batch_df.iterrows()):
            global_idx = start + local_idx
            sample_id = row["id"]
            prompt = row["text"]
            sample_type = row["type"]

            base_name = f"{global_idx:04d}_{str(sample_id)[:8]}"
            svg_path = icons_dir / f"{base_name}.svg"
            png_path = icons_dir / f"{base_name}.png"
            raw_txt_path = raw_response_dir / f"{base_name}.txt"
            formatted_prompt_path = formatted_prompt_dir / f"{base_name}.txt"

            if resume_skip_existing and svg_path.exists() and png_path.exists():
                results["samples"].append({
                    "id": sample_id,
                    "type": sample_type,
                    "prompt": prompt,
                    "status": "skipped_existing",
                    "svg_path": str(svg_path),
                    "png_path": str(png_path),
                })
                results["summary"]["skipped_count"] += 1
                continue

            batch_prompts.append(prompt)
            batch_subtypes.append(sample_type)
            batch_meta.append({
                "global_idx": global_idx,
                "sample_id": sample_id,
                "prompt": prompt,
                "sample_type": sample_type,
                "base_name": base_name,
                "svg_path": svg_path,
                "png_path": png_path,
                "raw_txt_path": raw_txt_path,
                "formatted_prompt_path": formatted_prompt_path,
            })

        if len(batch_prompts) == 0:
            pbar.update(len(batch_df))
            continue

        if save_formatted_prompt_txt:
            for meta, prompt in zip(batch_meta, batch_prompts):
                try:
                    formatted_prompt = build_formatted_prompt(tokenizer, prompt)
                    with open(meta["formatted_prompt_path"], "w", encoding="utf-8") as f:
                        f.write(formatted_prompt)
                except Exception as e:
                    print(f"[WARN] Failed to save formatted prompt for {meta['sample_id']}: {e}")

        try:
            batch_results = generate_batch_vllm(
                llm=llm,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                subtypes=batch_subtypes,
            )
        except Exception as e:
            print(f"\n[WARN] Batch generation failed for rows {start}:{start + len(batch_df)}: {e}")
            batch_results = []

            for prompt, subtype in zip(batch_prompts, batch_subtypes):
                try:
                    one_result = generate_batch_vllm(
                        llm=llm,
                        tokenizer=tokenizer,
                        prompts=[prompt],
                        subtypes=[subtype],
                        print_first_formatted_once=False,
                    )[0]
                except Exception as e_single:
                    print(f"[WARN] Single generation failed: {e_single}")
                    one_result = (None, None, None, False)
                batch_results.append(one_result)

        for meta, (svg_code, response, final_prompt_used, was_fixed) in zip(batch_meta, batch_results):
            sample_id = meta["sample_id"]
            prompt = meta["prompt"]
            sample_type = meta["sample_type"]
            svg_path = meta["svg_path"]
            png_path = meta["png_path"]
            raw_txt_path = meta["raw_txt_path"]

            if save_raw_response_txt and response is not None:
                try:
                    with open(raw_txt_path, "w", encoding="utf-8") as f:
                        f.write(response)
                except Exception as e:
                    print(f"[WARN] Failed to save raw response txt for {sample_id}: {e}")

            if svg_code:
                img = render_svg_to_image(svg_code)
                is_valid, reason = is_valid_svg(svg_code, img)
            else:
                img = None
                is_valid = False
                reason = "no_svg_in_response"

            if is_valid:
                try:
                    with open(svg_path, "w", encoding="utf-8") as f:
                        f.write(svg_code)
                    img.save(png_path)

                    results["samples"].append({
                        "id": sample_id,
                        "type": sample_type,
                        "prompt": prompt,
                        "status": "success",
                        "svg_path": str(svg_path),
                        "png_path": str(png_path),
                        "postprocess": "fixed_truncated_svg" if was_fixed else "none",
                    })
                    results["summary"]["success_count"] += 1
                    if was_fixed:
                        results["summary"]["fixed_svg_count"] += 1

                except Exception as e:
                    results["samples"].append({
                        "id": sample_id,
                        "type": sample_type,
                        "prompt": prompt,
                        "status": "failed",
                        "reason": f"save_failed: {e}",
                        "raw_response": response,
                        "formatted_prompt": final_prompt_used,
                        "postprocess": "fixed_truncated_svg" if was_fixed else "none",
                    })
                    results["summary"]["fail_count"] += 1
            else:
                results["samples"].append({
                    "id": sample_id,
                    "type": sample_type,
                    "prompt": prompt,
                    "status": "failed",
                    "reason": reason,
                    "raw_response": response,
                    "formatted_prompt": final_prompt_used,
                    "postprocess": "fixed_truncated_svg" if was_fixed else "none",
                })
                results["summary"]["fail_count"] += 1

        results_path = output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        pbar.update(len(batch_df))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()

    print("\n" + "=" * 60)
    print("GENERATION FINISHED")
    print("=" * 60)
    print(f"Success: {results['summary']['success_count']}")
    print(f"Failed:  {results['summary']['fail_count']}")
    print(f"Skipped: {results['summary']['skipped_count']}")
    print(f"Fixed:   {results['summary']['fixed_svg_count']}")

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save SVGs with vLLM.generate + SFT checkpoint chat template"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to SFT checkpoint / model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to MMSVGBench text2svg parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/icons",
        help="Output directory for generated SVGs and results",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Kept for compatibility")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs for vLLM TP")
    parser.add_argument("--batch_size", type=int, default=16, help="Offline batch size for vLLM.generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=16384, help="vLLM max model length")
    parser.add_argument("--resume_skip_existing", action="store_true", help="Skip samples whose SVG and PNG already exist")
    parser.add_argument("--save_raw_response_txt", action="store_true", help="Also save each raw response as a txt file")
    parser.add_argument("--save_formatted_prompt_txt", action="store_true", help="Also save the final prompt after applying checkpoint chat template")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    print("=" * 60)
    print("SVG generation with vLLM and the checkpoint chat template")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"TP size: {args.tensor_parallel_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU mem util: {args.gpu_memory_utilization}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Resume skip existing: {args.resume_skip_existing}")
    print(f"Save raw response txt: {args.save_raw_response_txt}")
    print(f"Save formatted prompt txt: {args.save_formatted_prompt_txt}")
    print("Task filter: icon only")
    print("Prompt style: SVG instruction -> checkpoint chat template")
    print("Truncated SVG auto-fix: enabled")
    print("=" * 60)

    _ = generate_svgs(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        resume_skip_existing=args.resume_skip_existing,
        save_raw_response_txt=args.save_raw_response_txt,
        save_formatted_prompt_txt=args.save_formatted_prompt_txt,
    )

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
