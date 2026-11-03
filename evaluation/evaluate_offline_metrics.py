#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
import gc
import warnings
from pathlib import Path
from contextlib import nullcontext

warnings.filterwarnings("ignore", ".*output shape of zoom.*")


"""
python evaluation/evaluate_offline_metrics.py \
  --results-json outputs/icons/results.json \
  --output-json outputs/icons/metrics.json \
  --metrics clip,aesthetic,hps \
  --device cuda
"""


def load_results(results_json):
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

        if sample_type not in grouped:
            continue
        if not png_path or not os.path.exists(png_path):
            continue

        grouped[sample_type]["paths"].append(png_path)
        grouped[sample_type]["prompts"].append(prompt)
        grouped[sample_type]["ids"].append(sample_id)

    return grouped


def compute_clip_score(image_paths, prompts, model_path, device="cuda", batch_size=32):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import CLIPProcessor, CLIPModel

    print("Computing CLIP scores...")
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()

    all_scores = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP"):
        batch_paths = image_paths[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]

        images = []
        valid_prompts = []
        for p, t in zip(batch_paths, batch_prompts):
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_prompts.append(t)
            except Exception as e:
                print(f"CLIP image load error for {p}: {e}")

        if not images:
            continue

        inputs = processor(
            text=valid_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            scores = (image_features * text_features).sum(dim=-1)
            all_scores.extend(scores.detach().cpu().tolist())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(np.mean(all_scores)) if all_scores else float("nan")


def compute_aesthetic_score(image_paths, device="cuda", batch_size=32,
                            aesthetic_model_path=None, clip_model_path="ViT-L/14"):
    import clip
    import numpy as np
    import torch
    import urllib.request
    import torch.nn as nn
    from PIL import Image
    from tqdm import tqdm

    print("Computing Aesthetic scores...")

    if aesthetic_model_path is None:
        aesthetic_model_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        aesthetic_model_path = os.path.join(os.path.expanduser("~"), ".cache", "aesthetic_predictor", "sac+logos+ava1-l14-linearMSE.pth")
        if not os.path.exists(aesthetic_model_path):
            os.makedirs(os.path.dirname(aesthetic_model_path), exist_ok=True)
            print("Downloading aesthetic predictor model...")
            urllib.request.urlretrieve(aesthetic_model_url, aesthetic_model_path)

    class MLP(torch.nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.layers(x)

    aesthetic_model = MLP(768)
    state_dict = torch.load(aesthetic_model_path, map_location=device, weights_only=True)
    aesthetic_model.load_state_dict(state_dict)
    aesthetic_model.to(device)
    aesthetic_model.eval()

    clip_model, preprocess = clip.load(clip_model_path, device=device)
    clip_model.eval()

    all_scores = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Aesthetic"):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(preprocess(img))
            except Exception as e:
                print(f"Aesthetic image load error for {p}: {e}")

        if not batch_tensors:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(batch_tensor)
            im_emb_arr = image_features.detach().cpu().numpy()
            im_emb_arr = im_emb_arr / np.linalg.norm(im_emb_arr, axis=-1, keepdims=True)

            score_tensor = aesthetic_model(torch.from_numpy(im_emb_arr).to(device).float())
            all_scores.extend(score_tensor.squeeze(-1).detach().cpu().tolist())

        del batch_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del aesthetic_model, clip_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(np.mean(all_scores)) if all_scores else float("nan")


def ensure_hps_bpe_vocab(local_bpe_vocab_path):
    """
    hpsv2/open_clip expects bpe_simple_vocab_16e6.txt.gz in its package directory.
    If missing, copy from the provided local path.
    """
    if not local_bpe_vocab_path:
        return

    if not os.path.exists(local_bpe_vocab_path):
        raise FileNotFoundError(f"Local HPS BPE vocab not found: {local_bpe_vocab_path}")

    import hpsv2
    hps_root = os.path.dirname(hpsv2.__file__)
    target_path = os.path.join(hps_root, "src", "open_clip", "bpe_simple_vocab_16e6.txt.gz")

    if not os.path.exists(target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copyfile(local_bpe_vocab_path, target_path)
        print(f"Copied local HPS BPE vocab to: {target_path}")
    else:
        print(f"HPS BPE vocab already present: {target_path}")


def compute_hps_score(image_paths, prompts, hps_model_path=None, local_bpe_vocab_path=None,
                      device="cuda", batch_size=16, clip_model_path="laion2B-s32B-b79K",
                      hps_version="v2.0"):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    print("Computing HPS scores...")

    try:
        ensure_hps_bpe_vocab(local_bpe_vocab_path)
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    except ImportError:
        print("HPSv2 not installed, skipping HPS score")
        return float("nan")

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        clip_model_path,
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )

    if hps_model_path is None:
        from huggingface_hub import hf_hub_download
        from hpsv2.utils import hps_version_map

        hps_model_path = hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])

    if os.path.exists(hps_model_path):
        checkpoint = torch.load(hps_model_path, map_location=device)
    else:
        raise FileNotFoundError(f"HPS checkpoint not found: {hps_model_path}")

    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    all_scores = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="HPS"):
        batch_paths = image_paths[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]

        images = []
        valid_prompts = []
        for p, t in zip(batch_paths, batch_prompts):
            try:
                images.append(preprocess_val(Image.open(p).convert("RGB")))
                valid_prompts.append(t)
            except Exception as e:
                print(f"HPS image load error for {p}: {e}")

        if not images:
            continue

        image_input = torch.stack(images).to(device)
        text_input = tokenizer(valid_prompts).to(device)

        with torch.no_grad():
            with autocast_ctx:
                outputs = model(image_input, text_input)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                scores = torch.diagonal(logits_per_image)
                all_scores.extend(scores.detach().cpu().tolist())

        del image_input, text_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(np.mean(all_scores)) if all_scores else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved outputs using offline metrics")
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metrics", type=str, default="clip,aesthetic,hps")

    parser.add_argument("--batch-size-clip", type=int, default=32)
    parser.add_argument("--batch-size-aesthetic", type=int, default=32)
    parser.add_argument("--batch-size-hps", type=int, default=16)

    parser.add_argument("--clip-model-path", type=str,
                        default="openai/clip-vit-base-patch32",
                        help="Hugging Face model ID or local CLIP directory")
    parser.add_argument("--aesthetic-model-path", type=str, default=None,
                        help="Local aesthetic predictor checkpoint (default: download public weights)")
    parser.add_argument("--aesthetic-clip-model-path", type=str, default="ViT-L/14",
                        help="CLIP model name or local checkpoint for aesthetic scoring")
    parser.add_argument("--hps-model-path", type=str, default=None,
                        help="Local HPS checkpoint (default: download public weights)")
    parser.add_argument("--hps-version", type=str, default="v2.0",
                        help="HPS checkpoint version when --hps-model-path is not supplied")
    parser.add_argument("--hps-clip-model-path", type=str, default="laion2B-s32B-b79K",
                        help="OpenCLIP pretrained name or local checkpoint for HPS")
    parser.add_argument("--hps-bpe-vocab-path", type=str, default=None,
                        help="Optional local BPE vocabulary to install if the HPS package lacks it")

    args = parser.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            args.device = "cpu"

    requested_metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    grouped = load_results(args.results_json)

    output = {
        "results_json": args.results_json,
        "device": args.device,
        "metrics_requested": requested_metrics,
        "metrics": {},
    }

    for sample_type in ["icon", "illustration"]:
        paths = grouped[sample_type]["paths"]
        prompts = grouped[sample_type]["prompts"]

        if not paths:
            print(f"No successful {sample_type} samples found, skipping")
            continue

        print("\n" + "=" * 60)
        print(f"{sample_type.upper()} OFFLINE EVAL")
        print(f"Num images: {len(paths)}")
        print("=" * 60)

        metrics = {}

        if "clip" in requested_metrics:
            try:
                clip_value = compute_clip_score(
                    image_paths=paths,
                    prompts=prompts,
                    model_path=args.clip_model_path,
                    device=args.device,
                    batch_size=args.batch_size_clip,
                )
                metrics["CLIP"] = float(clip_value)
                print(f"CLIP: {clip_value:.4f}")
            except Exception as e:
                print(f"CLIP failed for {sample_type}: {e}")
                metrics["CLIP"] = float("nan")

        if "aesthetic" in requested_metrics:
            try:
                aesthetic_value = compute_aesthetic_score(
                    image_paths=paths,
                    device=args.device,
                    batch_size=args.batch_size_aesthetic,
                    aesthetic_model_path=args.aesthetic_model_path,
                    clip_model_path=args.aesthetic_clip_model_path,
                )
                metrics["Aesthetic"] = float(aesthetic_value)
                print(f"Aesthetic: {aesthetic_value:.4f}")
            except Exception as e:
                print(f"Aesthetic failed for {sample_type}: {e}")
                metrics["Aesthetic"] = float("nan")

        if "hps" in requested_metrics:
            try:
                hps_value = compute_hps_score(
                    image_paths=paths,
                    prompts=prompts,
                    hps_model_path=args.hps_model_path,
                    local_bpe_vocab_path=args.hps_bpe_vocab_path,
                    device=args.device,
                    batch_size=args.batch_size_hps,
                    clip_model_path=args.hps_clip_model_path,
                    hps_version=args.hps_version,
                )
                metrics["HPS"] = float(hps_value)
                print(f"HPS: {hps_value:.4f}")
            except Exception as e:
                print(f"HPS failed for {sample_type}: {e}")
                metrics["HPS"] = float("nan")

        output["metrics"][sample_type] = metrics

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved offline eval results to: {output_path}")


if __name__ == "__main__":
    main()
