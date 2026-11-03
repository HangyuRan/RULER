"""Compute cosine similarity between an image and a text-file prompt."""

import argparse
from pathlib import Path


def calculate_clip_score(image_path, text_path, model_path="openai/clip-vit-base-patch32", device=None):
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    with open(text_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    inputs = processor(
        text=[caption],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()

    return similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_path", help="Text file containing the prompt")
    parser.add_argument("image_path", help="Image file to score")
    parser.add_argument("--model-path", default="openai/clip-vit-base-patch32",
                        help="Hugging Face model ID or local CLIP model directory")
    parser.add_argument("--device", default=None, help="Device (default: CUDA if available, otherwise CPU)")
    args = parser.parse_args()
    for path in (args.text_path, args.image_path):
        if not Path(path).is_file():
            parser.error(f"File not found: {path}")
    score = calculate_clip_score(args.image_path, args.text_path, args.model_path, args.device)
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
