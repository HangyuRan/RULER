#!/usr/bin/env python3
"""
Judge prompt and formatting that incorporates scoring_guide from rubric generation.
Uses continuous (0.0-1.0) scoring with per-item band anchors.
"""


def get_rubric_judging_prompt() -> str:
    prompt_template = """You are an expert visual evaluator for a text-to-SVG generation task.

You will evaluate a rendered image produced from SVG code.

You are given:
- the user prompt
- the candidate image
- a rubric with several evaluation items

User prompt:
"{{user_prompt}}"

Rubric:
{{rubric_items_formatted}}

Evaluate the candidate image separately for each rubric item.

==================================================
EVALUATION RULES
==================================================

1. Strictly Follow the Rubric
- The rubric is the complete evaluation standard.
- Do not introduce requirements that are not stated in the current rubric item.
- Base your judgment ONLY on visible evidence in the image. Do not guess intent.

2. Continuous Band Scoring
- Each rubric item includes a `scoring_guide` defining continuous score bands (e.g., 0.9-1.0, 0.4-0.6).
- First, map the visible evidence to the correct score band.
- Then, assign a precise decimal value (e.g., 0.85, 0.65, 0.45) that reflects the exact execution quality within that tier.
- Do NOT default to high scores just because the image is not blank or broadly recognizable.

3. Score Calibration (CRITICAL — prevents score inflation)
- 0.9-1.0 is RESERVED. Give these scores only when the evidence clearly and fully matches the highest band with no visible weakness.
- 0.7-0.8 means "strong but not exceptional." A candidate with minor flaws should land here, not above.
- 0.45-0.6 is the MEDIOCRE zone — the image partially satisfies the item but has obvious limitations. This should be the most common band for zero-shot baselines.
- 0.2-0.4 means weak or flawed. Use this generously when evidence is thin, ambiguous, or generic.
- 0.0-0.15 means absent, contradicted, or failed.
- Do NOT inflate scores. "Not blank" and "roughly recognizable" do NOT justify scores above 0.6.
- If most of your scores for a batch fall above 0.7, you are inflating. Re-calibrate downward.

4. Independent Scoring for Each Rubric Item
- Score each rubric item independently.
- Do not let the score of one item affect the score of any other item.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY a valid JSON object.
Important: You must write the `reason` BEFORE the `satisfaction` score to ensure proper step-by-step evaluation.

{
  "items": [
    {
      "item_index": 1,
      "title": "Item Title",
      "reason": "Briefly evaluate the visible evidence against the scoring guide to justify your band selection.",
      "satisfaction": 0.0
    }
  ]
}
"""
    return prompt_template


def format_rubric_items_for_judging(rubric_items: list) -> str:
    """
    Format rubric items into a numbered list for VLM judging.
    Includes scoring_guide if present.
    """
    formatted = []
    for i, item in enumerate(rubric_items):
        title = item.get("title", f"Item {i+1}")
        desc = item.get("description", "")
        sg = item.get("scoring_guide", "")

        weight = item.get("weight", 0)
        if isinstance(weight, float) and weight.is_integer():
            weight = int(weight)

        parts = [
            f"{i+1}. {title}",
            f"   Weight: {weight}",
            f"   Description: {desc}",
        ]
        if sg:
            parts.append(f"   Scoring Guide: {sg}")

        formatted.append("\n".join(parts))

    return "\n\n".join(formatted)
