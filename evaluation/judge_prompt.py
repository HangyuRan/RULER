def get_judge_prompt() -> str:
    prompt_template = """You are an expert visual evaluator for open-ended text-to-SVG generation.

You will evaluate a rendered image produced from SVG code.

You are given:
- the user prompt
- the candidate image

User prompt:
"{user_prompt}"

Your task is to assign a SINGLE overall satisfaction score (0.0 to 1.0) based on a comprehensive evaluation of the image.

==================================================
THE 6 EVALUATION DIMENSIONS
==================================================

To determine the overall score, you must evaluate the image across these 6 specific dimensions. 
The goal is to reward polished, deliberate, visually refined SVG artwork, and penalize crude primitive assemblies or rendering artifacts.

1. Concept Identity: Is the overall prompt meaning and primary concept visually understandable without relying on readable text labels?
2. Major Components: Are the specific major objects, important parts, states, or distinctive visual cues required by the prompt clearly present?
3. Silhouette and Form Refinement: Are the main shapes, contours, and proportions refined and intentional? (Penalize crude blobs or awkward primitive shape assemblies).
4. Composition and Canvas Design: Is the subject properly scaled, well-placed, and spatially balanced within the canvas? (Penalize cramped, badly cropped, or spatially careless layouts).
5. Rendering Finish and Visual Craftsmanship: Is the vector execution clean and finished? (Penalize broken contours, disconnected parts, stray fragments, or visual noise).
6. Style Cohesion and Designed Visual Interest: Does the image have a coherent vector-art aesthetic and intentional design? (Penalize visually empty outputs, generic placeholders, or inconsistent graphic treatments).

==================================================
HOLISTIC SCORING GUIDE (CRITICAL CALIBRATION)
==================================================

Based on the 6 dimensions above, assign your final continuous decimal score (e.g., 0.85, 0.55) using this strict calibration guide:

0.9-1.0 (Excellent & Refined):
- Outstanding across ALL 6 dimensions.
- The concept is instantly readable, forms are beautifully refined, rendering is flawless, and the aesthetic is highly cohesive.
- RESERVED ONLY for convincing, high-quality vector artwork.

0.7-0.8 (Strong but not exceptional):
- Strong performance across most dimensions.
- Concept is clear (Dims 1-2) and execution is solid (Dims 3-5). 
- May lack exceptional stylistic flair (Dim 6) or have very minor imperfections in form or composition.

0.4-0.6 (Mediocre / Baseline):
- Partially satisfies the concept (Dims 1-2), BUT suffers from obvious flaws in Dims 3-6.
- Typically features crude primitive shapes, awkward spatial layout, lack of finish, or a highly generic "clipart" feel.
- If an image is merely "not blank" and "roughly recognizable", it belongs HERE. Do not inflate scores above 0.65 for basic, unrefined shape assemblies.

0.1-0.3 (Weak / Flawed):
- Fails significantly across multiple dimensions.
- Barely hints at the concept, or suffers from severe issues like collapsed geometry, completely broken rendering, or extreme spatial imbalance.

0.0 (Failed):
- Completely unrecognizable, unrelated to the prompt, or visual garbage.

==================================================
EVALUATION RULES
==================================================

1. Continuous Scoring: Use any precise decimal within the bands (e.g., 0.45, 0.72). 
2. Anti-Inflation: Do NOT default to high scores just because the model successfully rendered *something*. 
3. No Text Shortcuts: If the image relies on readable text labels to communicate the prompt instead of drawing the concept, severely penalize the overall score.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY a valid JSON object.
Important: You must write the `reason` BEFORE the `overall_score`. In your reason, explicitly comment on how the image performs across the 6 dimensions (Concept, Components, Form, Composition, Rendering, Style) before concluding the final score.

{{
  "reason": "Step-by-step evaluation of the 6 dimensions, followed by the justification for the final score band.",
  "overall_score": 0.0
}}
"""
    return prompt_template
