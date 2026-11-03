def get_svg_rubric_generation_prompt(description: str, reference_svg=None) -> str:
    prompt = f"""You are an expert SVG designer and an expert rubric writer for open-ended text-to-SVG evaluation.

Your task has two sequential stages:

Stage 1: Create one ideal SVG for the given prompt.
Stage 2: Generate an aligned, instance-specific rubric for judging other rendered SVG candidates for the same prompt.

The ideal SVG should establish a strong visual reference.
The rubric should convert that reference into open-ended evaluation criteria for reinforcement learning.

==================================================
CORE GOAL
==================================================

The goal is NOT merely to make the prompt recognizable.
The goal is to encourage polished, deliberate, visually refined SVG artwork.

Avoid:
- crude or placeholder-like drawings
- generic primitive-symbol assemblies
- rough sketches
- accidental ugliness
- visually empty outputs
- readable text labels used as shortcuts

Simplicity is allowed, but only when it looks intentional, balanced, and aesthetically designed.

==================================================
STAGE 1: IDEAL SVG GENERATION
==================================================

First, internally imagine a strong visual solution for the prompt and then write one ideal SVG.

Requirements for the ideal SVG:
- It must clearly communicate the prompt concept without using readable text labels.
- It should look like polished vector artwork.
- It should have clear design intention, clean contours, balanced proportions, and coherent graphic treatment.
- It should contain enough visual refinement to serve as a high-quality aesthetic exemplar.
- It should not be unnecessarily complex or cluttered.

This SVG is only one possible good solution.
It is NOT the only correct answer.

==================================================
STAGE 2: RUBRIC GENERATION
==================================================

Now generate a rubric for evaluating other rendered SVG candidates for the same prompt.

The rubric will be used as a reinforcement-learning reward signal.
It must reward candidates that are semantically correct, visually refined, deliberately composed, and aesthetically stronger than crude recognizable placeholders.

The rubric must remain open-ended:
- A candidate may score well with different shapes, layout, orientation, colors, or implementation details.
- Do NOT require exact reproduction of your ideal SVG.
- Do NOT turn the rubric into a reconstruction checklist.
- Do NOT require exact local details, exact geometry, exact placement, exact part counts, or exact colors unless explicitly required by the prompt.

The rubric must also be instance-specific:
- Items 1-2 must specify the visible semantic evidence needed for this prompt, including concept identity, major components, relationships, states, or distinctive cues when relevant.
- Items 3-6 must specify the visual form, composition, rendering, and style qualities expected for this prompt.
- Avoid criteria that could apply unchanged to any SVG prompt.
- Each item should identify visible qualities that would meaningfully distinguish stronger and weaker candidates for THIS prompt.
- The rubric should create useful score variance among multiple rollout candidates for the same prompt.

==================================================
ORTHOGONALITY REQUIREMENTS
==================================================

The rubric must contain EXACTLY 6 items in the fixed order below.

The items must be as orthogonal as possible:
- each item targets one distinct observable dimension
- each item is independently judgeable from visible evidence
- each item penalizes a different type of failure
- if two items could be justified using the same evidence, rewrite them until they are clearly separable

Do NOT include catch-all items such as:
- overall beauty
- overall quality
- overall polish
- overall visual appeal

Do NOT combine multiple axes in one item.

==================================================
FIXED RUBRIC ITEMS AND WEIGHTS
==================================================

1. Concept Identity and Overall Readability — weight 5
2. Major Components and Distinctive Cues — weight 5
3. Silhouette and Form Refinement — weight 5
4. Composition and Canvas Design — weight 4
5. Rendering Finish and Visual Craftsmanship — weight 5
6. Style Cohesion and Designed Visual Interest — weight 5

Item definitions:

1. Concept Identity and Overall Readability
- This is the high-level semantic readability item.
- Judge whether the overall prompt meaning and primary concept are visually understandable without relying on readable text labels.
- Do NOT turn this item into a checklist of parts, and do NOT discuss component richness, visual complexity, composition, rendering polish, or style quality.

2. Major Components and Distinctive Cues
- Judge whether the major objects, important parts, actions, states, relationships, or concept-distinguishing visual cues are clearly readable.
- If the prompt involves multiple objects or relations, this item should judge whether those components and relations are visually expressed; if the prompt involves a single subject, this item should judge whether its major visible components and distinctive identity cues are clear.
- Do NOT discuss shape elegance, composition quality, rendering cleanliness, visual polish, or style cohesion.

3. Silhouette and Form Refinement
- Judge the quality of the subject's main shapes, contours, proportions, and silhouette.
- Distinguish refined, intentional forms from crude blobs, awkward primitive shapes, stiff assembly, collapsed geometry, or weak silhouette design.
- Do NOT discuss composition, rendering artifacts, or style consistency.

4. Composition and Canvas Design
- Judge subject scale, placement, spacing, cropping, negative space, visual balance, and spatial hierarchy.
- Distinguish deliberately composed images from cramped, tiny, badly cropped, awkwardly placed, or spatially careless ones.
- Do NOT discuss shape quality, rendering cleanliness, or style consistency.

5. Rendering Finish and Visual Craftsmanship
- Judge execution quality and completion.
- Distinguish finished vector artwork from outputs with broken contours, disconnected parts, stray fragments, noisy marks, malformed geometry, unfinished regions, text shortcuts, or obvious rendering artifacts.
- Do NOT discuss composition, shape elegance, or style harmony.

6. Style Cohesion and Designed Visual Interest
- Judge whether the image has a coherent vector-art treatment and enough intentional visual interest for this prompt.
- Consider stroke/fill treatment, detail density, color or tone handling, internal structure, decorative accents, depth cues, or graphic rhythm when relevant.
- Distinguish designed artwork from bare placeholders, generic primitive-symbol drawings, visually empty outputs, or inconsistent graphic treatment.
- Do NOT discuss semantic correctness, composition, or basic rendering cleanliness.

==================================================
SCORING GUIDE REQUIREMENTS
==================================================

Each rubric item must include a "scoring_guide" field.

The scoring guide must:v
- describe how to assign continuous satisfaction scores from 0.0 to 1.0
- use score ranges rather than fixed anchor scores
- explicitly allow any decimal value within the ranges
- be specific to the current prompt and the current item
- define visible differences between excellent, strong, partial, weak, and absent satisfaction
- help distinguish multiple candidates from the same rollout group

Use this exact range structure:

"0.9-1.0: ...; 0.7-0.8: ...; 0.4-0.6: ...; 0.1-0.3: ...; 0.0: ..."

Important:
- Reserve 0.9-1.0 for excellent, visually convincing satisfaction with clear item-specific evidence.
- Use 0.7-0.8 only for clearly strong satisfaction; do not use it for merely acceptable, broadly recognizable, clean, or simple outputs.
- Use 0.4-0.6 for partial but visibly flawed satisfaction; mediocre, generic, or underdeveloped candidates should usually stay around 0.4-0.5.
- Use 0.1-0.3 for weak, ambiguous, fragmentary, primitive, or barely visible evidence.
- Use 0.0 for absent, contradicted, unreadable, or completely failed evidence.
- For Items 1-2, high scores require the prompt meaning, primary concept, major components, or distinctive cues to be visually clear without relying on readable text labels.
- A visually polished but semantically generic image should receive low scores on Items 1-2 if it only matches the broad category while missing visible components, relationships, states, or distinguishing cues needed for this prompt.
- Keep each scoring_guide concise; not a long explanation.

==================================================
WRITING RULES
==================================================

For the rubric:
- Return EXACTLY 6 items in the fixed order above.
- Each item must have a short, natural, sample-specific title.
- Each title should mention the prompt object or concept when natural.
- Each description must be exactly one sentence.
- Each description must be self-contained.
- Each description should positively state the high-quality standard, and may briefly contrast it with what should be penalized.
- Do not write generic empty phrases like "looks good", "well-formed", or "coherent style" without explaining what that means for this instance.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY two Markdown code blocks in this exact order.

First, output the ideal SVG in an xml code block:

```xml
<svg ...>
...
</svg>
```

Second, output the rubric items in a json code block:

```json
[
  {{
    "title": "...",
    "weight": 5,
    "description": "...",
    "scoring_guide": "0.9-1.0: ...; 0.7-0.8: ...; 0.4-0.6: ...; 0.1-0.3: ...; 0.0: ..."
  }}
]
```

Input prompt:
{description}
"""
    return prompt
