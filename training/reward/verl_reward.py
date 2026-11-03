#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reward entry for sample-wise async reward loop.

Recommended VERL config:
  custom_reward_function.path=training/reward/verl_reward.py
  custom_reward_function.name=compute_score_fast
"""

import os
import sys
from typing import Any, Dict, Optional

if __package__:
    from .rubric_reward import compute_score_fast as _compute_score_fast_impl
else:
    # VERL loads custom reward files directly, without a package context.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rubric_reward import compute_score_fast as _compute_score_fast_impl


async def compute_score_fast(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, float]:
    return await _compute_score_fast_impl(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )



# Backward-compatible alias
compute_score_auto = compute_score_fast
