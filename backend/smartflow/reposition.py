"""重定位模块：根据用户实时反馈调整剩余候选的对数几率/权重。"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def apply_feedback_boost(
    logits: torch.Tensor,
    candidate_ids: Sequence[str],
    feedback: Dict[str, float],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    feedback: atom_id -> 用户显式评分 [-1, 1] 或 0/1 偏好。
    对 logits 做加性偏置：boost = scale * feedback[id]。
    """
    if not feedback:
        return logits
    device = logits.device
    bias = torch.zeros_like(logits)
    scale = 2.0 / max(temperature, 1e-6)
    for i, aid in enumerate(candidate_ids):
        if aid in feedback:
            bias[0, i] = scale * float(feedback[aid])
    return logits + bias


def mask_completed(logits: torch.Tensor, indices_done: List[int]) -> torch.Tensor:
    """已选位置置为 -inf。"""
    if not indices_done:
        return logits
    out = logits.clone()
    for j in indices_done:
        out[0, j] = float("-inf")
    return out
