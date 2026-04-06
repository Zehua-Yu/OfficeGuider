"""模块 A：任务特征向量 [N, dim]，供排序网络使用。"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from smartflow.atoms import ATOM_REGISTRY, AtomTask, IOKind, user_preference_vector

# one-hot 长度：与 IOKind 枚举顺序一致
_IO_KINDS = list(IOKind)


def _io_one_hot(kind: IOKind) -> np.ndarray:
    v = np.zeros(len(_IO_KINDS), dtype=np.float32)
    idx = _IO_KINDS.index(kind)
    v[idx] = 1.0
    return v


def atom_to_feature_vector(
    atom_id: str,
    preference_weight: float = 0.0,
    dim: int = 32,
) -> np.ndarray:
    """将单个原子映射到固定维向量（不足则零填充）。"""
    t = ATOM_REGISTRY.get(atom_id)
    if t is None:
        raw = np.zeros(dim, dtype=np.float32)
        raw[0] = 1.0  # unknown marker
        return raw
    base: List[float] = [
        t.complexity,
        min(t.est_duration_sec / 300.0, 1.0),
        preference_weight,
        float(len(t.tags)) / 5.0,
        float(len(t.requires)) / 3.0,
    ]
    base.extend(_io_one_hot(t.input_kind).tolist())
    base.extend(_io_one_hot(t.output_kind).tolist())
    arr = np.array(base, dtype=np.float32)
    if arr.size >= dim:
        return arr[:dim]
    out = np.zeros(dim, dtype=np.float32)
    out[: arr.size] = arr
    return out


def build_context_tensor(
    candidate_ids: Sequence[str],
    history_ids: Sequence[str],
    *,
    dim: int = 32,
) -> torch.Tensor:
    """候选任务特征矩阵，形状 [1, N, dim]。"""
    hist_counts: Dict[str, float] = {}
    for h in history_ids:
        hist_counts[h] = hist_counts.get(h, 0.0) + 1.0
    prefs = user_preference_vector(list(candidate_ids), hist_counts)
    rows = [atom_to_feature_vector(aid, prefs.get(aid, 0.0), dim=dim) for aid in candidate_ids]
    x = np.stack(rows, axis=0)
    return torch.from_numpy(x).float().unsqueeze(0)


def transition_feasibility(prev_output: IOKind, task: AtomTask) -> float:
    """简单 I/O 衔接可行性：0–1。"""
    if task.input_kind == IOKind.NONE:
        return 1.0
    if prev_output == task.input_kind:
        return 1.0
    if prev_output == IOKind.TEXT and task.input_kind in (IOKind.TEXT, IOKind.JSON):
        return 0.85
    if prev_output == IOKind.JSON and task.input_kind == IOKind.JSON:
        return 1.0
    if prev_output == IOKind.DOCX and task.input_kind == IOKind.TEXT:
        return 0.9
    if prev_output == IOKind.PDF and task.input_kind == IOKind.PDF:
        return 1.0
    return 0.35
