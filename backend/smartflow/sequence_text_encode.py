"""将办公原子序列格式化为适合句向量模型的文本。"""

from __future__ import annotations

from typing import List

from smartflow.atoms import ATOM_REGISTRY


def atoms_sequence_to_text(atom_ids: List[str], *, sep: str = " → ") -> str:
    """技术 ID + 中文展示名，便于多语言句向量捕获语义。"""
    parts: List[str] = []
    for aid in atom_ids:
        t = ATOM_REGISTRY.get(aid)
        if t:
            parts.append(f"{aid}（{t.display_name}）")
        else:
            parts.append(aid)
    return sep.join(parts)


def atoms_sequence_to_linear_text(atom_ids: List[str]) -> str:
    """空格分隔：ID 与中文名交替，便于仅用拉丁/仅中文子词的编码器。"""
    parts: List[str] = []
    for aid in atom_ids:
        t = ATOM_REGISTRY.get(aid)
        if t:
            parts.extend([aid, t.display_name])
        else:
            parts.append(aid)
    return " ".join(parts)


def atom_to_text(atom_id: str, *, linear: bool = False) -> str:
    """单个动作的编码用文本（与序列中单步格式一致）。"""
    if linear:
        return atoms_sequence_to_linear_text([atom_id])
    return atoms_sequence_to_text([atom_id])
