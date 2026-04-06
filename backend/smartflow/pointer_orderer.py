"""
模块 B：基于 Transformer 编码器 + 指针式解码的序列构造（与 VON 同类的排序/选次结构）。
未加载 checkpoint 时为随机初始化，仅用于打通管线；可用 train_reinforce.py 在模拟数据上训练。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from smartflow.atoms import ATOM_REGISTRY, IOKind
from smartflow.features import transition_feasibility


class _Encoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.enc(x, src_key_padding_mask=mask)


class PointerOrderer(nn.Module):
    def __init__(
        self,
        node_dim: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.d_model = d_model
        self.proj = nn.Linear(node_dim, d_model)
        self.encoder = _Encoder(d_model, n_heads, n_layers)
        self.ctx0 = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.ctx0, std=0.02)
        self.score = nn.Linear(d_model * 2, 1)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes: [B, N, node_dim] -> logits [B, N] for first pick (简化单步；多步在外部循环)."""
        h = self.proj(nodes)
        h = self.encoder(h)
        ctx = self.ctx0.expand(h.size(0), -1, -1)
        ctx_rep = ctx.expand(-1, h.size(1), -1)
        pair = torch.cat([h, ctx_rep], dim=-1)
        return self.score(pair).squeeze(-1)

    def step_logits(
        self,
        nodes: torch.Tensor,
        picked_indices: List[int],
        last_picked_idx: Optional[int],
    ) -> torch.Tensor:
        """多步：用已选节点均值池化作为上下文（轻量重定位：上下文随选择更新）。"""
        h = self.proj(nodes)
        h = self.encoder(h)
        b = h.size(0)
        if not picked_indices:
            ctx = self.ctx0.expand(b, 1, -1)
        else:
            idx_t = torch.tensor(picked_indices, device=nodes.device, dtype=torch.long)
            sel = h[:, idx_t, :].mean(dim=1, keepdim=True)
            ctx = sel
        if last_picked_idx is not None:
            ctx = ctx + 0.25 * h[:, last_picked_idx : last_picked_idx + 1, :]
        ctx_rep = ctx.expand(-1, h.size(1), -1)
        pair = torch.cat([h, ctx_rep], dim=-1)
        return self.score(pair).squeeze(-1)


def load_orderer_checkpoint(path: Path, device: torch.device) -> PointerOrderer:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    node_dim = int(ckpt.get("node_dim", 32))
    m = PointerOrderer(node_dim=node_dim)
    m.load_state_dict(ckpt["state_dict"])
    m.to(device)
    m.eval()
    return m


def greedy_order(
    model: PointerOrderer,
    nodes: torch.Tensor,
    candidate_ids: Sequence[str],
    history_ids: Sequence[str],
    feedback: Optional[Dict[str, float]] = None,
    topk: int = 5,
    max_steps: int = 8,
    io_bonus_scale: float = 0.5,
) -> Tuple[List[str], torch.Tensor]:
    """
    自回归贪心排序，输出完整序列建议的前 max_steps 步及最后一步的 Top-K（用于推荐下一批）。
    在 logits 上叠加 I/O 可行性偏置，模拟「可运行性」先验。
    """
    from smartflow.reposition import apply_feedback_boost, mask_completed

    device = next(model.parameters()).device
    nodes = nodes.to(device)
    feedback = feedback or {}
    picked: List[int] = []
    last_idx: Optional[int] = None
    order_names: List[str] = []

    last_out = IOKind.NONE
    if history_ids:
        last = ATOM_REGISTRY.get(history_ids[-1])
        if last:
            last_out = last.output_kind

    for step in range(min(max_steps, len(candidate_ids))):
        logits = model.step_logits(nodes, picked, last_idx)
        logits = apply_feedback_boost(logits, candidate_ids, feedback)
        logits = mask_completed(logits, picked)

        feas = torch.zeros_like(logits)
        for i, aid in enumerate(candidate_ids):
            if i in picked:
                continue
            atom = ATOM_REGISTRY.get(aid)
            if atom:
                feas[0, i] = transition_feasibility(last_out, atom) * io_bonus_scale
        logits = logits + feas

        j = int(torch.argmax(logits, dim=-1).item())
        picked.append(j)
        last_idx = j
        aid = candidate_ids[j]
        order_names.append(aid)
        atom = ATOM_REGISTRY.get(aid)
        if atom:
            last_out = atom.output_kind

    # 最后一步的 Top-K 候选
    logits = model.step_logits(nodes, picked, last_idx)
    logits = apply_feedback_boost(logits, candidate_ids, feedback)
    logits = mask_completed(logits, picked)
    feas = torch.zeros_like(logits)
    for i, aid in enumerate(candidate_ids):
        if i in picked:
            continue
        atom = ATOM_REGISTRY.get(aid)
        if atom:
            feas[0, i] = transition_feasibility(last_out, atom) * io_bonus_scale
    logits = logits + feas

    k = min(topk, logits.size(1) - len(picked))
    if k <= 0:
        return order_names, logits
    vals, idx = torch.topk(logits[0], k=k)
    return order_names, vals
