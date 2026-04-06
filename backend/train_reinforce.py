"""
可选：在模拟办公序列上 REINFORCE 训练 PointerOrderer（与 VON 同为策略梯度思路）。
完整对齐官方 VON 训练请使用 third_party/VON 目录内 run.py，并扩展 problem_order 中的 mission。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from smartflow.atoms import all_atom_ids
from smartflow.features import build_context_tensor
from smartflow.pointer_orderer import PointerOrderer
from smartflow.validator import combined_reward, kendall_tau_like, sequence_runnable

# 简易金标准：随机抽一条「模板链」作为 episode 目标
_TEMPLATES: List[List[str]] = [
    ["PDF_Import", "PDF_to_Word", "Summary_Generation", "Export_DOCX"],
    ["Excel_Import", "Table_Normalize", "Data_Validate", "Chart_Generate"],
]


def sample_episode() -> Tuple[List[str], List[str]]:
    gold = random.choice(_TEMPLATES)
    pool = all_atom_ids()
    history_len = random.randint(0, min(2, len(gold) - 1))
    history = gold[:history_len]
    return history, gold


def rollout_greedy(model: PointerOrderer, history: List[str], gold: List[str], max_add: int = 4):
    device = next(model.parameters()).device
    pool = all_atom_ids()
    remaining = [a for a in pool if a not in set(history)]
    nodes = build_context_tensor(remaining, history, dim=model.node_dim).to(device)
    model.eval()
    picked: List[int] = []
    last_idx = None
    seq = list(history)
    log_probs: List[torch.Tensor] = []

    from smartflow.atoms import ATOM_REGISTRY, IOKind
    from smartflow.features import transition_feasibility

    last_out = IOKind.NONE
    if history:
        la = ATOM_REGISTRY.get(history[-1])
        if la:
            last_out = la.output_kind

    for _ in range(min(max_add, len(remaining))):
        logits = model.step_logits(nodes, picked, last_idx)
        for i in picked:
            logits[0, i] = float("-inf")
        feas = torch.zeros_like(logits)
        for i, aid in enumerate(remaining):
            if i in picked:
                continue
            at = ATOM_REGISTRY.get(aid)
            if at:
                feas[0, i] = transition_feasibility(last_out, at) * 0.5
        logits = logits + feas
        dist = torch.distributions.Categorical(logits=torch.log_softmax(logits, dim=-1))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        j = int(action.item())
        picked.append(j)
        last_idx = j
        aid = remaining[j]
        seq.append(aid)
        at = ATOM_REGISTRY.get(aid)
        if at:
            last_out = at.output_kind

    run_ok, run_score = sequence_runnable(seq)
    dist_order = kendall_tau_like(seq, gold)
    r = combined_reward(True, run_ok, run_score, dist_order)
    return seq, r, log_probs


def train(steps: int = 200, lr: float = 1e-3, save: Path = Path("checkpoints/orderer.pt")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointerOrderer(node_dim=32).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for t in range(steps):
        model.train()
        hist, gold = sample_episode()
        _, reward, log_probs = rollout_greedy(model, hist, gold)
        if not log_probs:
            continue
        lp = torch.stack(log_probs).sum()
        loss = -(reward * lp)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (t + 1) % 50 == 0:
            print(f"step {t+1} reward~ {reward:.3f}")

    save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "node_dim": 32}, save)
    print(f"saved {save}")


if __name__ == "__main__":
    train()
