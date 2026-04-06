"""可运行办公链生成：基于 I/O 可行性随机游走，保证 ground truth 为合法序列。"""

from __future__ import annotations

import random
from typing import List, Set, Tuple

from smartflow.atoms import ATOM_REGISTRY, IOKind
from smartflow.features import transition_feasibility


def _feasible_next_atoms(prev_out: IOKind, used: Set[str], min_score: float = 0.5) -> List[str]:
    out: List[str] = []
    for aid, task in ATOM_REGISTRY.items():
        if aid in used:
            continue
        if transition_feasibility(prev_out, task) >= min_score:
            out.append(aid)
    return out


def random_valid_chain(rng: random.Random, min_len: int = 3, max_len: int = 8) -> List[str] | None:
    """从可起始原子出发随机扩展，无重复原子。"""
    starters = [a for a, t in ATOM_REGISTRY.items() if t.input_kind == IOKind.NONE]
    if not starters:
        return None
    target = rng.randint(min_len, max_len)
    first = rng.choice(starters)
    chain = [first]
    used = {first}
    prev_out = ATOM_REGISTRY[first].output_kind

    for _ in range(target - 1):
        nxt = _feasible_next_atoms(prev_out, used)
        if not nxt:
            break
        pick = rng.choice(nxt)
        chain.append(pick)
        used.add(pick)
        prev_out = ATOM_REGISTRY[pick].output_kind

    if len(chain) < min_len:
        return None
    return chain


# 手工模板：补充随机游走难以覆盖的典型业务链
CANONICAL_TEMPLATES: Tuple[Tuple[str, ...], ...] = (
    ("PDF_Import", "PDF_to_Word", "Summary_Generation", "Export_DOCX"),
    ("PDF_Import", "PDF_to_Word", "Export_DOCX"),
    ("Excel_Import", "Table_Normalize", "Data_Validate", "Chart_Generate"),
    ("Excel_Import", "Table_Normalize", "Schema_Check", "Data_Validate", "Chart_Generate"),
    ("Excel_Import", "Table_Normalize", "Schema_Check", "Chart_Generate"),
    ("Excel_Import", "Table_Normalize", "Data_Validate", "Schema_Check"),
    ("Code_Generate", "Summary_Generation", "Export_DOCX"),
    ("Excel_Import", "Table_Normalize", "Data_Validate", "Chart_Generate", "Schema_Check"),
)


def _feasible_atoms(prev_out: IOKind, min_score: float) -> List[str]:
    return [
        a
        for a, t in ATOM_REGISTRY.items()
        if transition_feasibility(prev_out, t) >= min_score
    ]


def random_long_feasible_chain(
    rng: random.Random,
    length: int,
    *,
    min_feas_low: float = 0.35,
    min_feas_high: float = 0.5,
    restart_stuck: bool = True,
) -> List[str]:
    """
    固定长度长链：允许同一原子多次出现（长办公流常见「校验—修正—再校验」）。
    在 I/O 可行集内随机游走；无路可走时可从 NONE 起点「开新子流程」以丰富模式。
    """
    starters = [a for a, t in ATOM_REGISTRY.items() if t.input_kind == IOKind.NONE]
    if not starters or length < 1:
        return []

    chain: List[str] = []
    prev_out = IOKind.NONE

    for _ in range(length):
        thr = rng.uniform(min_feas_low, min_feas_high)
        cands = _feasible_atoms(prev_out, thr)
        if not cands and thr > min_feas_low:
            cands = _feasible_atoms(prev_out, min_feas_low)
        if not cands:
            if restart_stuck:
                pick = rng.choice(starters)
                chain.append(pick)
                prev_out = ATOM_REGISTRY[pick].output_kind
            else:
                pick = rng.choice(list(ATOM_REGISTRY.keys()))
                chain.append(pick)
                prev_out = ATOM_REGISTRY[pick].output_kind
            continue

        if rng.random() < 0.12:
            strict = _feasible_atoms(prev_out, min_feas_high)
            pool = strict if strict else cands
        else:
            pool = cands
        pick = rng.choice(pool)
        chain.append(pick)
        prev_out = ATOM_REGISTRY[pick].output_kind

    return chain


def shuffled_permutation(rng: random.Random, gt: List[str]) -> List[str]:
    """打乱顺序；若长度>1 则保证与 GT 不同。"""
    if len(gt) <= 1:
        return list(gt)
    s = list(gt)
    for _ in range(64):
        rng.shuffle(s)
        if s != gt:
            return s
    # 理论上 n>=3 时几乎总能得到不同排列
    s.reverse()
    return s if s != gt else s[-1:] + s[:-1]
