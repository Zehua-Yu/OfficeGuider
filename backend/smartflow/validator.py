"""模块 C：输出校验（WPS Schema Stub）与序列可行性。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from smartflow.atoms import ATOM_REGISTRY, IOKind


def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_legacy_wps_output(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    将旧版 { version, rows } 自动包一层 sheets，便于与增强版 wps_table_v1 共存。
    """
    if instance.get("sheets"):
        return instance
    rows = instance.get("rows")
    if not isinstance(rows, list) or not rows:
        return instance
    ncols = 0
    for r in rows:
        cells = r.get("cells") if isinstance(r, dict) else None
        if isinstance(cells, list):
            ncols = max(ncols, len(cells))
    ncols = max(ncols, 1)
    columns = [{"key": f"col_{i}", "dtype": "string", "required": False} for i in range(ncols)]
    norm_rows = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        rid = r.get("id", "row")
        cells = r.get("cells")
        if not isinstance(cells, list):
            cells = []
        norm_rows.append({"id": str(rid), "cells": cells})
    return {
        **{k: v for k, v in instance.items() if k != "rows"},
        "sheets": [
            {
                "sheet_id": "main",
                "name": "Main",
                "columns": columns,
                "rows": norm_rows,
            }
        ],
    }


def validate_ai_output(instance: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    inst = normalize_legacy_wps_output(instance)
    try:
        Draft202012Validator(schema).validate(inst)
        return True, None
    except ValidationError as e:
        return False, e.message


def sequence_runnable(atom_ids: List[str]) -> Tuple[bool, float]:
    """
    简易可运行性：按 I/O 链检查相邻任务，返回 (是否全部可衔接, 平均可行性分)。
    """
    if not atom_ids:
        return True, 1.0
    from smartflow.features import transition_feasibility

    last_out = IOKind.NONE
    scores: List[float] = []
    for aid in atom_ids:
        t = ATOM_REGISTRY.get(aid)
        if not t:
            scores.append(0.0)
            continue
        s = transition_feasibility(last_out, t)
        scores.append(s)
        last_out = t.output_kind
    avg = sum(scores) / max(len(scores), 1)
    ok = all(x >= 0.5 for x in scores)
    return ok, avg


def kendall_tau_like(pred: List[str], gold: List[str]) -> float:
    """归一化逆序距离：0 最好，1 最差（仅考虑 gold 中出现的元素）。"""
    pred = [x for x in pred if x in gold]
    if len(gold) <= 1:
        return 0.0
    pos = {g: i for i, g in enumerate(gold)}
    inv = 0
    n = len(pred)
    for i in range(n):
        for j in range(i + 1, n):
            if pred[i] in pos and pred[j] in pos and pos[pred[i]] > pos[pred[j]]:
                inv += 1
    pairs = len(gold) * (len(gold) - 1) / 2
    return inv / max(pairs, 1.0)


def combined_reward(
    schema_ok: bool,
    runnable_ok: bool,
    runnable_score: float,
    order_distance: float,
    w_schema: float = 0.35,
    w_run: float = 0.35,
    w_order: float = 0.3,
) -> float:
    """
    奖励越大越好：Schema 通过、可运行、与金标准序接近。
    order_distance 为 kendall_tau_like，需转为「相似度」。
    """
    r_schema = 1.0 if schema_ok else 0.0
    r_run = (1.0 if runnable_ok else 0.5) * runnable_score
    r_order = 1.0 - min(order_distance, 1.0)
    return w_schema * r_schema + w_run * r_run + w_order * r_order
