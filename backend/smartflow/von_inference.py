"""加载 third_party/VON 预训练权重，对 2D 坐标点集做贪心排序（与 eval.py 一致）。"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# 项目根目录（backend 的父目录）
_SMARTFLOW_ROOT = Path(__file__).resolve().parents[2]
_VON_ROOT = _SMARTFLOW_ROOT / "third_party" / "VON"


def resolve_pretrained_dir(raw: Optional[str]) -> Path:
    if not raw or not raw.strip():
        raise ValueError(
            "未配置 SMARTFLOW_VON_PRETRAINED：请设为 VON 预训练目录（含 args.json 与 epoch-*.pt），"
            "例如 third_party/VON/pretrained/my_run"
        )
    p = Path(raw.strip())
    if not p.is_absolute():
        p = (_SMARTFLOW_ROOT / p).resolve()
    return p


def load_atoms_xy(path: Path) -> Dict[str, Tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Tuple[float, float]] = {}
    for aid, xy in data["atoms"].items():
        out[aid] = (float(xy["x"]), float(xy["y"]))
    return out


def _ensure_von_import_path() -> None:
    von = str(_VON_ROOT.resolve())
    if von not in sys.path:
        sys.path.insert(0, von)


def load_von_model(model_dir: Path):
    """等价于 eval.py 中 load_model：从目录读取 args.json 与最大 epoch-*.pt。"""
    _ensure_von_import_path()
    from utils.functions import load_model  # noqa: WPS433

    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"预训练目录不存在: {model_dir}")
    pts = list(model_dir.glob("epoch-*.pt"))
    if not (model_dir / "args.json").is_file():
        raise FileNotFoundError(f"缺少 args.json: {model_dir}")
    if not pts:
        raise FileNotFoundError(
            f"目录中无 epoch-*.pt 权重文件: {model_dir}，请将训练好的 .pt 放入该目录"
        )

    model, args = load_model(str(model_dir))
    model.eval()
    return model, args


def infer_greedy_order(
    atom_ids: List[str],
    *,
    atoms_xy: Dict[str, Tuple[float, float]],
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
) -> Tuple[List[str], List[int], List[Dict[str, float]], float]:
    """
    将 atom 按 2D 坐标组成 batch，调用 VON AttentionModel.sample_many(greedy)。

    返回：
    - ordered_atom_ids：模型给出的访问顺序对应的原子 ID
    - index_order：在输入 atom_ids 位置上的下标排列
    - coords_ordered：每步的 {x,y}（与 ordered 对齐）
    - cost：TSP 路径长度（与训练 metric 一致时的标量）
    """
    if len(atom_ids) < 2:
        raise ValueError("至少需要 2 个动作")
    for a in atom_ids:
        if a not in atoms_xy:
            raise KeyError(f"未知 atom_id（请检查 atom_embeddings_2d/atoms_2d.json）: {a}")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = [[atoms_xy[a][0], atoms_xy[a][1]] for a in atom_ids]
    loc = torch.tensor([coords], dtype=torch.float32, device=dev)

    model = model.to(dev)
    model.set_decode_type("greedy")

    with torch.no_grad():
        pi, cost = model.sample_many(loc, batch_rep=1, iter_rep=1)

    # pi: [batch, graph_size]
    seq = pi[0].detach().cpu().long().tolist()
    n = len(atom_ids)
    seq = seq[:n]
    index_order = [int(i) for i in seq]
    ordered_atom_ids = [atom_ids[i] for i in index_order]
    coords_ordered = [{"x": coords[i][0], "y": coords[i][1]} for i in index_order]
    c0 = float(cost[0].detach().cpu().item()) if cost.numel() else 0.0
    return ordered_atom_ids, index_order, coords_ordered, c0


class VonModelCache:
    """进程内单例，避免每次请求重复加载权重。"""

    def __init__(self) -> None:
        self._model = None
        self._args: Optional[Dict[str, Any]] = None
        self._dir: Optional[str] = None

    def get(self, model_dir: Path):
        key = str(model_dir.resolve())
        if self._model is None or self._dir != key:
            self._model, self._args = load_von_model(model_dir)
            self._dir = key
        return self._model, self._args

    def clear(self) -> None:
        self._model = None
        self._args = None
        self._dir = None


von_cache = VonModelCache()


def euclidean_tour_length(coords_ordered: List[Dict[str, float]], *, closed: bool = False) -> float:
    """沿当前访问顺序的欧氏边长之和；closed=True 时再加上首尾闭合边。"""
    if len(coords_ordered) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords_ordered) - 1):
        a, b = coords_ordered[i], coords_ordered[i + 1]
        total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    if closed and len(coords_ordered) >= 2:
        a, b = coords_ordered[-1], coords_ordered[0]
        total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    return total


def pairwise_euclidean_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(coords)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            m[i][j] = m[j][i] = round(d, 6)
    return m


def centroid_and_bbox(coords: List[Tuple[float, float]]) -> Dict[str, Any]:
    if not coords:
        return {"centroid": None, "bbox": None}
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    n = len(coords)
    return {
        "centroid": {"x": round(sum(xs) / n, 6), "y": round(sum(ys) / n, 6)},
        "bbox": {
            "x_min": round(min(xs), 6),
            "x_max": round(max(xs), 6),
            "y_min": round(min(ys), 6),
            "y_max": round(max(ys), 6),
        },
    }
