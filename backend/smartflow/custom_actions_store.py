"""用户自定义办公动作：持久化 JSON + 与内置 catalog 合并供 VON 使用。"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smartflow.atoms import IOKind

_SMARTFLOW_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUSTOM_PATH = _SMARTFLOW_ROOT / "data" / "ordering" / "custom_office_actions.json"


def _xy_from_atom_id(atom_id: str) -> Tuple[float, float]:
    """确定性 2D 坐标，使自定义动作可参与 VON 而无需单独训练编码器。"""
    h = hashlib.sha256(atom_id.encode()).digest()
    x = int.from_bytes(h[0:4], "big", signed=True) / 2147483648.0
    y = int.from_bytes(h[4:8], "big", signed=True) / 2147483648.0
    return max(-1.0, min(1.0, float(x))), max(-1.0, min(1.0, float(y)))


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", s.strip())[:32]
    return s.strip("_") or "action"


def _custom_path() -> Path:
    import os

    raw = os.environ.get("SMARTFLOW_CUSTOM_ACTIONS_JSON", "").strip()
    return Path(raw) if raw else DEFAULT_CUSTOM_PATH


def load_atoms_dict() -> Dict[str, Any]:
    p = _custom_path()
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return dict(data.get("atoms") or {})


def _save_atoms_dict(atoms: Dict[str, Any]) -> None:
    p = _custom_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "atoms": atoms}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_atoms_xy() -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for aid, v in load_atoms_dict().items():
        if "x" in v and "y" in v:
            out[aid] = (float(v["x"]), float(v["y"]))
        else:
            xy = _xy_from_atom_id(aid)
            out[aid] = xy
    return out


def list_atoms_for_api() -> List[Dict[str, Any]]:
    raw = load_atoms_dict()
    rows: List[Dict[str, Any]] = []
    for aid in sorted(raw.keys()):
        v = raw[aid]
        rows.append(
            {
                "id": aid,
                "display_name": v.get("display_name", aid),
                "complexity": float(v.get("complexity", 0.5)),
                "est_duration_sec": float(v.get("est_duration_sec", 60)),
                "input_kind": IOKind.NONE.value,
                "output_kind": IOKind.NONE.value,
                "tags": ["custom"],
                "requires": [],
                "source": "custom",
            }
        )
    return rows


def _builtin_catalog_ids() -> set:
    import os

    p = Path(
        os.environ.get(
            "SMARTFLOW_OFFICE_CATALOG",
            str(_SMARTFLOW_ROOT / "data" / "ordering" / "atom_embeddings_2d" / "office_actions_200.json"),
        )
    )
    if not p.is_file():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    return set((data.get("atoms") or {}).keys())


def _ensure_unique_id(
    requested: Optional[str],
    display_name: str,
    *,
    reserved: set,
) -> str:
    if requested and requested.strip():
        aid = requested.strip()
        if aid not in reserved:
            return aid
    base = "cst_" + _slug(display_name) + "_"
    for _ in range(40):
        aid = base + uuid.uuid4().hex[:8]
        if aid not in reserved:
            return aid
    aid = "cst_" + uuid.uuid4().hex
    while aid in reserved:
        aid = "cst_" + uuid.uuid4().hex
    return aid


def add_actions(items: List[Dict[str, Any]]) -> List[str]:
    """每项支持 display_name、可选 id。返回新加入的 id 列表。"""
    atoms = load_atoms_dict()
    added: List[str] = []
    reserved = set(atoms.keys()) | _builtin_catalog_ids()
    for it in items:
        dn = (it.get("display_name") or "").strip()
        if not dn:
            continue
        aid = _ensure_unique_id(it.get("id"), dn, reserved=reserved)
        reserved.add(aid)
        x, y = _xy_from_atom_id(aid)
        atoms[aid] = {
            "display_name": dn,
            "x": round(x, 6),
            "y": round(y, 6),
            "complexity": float(it.get("complexity", 0.5)),
            "est_duration_sec": float(it.get("est_duration_sec", 60)),
        }
        added.append(aid)
    if added:
        _save_atoms_dict(atoms)
    return added


def delete_action(atom_id: str) -> bool:
    atoms = load_atoms_dict()
    if atom_id not in atoms:
        return False
    del atoms[atom_id]
    _save_atoms_dict(atoms)
    return True
