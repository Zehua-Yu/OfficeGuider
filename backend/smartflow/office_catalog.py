"""从 office_actions_*.json 加载办公动作列表（与 VON 2D 坐标同源）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from smartflow.atoms import IOKind


def list_atoms_for_api(catalog_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    atoms = data.get("atoms") or {}
    rows: List[Dict[str, Any]] = []
    for aid in sorted(atoms.keys()):
        v = atoms[aid]
        rows.append(
            {
                "id": aid,
                "display_name": v.get("display_name", aid),
                "complexity": float(v.get("complexity", 0.5)),
                "est_duration_sec": float(v.get("est_duration_sec", 60)),
                "input_kind": IOKind.NONE.value,
                "output_kind": IOKind.NONE.value,
                "tags": [],
                "requires": [],
                "source": "catalog",
            }
        )
    return rows
