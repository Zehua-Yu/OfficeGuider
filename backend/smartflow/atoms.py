"""模块 A：原子任务定义与元数据。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IOKind(str, Enum):
    NONE = "none"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    JSON = "json"
    IMAGE = "image"
    TEXT = "text"


@dataclass
class AtomTask:
    """办公原子节点：名称、复杂度、耗时、输入输出、依赖提示。"""

    id: str
    display_name: str
    complexity: float  # 0–1
    est_duration_sec: float
    input_kind: IOKind
    output_kind: IOKind
    tags: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)  # 前置能力标签，非严格 DAG

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "complexity": self.complexity,
            "est_duration_sec": self.est_duration_sec,
            "input_kind": self.input_kind.value,
            "output_kind": self.output_kind.value,
            "tags": list(self.tags),
            "requires": list(self.requires),
        }


# 常见办公原子（可扩展）
ATOM_REGISTRY: Dict[str, AtomTask] = {
    "PDF_Import": AtomTask(
        "PDF_Import", "导入 PDF", 0.2, 5, IOKind.NONE, IOKind.PDF, ["ingest"],
    ),
    "PDF_to_Word": AtomTask(
        "PDF_to_Word",
        "PDF 转 Word",
        0.6,
        120,
        IOKind.PDF,
        IOKind.DOCX,
        ["convert"],
        requires=["ingest"],
    ),
    "Summary_Generation": AtomTask(
        "Summary_Generation",
        "摘要生成",
        0.5,
        60,
        IOKind.TEXT,
        IOKind.TEXT,
        ["nlp"],
    ),
    "Excel_Import": AtomTask(
        "Excel_Import", "导入表格", 0.25, 8, IOKind.NONE, IOKind.XLSX, ["ingest"],
    ),
    "Table_Normalize": AtomTask(
        "Table_Normalize",
        "表格规范化",
        0.45,
        30,
        IOKind.XLSX,
        IOKind.JSON,
        ["table"],
    ),
    "Data_Validate": AtomTask(
        "Data_Validate",
        "数据校验",
        0.35,
        20,
        IOKind.JSON,
        IOKind.JSON,
        ["validate"],
    ),
    "Chart_Generate": AtomTask(
        "Chart_Generate",
        "图表生成",
        0.55,
        45,
        IOKind.JSON,
        IOKind.IMAGE,
        ["viz"],
    ),
    "Export_DOCX": AtomTask(
        "Export_DOCX",
        "导出 DOCX",
        0.3,
        15,
        IOKind.TEXT,
        IOKind.DOCX,
        ["export"],
    ),
    "Code_Generate": AtomTask(
        "Code_Generate",
        "代码生成",
        0.7,
        90,
        IOKind.TEXT,
        IOKind.TEXT,
        ["code"],
    ),
    "Schema_Check": AtomTask(
        "Schema_Check",
        "Schema 校验",
        0.2,
        5,
        IOKind.JSON,
        IOKind.JSON,
        ["validate"],
    ),
}


def get_atom(atom_id: str) -> Optional[AtomTask]:
    return ATOM_REGISTRY.get(atom_id)


def all_atom_ids() -> List[str]:
    return list(ATOM_REGISTRY.keys())


def user_preference_vector(atom_ids: List[str], history_counts: Dict[str, float]) -> Dict[str, float]:
    """从历史频次构造偏好权重（模块 A 特征的一部分）。"""
    out: Dict[str, float] = {}
    for aid in atom_ids:
        out[aid] = float(history_counts.get(aid, 0.0))
    m = max(out.values()) if out else 1.0
    if m <= 0:
        return {a: 0.0 for a in atom_ids}
    return {a: out[a] / m for a in atom_ids}
