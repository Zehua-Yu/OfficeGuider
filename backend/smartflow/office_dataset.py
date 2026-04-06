"""办公会话日志（v2）加载与 JSON Schema 校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonschema import Draft202012Validator

_ROOT = Path(__file__).resolve().parents[2]


def office_log_schema_path() -> Path:
    return _ROOT / "data" / "schemas" / "office_session_log.schema.json"


def load_office_log_schema() -> Dict[str, Any]:
    p = office_log_schema_path()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def validate_office_log_bundle(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """校验整包 mock_logs / LLM 输出是否符合 v2 schema。"""
    schema = load_office_log_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.json_path)
    if not errors:
        return True, []
    msgs = [f"{e.json_path or '$'}: {e.message}" for e in errors[:50]]
    if len(errors) > 50:
        msgs.append(f"... 另有 {len(errors) - 50} 条错误")
    return False, msgs


def extract_event_atoms(session: Dict[str, Any]) -> List[str]:
    return [e["atom"] for e in session.get("events", []) if isinstance(e, dict) and "atom" in e]
