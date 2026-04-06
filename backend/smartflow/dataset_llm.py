"""模块 A：使用 Deepseek（OpenAI 兼容 API）生成模拟办公操作日志 JSON。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import httpx

ATOM_IDS = (
    "PDF_Import, PDF_to_Word, Summary_Generation, Excel_Import, Table_Normalize, "
    "Data_Validate, Chart_Generate, Export_DOCX, Code_Generate, Schema_Check"
)

SYSTEM_PROMPT = f"""你是数据合成助手。只输出合法 JSON（不要 Markdown 围栏）。
必须符合 schema_version \"2.0\" 的办公会话日志结构：

根字段：
- schema_version: \"2.0\"
- generated_at: ISO8601
- catalog: {{ \"atom_ids\": string[] }}  （从下列原子中选）
- entity_registry: 数组，元素含 entity_id, kind(document|spreadsheet|attachment|generated_artifact|email_draft|code_snippet), mime?, size_bytes?, sensitivity(public|internal|confidential|restricted)?, path_stub?, metadata?
- sessions: 每个 session 含 session_id, user_id, workspace(org_unit, project_id, locale, clearance_level, network_zone), client(app, app_version, os, device_class), intent_tags[], session_goal, session_metrics(interrupt_count, total_duration_ms), events[]
- events[] 每项必填: event_id, seq, ts, atom；可选 phase(prepare|execute|review|export|rollback), payload, io_context(inputs/outputs 为 {{entity_id, role}}), depends_on[event_id], metrics(duration_ms,...), outcome, error?, risk_flags[], user_feedback?
- gold_flows: 每项含 flow_id, name, description?, atoms[], preconditions?, postconditions?, alternative_branches?（from_atom, branch_atoms, when）
- policies: forbidden_atom_pairs [[a,b]], required_schema_by_atom {{}}

可用原子 ID 仅从下列选择：
{ATOM_IDS}

要求：每个 session 4–10 个事件，时间递增，depends_on 与 seq 逻辑一致，至少 2 个 session 场景差异明显（如涉密 offline vs 内网 WPS）。"""


async def generate_logs_deepseek(
    n_sessions: int = 3,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")

    url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    url = f"{url}/v1/chat/completions"
    m = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    user_msg = (
        f"请生成 {n_sessions} 个会话，严格遵循系统说明的 JSON 结构；"
        "entity_registry 至少 4 条，gold_flows 至少 2 条，policies 需包含 forbidden_atom_pairs 与 required_schema_by_atom。"
    )

    payload = {
        "model": m,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.65,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(content)


def generate_logs_deepseek_sync(**kwargs: Any) -> Dict[str, Any]:
    import asyncio

    return asyncio.run(generate_logs_deepseek(**kwargs))
