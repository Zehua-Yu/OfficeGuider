"""OpenAI 兼容 Chat Completions + 从回复中解析办公动作 JSON。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import httpx

SYSTEM_PROMPT = """你是办公自动化助手，帮助用户梳理工作流中需要的具体办公动作。
规则：
1. 用中文简洁回复，可追问细节。
2. 每当用户确认或列举出可执行的办公步骤时，在回复最后**必须**单独附上一个 JSON 代码块（markdown fenced），格式严格为：
```json
{"actions": [{"display_name": "动作的自然语言描述"}]}
```
3. 同一类型但目标不同的动作要写成不同 display_name（例如不同文件名、客户、项目名）。
4. 若本轮没有新的动作可登记，使用 {"actions": []}。
5. 除上述 JSON 代码块外，不要用其他格式输出机器解析用的结构化数据。"""


def extract_actions_from_assistant_text(text: str) -> List[Dict[str, Any]]:
    """从助手回复中提取 ```json ...``` 内的 actions 列表。"""
    blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    for block in reversed(blocks):
        block = block.strip()
        if not block:
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "actions" in data:
            raw = data["actions"]
            if isinstance(raw, list):
                out: List[Dict[str, Any]] = []
                for item in raw:
                    if isinstance(item, str):
                        dn = item.strip()
                        if dn:
                            out.append({"display_name": dn})
                    elif isinstance(item, dict) and item.get("display_name"):
                        out.append({"display_name": str(item["display_name"]).strip()})
                return out
        if isinstance(data, list):
            return [{"display_name": str(x).strip()} for x in data if str(x).strip()]
    return []


def chat_openai_compatible(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float = 120.0,
) -> Tuple[str, str]:
    """
    调用 OpenAI 兼容 /v1/chat/completions。
    返回 (assistant_content, error_message)；error 非空表示失败。
    """
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
    }

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            r = client.post(url, headers=headers, json=payload)
    except httpx.RequestError as e:
        return "", f"网络错误: {e}"

    if r.status_code >= 400:
        return "", f"HTTP {r.status_code}: {r.text[:2000]}"

    try:
        data = r.json()
    except json.JSONDecodeError:
        return "", "响应非 JSON"

    try:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        return str(content), ""
    except (KeyError, IndexError, TypeError):
        return "", f"无法解析响应: {str(data)[:500]}"


def build_agent_messages(user_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in user_messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role not in ("user", "assistant"):
            role = "user"
        out.append({"role": role, "content": content})
    return out
