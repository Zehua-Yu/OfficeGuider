"""OfficeGuider HTTP API（FastAPI）。"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from smartflow import agent_llm
from smartflow import custom_actions_store
from smartflow.dataset_llm import generate_logs_deepseek
from smartflow.engine import SmartFlowEngine, list_atoms
from smartflow.office_dataset import load_office_log_schema, validate_office_log_bundle
from smartflow import von_inference as von_mod
from smartflow import von_train_job

_ROOT = Path(__file__).resolve().parents[1]
# 未设置 SMARTFLOW_VON_PRETRAINED 时使用官方 vendor 的 MI（Moran's I）checkpoint
_DEFAULT_VON_PRETRAINED = _ROOT / "third_party" / "VON" / "pretrained" / "TSP"
_ACTIVE_MODEL_JSON = _ROOT / "data" / "user_training" / "active_model.json"


def _von_pretrained_raw() -> str:
    env = os.environ.get("SMARTFLOW_VON_PRETRAINED", "").strip()
    if env:
        return env
    if _ACTIVE_MODEL_JSON.is_file():
        try:
            j = json.loads(_ACTIVE_MODEL_JSON.read_text(encoding="utf-8"))
            p = (j.get("pretrained_dir") or "").strip()
            if p and Path(p).is_dir():
                return str(Path(p).resolve())
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return str(_DEFAULT_VON_PRETRAINED)


def _pick_directory_native() -> Optional[str]:
    """在运行后端的机器上弹出系统目录选择（macOS 用 osascript，其它平台尝试 tkinter）。"""
    if platform.system() == "Darwin":
        script = (
            "try\n"
            'set f to choose folder with prompt "选择 VON 模型目录"\n'
            "return POSIX path of f\n"
            "on error\n"
            'return ""\n'
            "end try"
        )
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = (r.stdout or "").strip()
        return out if out else None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            p = filedialog.askdirectory(title="选择 VON 预训练模型目录")
        finally:
            root.destroy()
        return p if p else None
    except Exception:
        return None


_ckpt = os.environ.get("SMARTFLOW_CHECKPOINT")
_engine = SmartFlowEngine(
    checkpoint=Path(_ckpt) if _ckpt else None,
    device=os.environ.get("SMARTFLOW_DEVICE"),
)
_ATOMS_2D_JSON = Path(
    os.environ.get(
        "SMARTFLOW_ATOMS_2D",
        str(_ROOT / "data" / "ordering" / "atom_embeddings_2d" / "office_actions_200.json"),
    )
)
_atoms_xy_cache: dict | None = None
_catalog_atoms_cache: Dict[str, Any] | None = None


def _invalidate_atoms_xy_cache() -> None:
    global _atoms_xy_cache
    _atoms_xy_cache = None


def _get_atoms_xy() -> dict:
    global _atoms_xy_cache
    if _atoms_xy_cache is None:
        if not _ATOMS_2D_JSON.is_file():
            raise FileNotFoundError(f"缺少原子 2D 坐标文件: {_ATOMS_2D_JSON}")
        base = von_mod.load_atoms_xy(_ATOMS_2D_JSON)
        custom = custom_actions_store.load_atoms_xy()
        _atoms_xy_cache = {**base, **custom}
    return _atoms_xy_cache


def _get_catalog_atom_entries() -> Dict[str, Any]:
    global _catalog_atoms_cache
    if _catalog_atoms_cache is None:
        if not _ATOMS_2D_JSON.is_file():
            _catalog_atoms_cache = {}
        else:
            data = json.loads(_ATOMS_2D_JSON.read_text(encoding="utf-8"))
            _catalog_atoms_cache = data.get("atoms") or {}
    return _catalog_atoms_cache


def _get_merged_atom_entries() -> Dict[str, Any]:
    m = dict(_get_catalog_atom_entries())
    m.update(custom_actions_store.load_atoms_dict())
    return m


def _von_process_payload(
    *,
    atom_ids: List[str],
    atoms_xy: dict,
    index_order: List[int],
    coords_ordered: List[Dict[str, float]],
    cost: float,
    model_dir: Path,
    model_args: Dict[str, Any],
) -> Dict[str, Any]:
    cat = _get_merged_atom_entries()
    n = len(atom_ids)
    loc_rows = [[round(atoms_xy[a][0], 6), round(atoms_xy[a][1], 6)] for a in atom_ids]
    coords_plain = [(atoms_xy[a][0], atoms_xy[a][1]) for a in atom_ids]

    edge_lengths: List[float] = []
    for i in range(len(coords_ordered) - 1):
        a, b = coords_ordered[i], coords_ordered[i + 1]
        edge_lengths.append(
            round(
                (
                    (b["x"] - a["x"]) ** 2
                    + (b["y"] - a["y"]) ** 2
                )
                ** 0.5,
                6,
            )
        )

    pairwise = None
    if n <= 28:
        pairwise = von_mod.pairwise_euclidean_matrix(coords_plain)

    keys = (
        "cost_choose",
        "graph_size",
        "embedding_dim",
        "hidden_dim",
        "mission",
        "problem",
        "model",
        "n_encode_layers",
        "normalization",
    )
    args_subset = {k: model_args.get(k) for k in keys if k in model_args}

    return {
        "catalog_path": str(_ATOMS_2D_JSON),
        "custom_actions_path": str(custom_actions_store.DEFAULT_CUSTOM_PATH),
        "pipeline": [
            {
                "step": 1,
                "name": "动作 → 2D 编码",
                "detail": "每个 atom_id 在 catalog（SMARTFLOW_ATOMS_2D）中有唯一 (x,y)，作为该动作在嵌入平面上的坐标；本阶段无在线神经网络编码器，坐标为数据准备阶段写入。",
            },
            {
                "step": 2,
                "name": "构造 loc 张量",
                "detail": "按用户勾选顺序将坐标堆叠为 float32 张量，形状 [1, N, 2]，与 VON 论文中二维点集排序输入一致。",
            },
            {
                "step": 3,
                "name": "VON greedy 解码",
                "detail": "AttentionModel.set_decode_type('greedy') 后 sample_many，得到访问顺序 π（长度为 N 的下标序列）。",
            },
            {
                "step": 4,
                "name": "模型代价 cost",
                "detail": "标量 cost 由网络内部度量（见 cost_choose，如 tsp / moransI）计算；可与下方欧氏路径长对照理解。",
            },
        ],
        "encoding_input_order": [
            {
                "input_index": i,
                "atom_id": aid,
                "display_name": (cat.get(aid) or {}).get("display_name", aid),
                "x": round(atoms_xy[aid][0], 6),
                "y": round(atoms_xy[aid][1], 6),
            }
            for i, aid in enumerate(atom_ids)
        ],
        "loc_tensor": {
            "shape": [1, n, 2],
            "dtype": "float32",
            "values_row_major": loc_rows,
        },
        "permutation_pi": list(index_order),
        "permutation_note": "π[k] 表示：访问顺序中第 k 步对应「输入序列中第 π[k] 个」动作（0-based 下标）。",
        "euclidean": {
            "edge_lengths_along_output_order": edge_lengths,
            "tour_length_open": round(von_mod.euclidean_tour_length(coords_ordered, closed=False), 6),
            "tour_length_closed": round(von_mod.euclidean_tour_length(coords_ordered, closed=True), 6),
        },
        "point_stats": von_mod.centroid_and_bbox(coords_plain),
        "pairwise_euclidean": pairwise,
        "pairwise_note": "仅当 N≤28 时返回；矩阵行列与 encoding_input_order 下标一致。",
        "model_cost_reported": float(cost),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": {
            "pretrained_dir": str(model_dir.resolve()),
            "args": args_subset,
        },
    }


app = FastAPI(title="SmartFlow API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_origin_regex=os.environ.get(
        "CORS_ORIGIN_REGEX",
        r"http://(localhost|127\.0\.0\.1):\d+",
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    history: List[str] = Field(default_factory=list)
    candidate_pool: Optional[List[str]] = None
    feedback: Optional[Dict[str, float]] = None
    topk: int = 5
    preview_depth: int = 0


class ValidateRequest(BaseModel):
    sequence: List[str]
    ai_output: Optional[Dict[str, Any]] = None
    gold_sequence: Optional[List[str]] = None


class VonOrderRequest(BaseModel):
    """与 eval.py 一致：对给定动作集合的 2D 坐标做贪心排序。"""

    atom_ids: List[str] = Field(..., min_length=2)
    pretrained_dir: Optional[str] = Field(
        None,
        description="可选。非空时使用该目录作为 VON 预训练路径；否则沿用环境变量 / active_model / 默认目录。",
    )


class AgentMessage(BaseModel):
    role: str = "user"
    content: str


class AgentChatRequest(BaseModel):
    messages: List[AgentMessage]
    api_base: str = Field(..., description="OpenAI 兼容 API 根，如 https://api.openai.com/v1 或 https://api.deepseek.com/v1")
    api_key: str = ""
    model: str = "gpt-4o-mini"


class CustomActionItem(BaseModel):
    display_name: str = Field(..., min_length=1)
    id: Optional[str] = None


class CustomActionsAddRequest(BaseModel):
    actions: List[CustomActionItem]


class TrainingJobCreate(BaseModel):
    """用户数据集：每个实例为 [sample_size, 2] 的平面点坐标（与 VON order 任务一致）。"""

    instances: List[List[List[float]]]
    metric: str = Field("tsp", description="tsp 或 moransI")
    sample_size: int = Field(8, ge=2, le=200)
    n_epochs: int = Field(3, ge=1, le=500)
    epoch_size: int = Field(32, ge=4)
    batch_size: int = Field(8, ge=1)
    val_size: int = Field(8, ge=1)
    run_name: str = "officeguider_user"
    set_active_on_finish: bool = True


class SetActiveModelRequest(BaseModel):
    pretrained_dir: str


def _pretrained_source() -> str:
    if os.environ.get("SMARTFLOW_VON_PRETRAINED", "").strip():
        return "env"
    if _ACTIVE_MODEL_JSON.is_file():
        try:
            j = json.loads(_ACTIVE_MODEL_JSON.read_text(encoding="utf-8"))
            if (j.get("pretrained_dir") or "").strip():
                return "active_file"
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return "default"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "officeguider"}


@app.get("/atoms")
def get_atoms() -> dict:
    return {"atoms": list_atoms()}


@app.post("/recommend")
def recommend(body: RecommendRequest) -> dict:
    return _engine.recommend(
        body.history,
        candidate_pool=body.candidate_pool,
        feedback=body.feedback,
        topk=body.topk,
        preview_depth=body.preview_depth,
    )


@app.get("/von/status")
def von_status() -> dict:
    raw = _von_pretrained_raw()
    ready = False
    path_resolved = None
    try:
        p = von_mod.resolve_pretrained_dir(raw)
        path_resolved = str(p)
        ready = (
            p.is_dir()
            and (p / "args.json").is_file()
            and bool(list(p.glob("epoch-*.pt")))
        )
    except ValueError:
        pass
    env_only = os.environ.get("SMARTFLOW_VON_PRETRAINED", "").strip()
    return {
        "pretrained_env": env_only or None,
        "pretrained_dir": path_resolved,
        "using_default_pretrained": not bool(env_only),
        "pretrained_source": _pretrained_source(),
        "active_model_file": str(_ACTIVE_MODEL_JSON),
        "ready": ready,
        "atoms_2d_json": str(_ATOMS_2D_JSON),
    }


@app.post("/von/pick-pretrained-dir")
async def pick_pretrained_dir() -> dict:
    """在服务端本机弹出系统目录选择器，返回所选路径（与浏览器同源开发时常用）。"""
    loop = asyncio.get_running_loop()
    try:
        path = await loop.run_in_executor(None, _pick_directory_native)
    except Exception as e:
        raise HTTPException(503, f"无法打开目录选择器: {e}") from e
    if path is None:
        return {"cancelled": True}
    return {"path": path}


@app.post("/von/order")
def von_order(body: VonOrderRequest) -> dict:
    """使用 VON pretrained（greedy）对动作集合排序。"""
    raw = (body.pretrained_dir or "").strip() or _von_pretrained_raw()
    try:
        model_dir = von_mod.resolve_pretrained_dir(raw)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    try:
        model, model_args = von_mod.von_cache.get(model_dir)
        atoms_xy = _get_atoms_xy()
        ordered, index_order, coords_ordered, cost = von_mod.infer_greedy_order(
            body.atom_ids,
            atoms_xy=atoms_xy,
            model=model,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except KeyError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"VON 推理失败: {e}") from e

    ma = model_args if isinstance(model_args, dict) else {}
    process = _von_process_payload(
        atom_ids=body.atom_ids,
        atoms_xy=atoms_xy,
        index_order=index_order,
        coords_ordered=coords_ordered,
        cost=cost,
        model_dir=model_dir,
        model_args=ma,
    )

    return {
        "ordered_atom_ids": ordered,
        "index_order": index_order,
        "coords_ordered": coords_ordered,
        "cost": cost,
        "input_atom_ids": body.atom_ids,
        "process": process,
    }


@app.post("/agent/chat")
def agent_chat(body: AgentChatRequest) -> dict:
    """多轮对话：由用户配置的 OpenAI 兼容 LLM 生成办公动作描述；解析 JSON actions。"""
    if not body.messages:
        raise HTTPException(400, "messages 不能为空")
    if not body.api_base.strip():
        raise HTTPException(400, "请填写 api_base")
    msgs = agent_llm.build_agent_messages([m.model_dump() for m in body.messages])
    content, err = agent_llm.chat_openai_compatible(
        api_base=body.api_base.strip(),
        api_key=body.api_key,
        model=body.model.strip() or "gpt-4o-mini",
        messages=msgs,
    )
    if err:
        raise HTTPException(502, err)
    suggested = agent_llm.extract_actions_from_assistant_text(content)
    return {"reply": content, "suggested_actions": suggested}


@app.get("/custom/actions")
def custom_actions_list() -> dict:
    return {"atoms": custom_actions_store.list_atoms_for_api()}


@app.post("/custom/actions")
def custom_actions_add(body: CustomActionsAddRequest) -> dict:
    if not body.actions:
        raise HTTPException(400, "actions 为空")
    added = custom_actions_store.add_actions([a.model_dump() for a in body.actions])
    _invalidate_atoms_xy_cache()
    return {"added_ids": added, "count": len(added)}


@app.delete("/custom/actions/{atom_id}")
def custom_actions_delete(atom_id: str) -> dict:
    if not custom_actions_store.delete_action(atom_id):
        raise HTTPException(404, "未找到该自定义动作")
    _invalidate_atoms_xy_cache()
    return {"ok": True}


@app.post("/training/jobs")
def training_create(body: TrainingJobCreate) -> dict:
    try:
        job_id = von_train_job.start_training_job(
            instances=body.instances,
            metric=body.metric,
            sample_size=body.sample_size,
            n_epochs=body.n_epochs,
            epoch_size=body.epoch_size,
            batch_size=body.batch_size,
            val_size=body.val_size,
            run_name=body.run_name,
            set_active_on_finish=body.set_active_on_finish,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"job_id": job_id}


@app.get("/training/jobs/{job_id}")
def training_get(job_id: str) -> dict:
    j = von_train_job.get_job(job_id)
    if j is None:
        raise HTTPException(404, "任务不存在")
    return j


@app.post("/training/set-active")
def training_set_active(body: SetActiveModelRequest) -> dict:
    try:
        von_train_job.set_active_pretrained_dir(body.pretrained_dir)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@app.post("/validate")
def validate(body: ValidateRequest) -> dict:
    return _engine.validate_flow_and_output(
        body.sequence,
        ai_output=body.ai_output,
        gold_sequence=body.gold_sequence,
    )


@app.get("/mock/logs")
def mock_logs() -> dict:
    root = Path(__file__).resolve().parents[1]
    p = root / "data" / "mock_logs.json"
    if not p.is_file():
        raise HTTPException(404, "mock_logs.json not found")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/data/office-log-schema")
def office_log_schema() -> dict:
    return load_office_log_schema()


@app.post("/data/validate-logs")
def validate_logs_bundle(body: dict) -> dict:
    ok, errors = validate_office_log_bundle(body)
    return {"ok": ok, "errors": errors}


class DatasetGenRequest(BaseModel):
    n_sessions: int = 3


@app.post("/dataset/generate")
async def dataset_generate(body: DatasetGenRequest) -> dict:
    try:
        data = await generate_logs_deepseek(body.n_sessions)
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}") from e
    return {"data": data}
