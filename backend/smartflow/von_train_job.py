"""在 third_party/VON 中启动用户数据集训练任务，并收集日志。"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_SMARTFLOW_ROOT = Path(__file__).resolve().parents[2]
_VON_ROOT = _SMARTFLOW_ROOT / "third_party" / "VON"
_JOBS_ROOT = _SMARTFLOW_ROOT / "data" / "user_training" / "jobs"
_ACTIVE_MODEL = _SMARTFLOW_ROOT / "data" / "user_training" / "active_model.json"

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _find_pretrained_dir(job_dir: Path, sample_size: int) -> Optional[Path]:
    """训练完成后在输出目录下查找最近一次保存的 run 目录（含 args.json）。"""
    base = job_dir / "von_outputs" / f"order_{sample_size}"
    if not base.is_dir():
        return None
    best: Optional[Path] = None
    best_t = 0.0
    for args_f in base.rglob("args.json"):
        d = args_f.parent
        t = args_f.stat().st_mtime
        if t >= best_t and list(d.glob("epoch-*.pt")):
            best_t = t
            best = d
    return best


def start_training_job(
    *,
    instances: List[Any],
    metric: str,
    sample_size: int,
    n_epochs: int,
    epoch_size: int,
    batch_size: int,
    val_size: int,
    run_name: str,
    set_active_on_finish: bool,
) -> str:
    if metric not in ("tsp", "moransI"):
        raise ValueError("metric 仅支持 tsp 或 moransI（stress 需 GPU 且未在界面开放）")
    if epoch_size % batch_size != 0:
        raise ValueError("epoch_size 必须能被 batch_size 整除")
    if not instances:
        raise ValueError("instances 不能为空")

    rows: List[np.ndarray] = []
    for inst in instances:
        arr = np.asarray(inst, dtype=np.float32)
        if arr.shape != (sample_size, 2):
            raise ValueError(
                f"每个实例须为 shape ({sample_size}, 2)，当前为 {arr.shape}"
            )
        rows.append(arr)

    job_id = uuid.uuid4().hex[:12]
    job_dir = _JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = job_dir / "dataset.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(rows, f)

    output_dir = (job_dir / "von_outputs").resolve()
    run_name_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)[:48] or "sf_run"

    cmd: List[str] = [
        sys.executable,
        "-u",
        "run.py",
        "--mission",
        "user_pkl",
        "--metric",
        metric,
        "--train_dataset",
        str(pkl_path.resolve()),
        "--sample_size",
        str(sample_size),
        "--n_epochs",
        str(n_epochs),
        "--epoch_size",
        str(epoch_size),
        "--batch_size",
        str(batch_size),
        "--val_size",
        str(min(val_size, len(rows))),
        "--output_dir",
        str(output_dir),
        "--run_name",
        run_name_safe,
        "--run_mode",
        "train",
        "--baseline",
        "rollout",
        "--no_tensorboard",
        "--no_progress_bar",
        "--checkpoint_epochs",
        "1",
        "--log_step",
        "999",
    ]

    if not os.environ.get("SMARTFLOW_FORCE_CUDA"):
        import torch

        if not torch.cuda.is_available():
            cmd.append("--no_cuda")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_VON_ROOT.resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "log": [],
            "created": time.time(),
            "metric": metric,
            "sample_size": sample_size,
            "job_dir": str(job_dir),
            "pretrained_dir": None,
            "error": None,
            "exit_code": None,
            "set_active_on_finish": set_active_on_finish,
        }

    def worker() -> None:
        log_lines: List[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(_VON_ROOT.resolve()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_lines.append(line)
                with _jobs_lock:
                    if job_id in _jobs:
                        _jobs[job_id]["log"] = log_lines
                        _jobs[job_id]["last_line"] = line.rstrip()
            proc.wait()
            exit_code = proc.returncode or 0
            with _jobs_lock:
                if job_id not in _jobs:
                    return
                _jobs[job_id]["exit_code"] = exit_code
                _jobs[job_id]["log"] = log_lines[-2000:]
                if exit_code != 0:
                    _jobs[job_id]["status"] = "failed"
                    _jobs[job_id]["error"] = f"进程退出码 {exit_code}"
                    return

            pd = _find_pretrained_dir(job_dir, sample_size)
            with _jobs_lock:
                if job_id not in _jobs:
                    return
                _jobs[job_id]["status"] = "done"
                if pd is not None:
                    _jobs[job_id]["pretrained_dir"] = str(pd.resolve())
                    if set_active_on_finish:
                        _write_active_model(pd.resolve())
                else:
                    _jobs[job_id]["error"] = "未找到输出目录（含 epoch-*.pt）"
        except Exception as e:
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["status"] = "failed"
                    _jobs[job_id]["error"] = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _write_active_model(pretrained_dir: Path) -> None:
    from smartflow import von_inference as von_mod

    _ACTIVE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pretrained_dir": str(pretrained_dir.resolve()),
        "updated_at": time.time(),
    }
    _ACTIVE_MODEL.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    von_mod.von_cache.clear()


def set_active_pretrained_dir(path: str) -> None:
    p = Path(path).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"不是有效目录: {p}")
    if not (p / "args.json").is_file():
        raise FileNotFoundError(f"缺少 args.json: {p}")
    if not list(p.glob("epoch-*.pt")):
        raise FileNotFoundError(f"缺少 epoch-*.pt: {p}")
    _write_active_model(p)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            return None
        return {
            "id": job_id,
            "status": j.get("status"),
            "log": "\n".join(j.get("log", [])),
            "last_line": j.get("last_line"),
            "metric": j.get("metric"),
            "sample_size": j.get("sample_size"),
            "pretrained_dir": j.get("pretrained_dir"),
            "error": j.get("error"),
            "exit_code": j.get("exit_code"),
            "job_dir": j.get("job_dir"),
        }
