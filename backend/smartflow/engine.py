"""编排：特征 → 指针网络 → 重定位 → Top-K 推荐与验证。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from smartflow import atoms
from smartflow.features import build_context_tensor
from smartflow.pointer_orderer import PointerOrderer, greedy_order, load_orderer_checkpoint
from smartflow.reposition import apply_feedback_boost
from smartflow.validator import (
    combined_reward,
    kendall_tau_like,
    load_schema,
    sequence_runnable,
    validate_ai_output,
)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMA = _ROOT / "data" / "wps_schema_stub.json"
_DEFAULT_OFFICE_CATALOG = _ROOT / "data" / "ordering" / "atom_embeddings_2d" / "office_actions_200.json"


class SmartFlowEngine:
    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        node_dim: int = 32,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = Path(checkpoint) if checkpoint else None
        if ck and ck.is_file():
            self.model = load_orderer_checkpoint(ck, self.device)
        else:
            self.model = PointerOrderer(node_dim=node_dim).to(self.device)
        self.model.eval()
        self.node_dim = node_dim
        self._schema: Optional[dict] = None

    def schema(self) -> dict:
        if self._schema is None:
            p = _DEFAULT_SCHEMA
            if not p.is_file():
                p = Path(__file__).resolve().parents[1] / "data" / "wps_schema_stub.json"
            self._schema = load_schema(p) if p.is_file() else {}
        return self._schema

    def recommend(
        self,
        history: List[str],
        *,
        candidate_pool: Optional[List[str]] = None,
        feedback: Optional[Dict[str, float]] = None,
        topk: int = 5,
        preview_depth: int = 0,
        io_bonus_scale: float = 0.5,
    ) -> Dict[str, Any]:
        pool = list(candidate_pool) if candidate_pool else atoms.all_atom_ids()
        hist_set = set(history)
        remaining = [a for a in pool if a not in hist_set]
        if not remaining:
            return {"topk": [], "preview_chain": [], "candidates": []}

        nodes = build_context_tensor(remaining, history, dim=self.node_dim).to(self.device)
        feedback = feedback or {}

        from smartflow.atoms import ATOM_REGISTRY, IOKind
        from smartflow.features import transition_feasibility

        last_out = IOKind.NONE
        if history:
            la = ATOM_REGISTRY.get(history[-1])
            if la:
                last_out = la.output_kind

        with torch.no_grad():
            logits = self.model.step_logits(nodes, [], None)
            logits = apply_feedback_boost(logits, remaining, feedback)
            feas = torch.zeros_like(logits)
            for i, aid in enumerate(remaining):
                at = ATOM_REGISTRY.get(aid)
                if at:
                    feas[0, i] = transition_feasibility(last_out, at) * io_bonus_scale
            logits = logits + feas

            k = min(topk, len(remaining))
            scores, idx = torch.topk(logits[0], k=k)
            topk_list = [
                {"atom": remaining[int(i)], "score": float(scores[j].item())}
                for j, i in enumerate(idx.tolist())
            ]

            preview_chain: List[str] = []
            if preview_depth > 0:
                preview_chain, _ = greedy_order(
                    self.model,
                    nodes,
                    remaining,
                    history,
                    feedback,
                    topk=1,
                    max_steps=min(preview_depth, len(remaining)),
                    io_bonus_scale=io_bonus_scale,
                )

        return {
            "topk": topk_list,
            "preview_chain": preview_chain,
            "candidates": remaining,
        }

    def validate_flow_and_output(
        self,
        sequence: List[str],
        ai_output: Optional[Dict[str, Any]] = None,
        gold_sequence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        schema = self.schema()
        schema_ok = True
        schema_msg = None
        if ai_output is not None and schema:
            schema_ok, schema_msg = validate_ai_output(ai_output, schema)

        run_ok, run_score = sequence_runnable(sequence)
        order_dist = 0.0
        if gold_sequence:
            order_dist = kendall_tau_like(sequence, gold_sequence)
        reward = combined_reward(schema_ok, run_ok, run_score, order_dist)

        return {
            "schema_ok": schema_ok,
            "schema_message": schema_msg,
            "runnable_ok": run_ok,
            "runnable_score": run_score,
            "order_distance": order_dist,
            "reward": reward,
        }


def list_atoms() -> List[Dict[str, Any]]:
    from smartflow import custom_actions_store
    from smartflow import office_catalog

    rows: List[Dict[str, Any]] = []
    p = Path(os.environ.get("SMARTFLOW_OFFICE_CATALOG", str(_DEFAULT_OFFICE_CATALOG)))
    if p.is_file():
        rows.extend(office_catalog.list_atoms_for_api(p))
    else:
        for t in atoms.ATOM_REGISTRY.values():
            d = t.to_dict()
            d["source"] = "catalog"
            rows.append(d)

    rows.extend(custom_actions_store.list_atoms_for_api())
    return rows
