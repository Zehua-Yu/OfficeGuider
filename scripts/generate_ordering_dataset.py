#!/usr/bin/env python3
"""
生成办公流程排序数据集：ground truth 链 + 打乱后的输入序列，划分 train/test。

用法（在仓库根目录）:
  PYTHONPATH=backend python scripts/generate_ordering_dataset.py

输出:
  data/ordering/train.jsonl
  data/ordering/test.jsonl
  data/ordering/manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# 仓库根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from smartflow.ordering_chain_gen import (  # noqa: E402
    CANONICAL_TEMPLATES,
    random_valid_chain,
    shuffled_permutation,
)


def build_ground_truth_bank(seed: int, min_unique: int = 80) -> list[list[str]]:
    """先尽量收集不重复的可行链，再供 500 条样本循环抽样。"""
    rng = random.Random(seed)
    seen: set[tuple[str, ...]] = set()
    bank: list[list[str]] = []

    for t in CANONICAL_TEMPLATES:
        key = tuple(t)
        if key not in seen:
            seen.add(key)
            bank.append(list(t))

    attempts = 0
    max_attempts = 200000
    while len(bank) < min_unique and attempts < max_attempts:
        attempts += 1
        c = random_valid_chain(rng, min_len=3, max_len=8)
        if not c:
            continue
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        bank.append(c)

    if len(bank) < 3:
        raise RuntimeError("未能生成足够 ground truth 链，请检查 ATOM_REGISTRY")

    return bank


def assign_ground_truths(n: int, bank: list[list[str]]) -> list[list[str]]:
    """每条样本一条 GT；bank 不足时循环使用（打乱种子仍使 shuffled_input 不同）。"""
    return [bank[i % len(bank)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=500, help="总样本数")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    ap.add_argument("--seed", type=int, default=42, help="主随机种子")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "ordering")
    args = ap.parse_args()

    total: int = args.total
    train_ratio: float = args.train_ratio
    seed: int = args.seed
    out_dir: Path = args.out_dir

    n_train = int(round(total * train_ratio))
    n_test = total - n_train

    bank = build_ground_truth_bank(seed + 11, min_unique=120)
    ground_truths = assign_ground_truths(total, bank)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    rng_shuffle = random.Random(seed)

    train_f = train_path.open("w", encoding="utf-8")
    test_f = test_path.open("w", encoding="utf-8")
    try:
        for i in range(total):
            gt = ground_truths[i]
            rng_i = random.Random(seed * 1000003 + i * 7919)
            shuffled = shuffled_permutation(rng_i, gt)
            assert sorted(gt) == sorted(shuffled), (i, gt, shuffled)
            if len(gt) > 1:
                assert shuffled != gt, (i, gt)
            split = "train" if i < n_train else "test"
            rec = {
                "sample_id": f"ord_{i:05d}",
                "split": split,
                "chain_len": len(gt),
                "ground_truth": gt,
                "shuffled_input": shuffled,
                "shuffle_seed": seed * 1000003 + i * 7919,
                "meta": {
                    "task": "linear_order_recovery",
                    "description": "将 shuffled_input 重排为与 ground_truth 一致",
                },
            }
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if split == "train":
                train_f.write(line)
            else:
                test_f.write(line)
    finally:
        train_f.close()
        test_f.close()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "ordering_v1",
        "total": total,
        "train": n_train,
        "test": n_test,
        "train_ratio": train_ratio,
        "seed": seed,
        "unique_ground_truth_chains_in_bank": len(bank),
        "files": {
            "train": str(train_path.relative_to(ROOT)),
            "test": str(test_path.relative_to(ROOT)),
        },
        "fields": {
            "ground_truth": "正确办公原子顺序（标签）",
            "shuffled_input": "同一 multiset 的乱序，作为模型输入",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {n_train} train + {n_test} test -> {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
