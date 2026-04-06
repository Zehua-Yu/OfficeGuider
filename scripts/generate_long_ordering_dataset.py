#!/usr/bin/env python3
"""
长链路办公流排序数据：固定链长（默认 50）、多样 I/O 随机游走，样本量大（默认 10000）。

用法:
  PYTHONPATH=backend python scripts/generate_long_ordering_dataset.py --total 10000 --chain-len 50

输出目录默认: data/ordering/long_10k/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rel_to_root(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
sys.path.insert(0, str(ROOT / "backend"))

from smartflow.ordering_chain_gen import random_long_feasible_chain, shuffled_permutation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=10000)
    ap.add_argument("--chain-len", type=int, default=50)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20260402)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "ordering" / "long_10k",
    )
    args = ap.parse_args()

    total = args.total
    L = args.chain_len
    seed = args.seed
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(round(total * args.train_ratio))
    n_test = total - n_train

    train_f = (out_dir / "train.jsonl").open("w", encoding="utf-8")
    test_f = (out_dir / "test.jsonl").open("w", encoding="utf-8")
    try:
        for i in range(total):
            rng_chain = random.Random(seed + i * 100003)
            low = 0.32 + (i % 7) * 0.02
            high = min(0.55, low + 0.12)
            gt = random_long_feasible_chain(
                rng_chain,
                L,
                min_feas_low=low,
                min_feas_high=high,
            )
            if len(gt) != L:
                raise RuntimeError(f"internal: len {len(gt)} != {L}")

            rng_shuf = random.Random(seed * 1000003 + i * 7919)
            shuffled = shuffled_permutation(rng_shuf, gt)
            for _ in range(256):
                if shuffled != gt:
                    break
                s = list(gt)
                rng_shuf.shuffle(s)
                shuffled = s
            assert sorted(gt) == sorted(shuffled)
            assert shuffled != gt, "乱序与 GT 完全相同（可能为全同原子），请换 seed"

            split = "train" if i < n_train else "test"
            rec = {
                "sample_id": f"ord_long_{i:06d}",
                "split": split,
                "chain_len": L,
                "ground_truth": gt,
                "shuffled_input": shuffled,
                "shuffle_seed": seed * 1000003 + i * 7919,
                "meta": {
                    "task": "linear_order_recovery",
                    "variant": "long_chain",
                    "feas_band": [round(low, 3), round(high, 3)],
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
        "schema_version": "ordering_long_v1",
        "total": total,
        "chain_len": L,
        "train": n_train,
        "test": n_test,
        "train_ratio": args.train_ratio,
        "seed": seed,
        "files": {
            "train": _rel_to_root(out_dir / "train.jsonl"),
            "test": _rel_to_root(out_dir / "test.jsonl"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {n_train} train + {n_test} test (chain_len={L}) -> {out_dir}")


if __name__ == "__main__":
    main()
