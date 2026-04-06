#!/usr/bin/env python3
"""
用 atom 的 2D 坐标按顺序拼成序列张量，形状 [N, L, 2]：
  N = 序列条数（如训练集样本数）
  L = 每条序列统一长度（action 槽位数，短序列右侧 padding）
  2 = (x, y) 坐标

默认 L = 当前数据集中该 split 的最大链长；可用 --fixed-len 指定。
短序列用零坐标填充，并生成 mask [N, L]（1=真实 action，0=padding）。

用法:
  PYTHONPATH=backend python scripts/build_sequence_xy_tensors.py --split train
  PYTHONPATH=backend python scripts/build_sequence_xy_tensors.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

_ORDERING = ROOT / "data" / "ordering"
_ATOMS_JSON = _ORDERING / "atom_embeddings_2d" / "atoms_2d.json"


def _rel_root(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_atom_xy(path: Path) -> dict[str, tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for aid, xy in data["atoms"].items():
        out[aid] = (float(xy["x"]), float(xy["y"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test", "both"), default="train")
    ap.add_argument(
        "--ordering-dir",
        type=Path,
        default=_ORDERING,
        help="含 train.jsonl / test.jsonl 的目录，如 data/ordering/long_10k",
    )
    ap.add_argument("--atoms-2d", type=Path, default=_ATOMS_JSON)
    ap.add_argument(
        "--field",
        choices=("shuffled_input", "ground_truth"),
        default="shuffled_input",
        help="用哪条序列拼坐标（模型输入一般为乱序）",
    )
    ap.add_argument(
        "--fixed-len",
        type=int,
        default=None,
        help="统一长度 L；默认取该 split 中最大链长",
    )
    ap.add_argument(
        "--pad-value",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("PX", "PY"),
        help="padding 槽位坐标",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="默认 data/ordering/xy_tensors/",
    )
    ap.add_argument(
        "--combine-train-test",
        action="store_true",
        help="在 --split both 时另存 train+test 沿 N 拼接为 all_*（形状 [N_train+N_test, L, 2]）",
    )
    args = ap.parse_args()

    atom_xy = load_atom_xy(args.atoms_2d)
    out_dir = args.out_dir or (args.ordering_dir / "xy_tensors")
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = np.array(args.pad_value, dtype=np.float32)

    def process_split(name: str) -> tuple[np.ndarray, np.ndarray, int] | None:
        path = args.ordering_dir / f"{name}.jsonl"
        rows = load_jsonl(path)
        if not rows:
            print(f"跳过空文件: {path}", file=sys.stderr)
            return None

        lens = [len(r[args.field]) for r in rows]
        L = args.fixed_len if args.fixed_len is not None else max(lens)
        if args.fixed_len is not None and max(lens) > L:
            raise SystemExit(
                f"{name}: 存在链长 {max(lens)} > --fixed-len {L}，请增大 L 或截断策略"
            )

        N = len(rows)
        arr = np.zeros((N, L, 2), dtype=np.float32)
        mask = np.zeros((N, L), dtype=np.float32)

        for i, r in enumerate(rows):
            seq = r[args.field]
            ln = len(seq)
            if ln > L:
                raise SystemExit(f"{name} sample {r.get('sample_id')}: len {ln} > L {L}")
            for j, aid in enumerate(seq):
                if aid not in atom_xy:
                    raise KeyError(f"未知 atom_id: {aid}，请检查 atoms_2d.json")
                arr[i, j, 0], arr[i, j, 1] = atom_xy[aid]
                mask[i, j] = 1.0
            if ln < L:
                arr[i, ln:, :] = pad

        stem = f"{name}_xy_L{L}_{args.field}"
        np.save(out_dir / f"{stem}.npy", arr)
        np.save(out_dir / f"{stem}_mask.npy", mask)
        meta = {
            "shape": [N, L, 2],
            "dtype": "float32",
            "split": name,
            "num_sequences": N,
            "seq_len": L,
            "field": args.field,
            "atoms_2d_json": _rel_root(args.atoms_2d),
            "source_jsonl": _rel_root(path),
            "pad_value": [float(pad[0]), float(pad[1])],
            "chain_len_min": int(min(lens)),
            "chain_len_max": int(max(lens)),
            "files": {
                "xy": _rel_root(out_dir / f"{stem}.npy"),
                "mask": _rel_root(out_dir / f"{stem}_mask.npy"),
            },
        }
        (out_dir / f"{stem}_manifest.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{name}: saved [N,L,2]={arr.shape} -> {out_dir / (stem + '.npy')}")
        return arr, mask, L

    if args.split == "both":
        r_tr = process_split("train")
        r_te = process_split("test")
        if (
            args.combine_train_test
            and r_tr is not None
            and r_te is not None
            and r_tr[2] == r_te[2]
        ):
            L = r_tr[2]
            a_all = np.concatenate([r_tr[0], r_te[0]], axis=0)
            m_all = np.concatenate([r_tr[1], r_te[1]], axis=0)
            stem = f"all_xy_L{L}_{args.field}"
            np.save(out_dir / f"{stem}.npy", a_all)
            np.save(out_dir / f"{stem}_mask.npy", m_all)
            (out_dir / f"{stem}_manifest.json").write_text(
                json.dumps(
                    {
                        "shape": list(a_all.shape),
                        "dtype": "float32",
                        "split": "train+test_concat",
                        "seq_len": L,
                        "field": args.field,
                        "files": {
                            "xy": _rel_root(out_dir / f"{stem}.npy"),
                            "mask": _rel_root(out_dir / f"{stem}_mask.npy"),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"combined: [N,L,2]={a_all.shape} -> {out_dir / (stem + '.npy')}")
    else:
        process_split(args.split)


if __name__ == "__main__":
    main()
