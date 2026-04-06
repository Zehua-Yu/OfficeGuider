#!/usr/bin/env python3
"""
将「每个原子动作」单独编码为向量，再在同一套 PCA 下映射到 2D。
输出：每个 action 一个固定二维坐标（不是整条序列一个点）。

用法（仓库根目录）:
  PYTHONPATH=backend python scripts/encode_atoms_2d.py
  PYTHONPATH=backend python scripts/encode_atoms_2d.py --encoder sentence_transformer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

_DEFAULT_OUT = ROOT / "data" / "ordering" / "atom_embeddings_2d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="输出目录",
    )
    ap.add_argument(
        "--encoder",
        choices=("sentence_transformer", "tfidf_svd"),
        default="tfidf_svd",
        help="tfidf_svd 离线；sentence_transformer 需模型或网络",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get(
            "SENTENCE_TRANSFORMER_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-npy", action="store_true")
    ap.add_argument("--pca-seed", type=int, default=42)
    ap.add_argument(
        "--text-format",
        choices=("arrow", "linear"),
        default="arrow",
    )
    ap.add_argument("--svd-dim", type=int, default=32, help="tfidf_svd 中间维，再 PCA→2")
    args = ap.parse_args()

    try:
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        print("缺少 scikit-learn\n", file=sys.stderr)
        raise SystemExit(1) from e

    from smartflow.atoms import all_atom_ids
    from smartflow.sequence_text_encode import atom_to_text

    atom_ids = sorted(all_atom_ids())
    n = len(atom_ids)
    if n < 2:
        raise SystemExit("原子数量不足，无法做 2D 投影")

    linear = args.text_format == "linear"
    texts = [atom_to_text(aid, linear=linear) for aid in atom_ids]

    manifest_extra: dict[str, Any] = {}

    if args.encoder == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("缺少 sentence-transformers，请 pip install 或改用 --encoder tfidf_svd\n", file=sys.stderr)
            raise SystemExit(1)

        print(f"加载模型 {args.model} …")
        model = SentenceTransformer(args.model)
        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        manifest_extra["sentence_transformer_model"] = args.model
    else:
        print("TF-IDF + TruncatedSVD（按原子语料拟合）…")
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
        x_sparse = vec.fit_transform(texts)
        n_feat = x_sparse.shape[1]
        n_svd = min(args.svd_dim, max(2, n_feat - 1), max(2, n - 1))
        svd = TruncatedSVD(n_components=n_svd, random_state=args.pca_seed)
        emb = svd.fit_transform(x_sparse)
        manifest_extra["tfidf"] = {"analyzer": "char_wb", "ngram_range": [2, 5]}
        manifest_extra["truncated_svd_dim"] = int(emb.shape[1])
        manifest_extra["tfidf_feature_count"] = int(n_feat)

    pca = PCA(n_components=2, random_state=args.pca_seed)
    xy = pca.fit_transform(emb)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    atoms_map: dict[str, dict[str, float]] = {}
    for aid, (tx, (xa, ya)) in zip(atom_ids, zip(texts, xy), strict=True):
        rec = {
            "atom_id": aid,
            "text": tx,
            "x2d": float(xa),
            "y2d": float(ya),
            "embedding_dim_before_pca": int(emb.shape[1]),
        }
        rows.append(rec)
        atoms_map[aid] = {"x": float(xa), "y": float(ya)}

    (out_dir / "atoms_2d.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    (out_dir / "atoms_2d.json").write_text(
        json.dumps({"atoms": atoms_map}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "kind": "per_atom_2d",
        "encoder": args.encoder,
        "text_format": args.text_format,
        "num_atoms": n,
        "pca_seed": args.pca_seed,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "files": {
            "atoms_2d_jsonl": str((out_dir / "atoms_2d.jsonl").relative_to(ROOT)),
            "atoms_2d_json": str((out_dir / "atoms_2d.json").relative_to(ROOT)),
        },
        **manifest_extra,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_npy:
        np.save(out_dir / "atom_embeddings_pre_pca.npy", emb)
        np.save(out_dir / "atom_coords_2d.npy", xy)

    print(f"已写入 {n} 个原子的 2D 坐标 -> {out_dir}")
    print(f"PCA 解释方差比: {pca.explained_variance_ratio_}")


if __name__ == "__main__":
    main()
