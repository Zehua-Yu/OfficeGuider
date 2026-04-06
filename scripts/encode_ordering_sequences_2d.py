#!/usr/bin/env python3
"""
读取 ordering 的 train/test jsonl：文本编码为向量，再 PCA 降至 2 维。

训练集上 fit，测试集仅 transform。

用法（仓库根目录）:
  PYTHONPATH=backend python scripts/encode_ordering_sequences_2d.py
  PYTHONPATH=backend python scripts/encode_ordering_sequences_2d.py --encoder tfidf_svd   # 无需下载模型

安装（句向量）:
  pip install -r backend/requirements-encode.txt
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

_ORDERING_DIR = ROOT / "data" / "ordering"


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=Path, default=_ORDERING_DIR / "train.jsonl")
    ap.add_argument("--test-jsonl", type=Path, default=_ORDERING_DIR / "test.jsonl")
    ap.add_argument("--out-dir", type=Path, default=_ORDERING_DIR / "encoded_2d")
    ap.add_argument(
        "--encoder",
        choices=("sentence_transformer", "tfidf_svd"),
        default="tfidf_svd",
        help="默认 tfidf_svd 离线可跑；sentence_transformer 需能访问 Hugging Face 或本地模型路径",
    )
    ap.add_argument(
        "--field",
        choices=("ground_truth", "shuffled_input", "both"),
        default="ground_truth",
        help="编码哪条序列；both 时每样本输出两行（后缀 _gt / _shuf）",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get(
            "SENTENCE_TRANSFORMER_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ),
        help="sentence-transformers 模型名或本地路径",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-npy", action="store_true", help="另存中间向量 .npy")
    ap.add_argument("--pca-seed", type=int, default=42)
    ap.add_argument(
        "--text-format",
        choices=("arrow", "linear"),
        default="arrow",
        help="arrow: ID（中文名）用箭头连接；linear: ID 与中文名空格分隔",
    )
    ap.add_argument("--svd-dim", type=int, default=48, help="tfidf_svd 中间维度（再 PCA→2）")
    args = ap.parse_args()

    try:
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        print("缺少 scikit-learn，请执行: pip install scikit-learn\n", file=sys.stderr)
        raise SystemExit(1) from e

    from smartflow.sequence_text_encode import atoms_sequence_to_linear_text, atoms_sequence_to_text

    fmt = (
        atoms_sequence_to_text
        if args.text_format == "arrow"
        else atoms_sequence_to_linear_text
    )

    train_rows = load_jsonl(args.train_jsonl)
    test_rows = load_jsonl(args.test_jsonl)

    def texts_from_rows(rows: list[dict], key: str) -> list[str]:
        return [fmt(r[key]) for r in rows]

    if args.field == "both":
        texts_train = texts_from_rows(train_rows, "ground_truth") + texts_from_rows(
            train_rows, "shuffled_input"
        )
        texts_test = texts_from_rows(test_rows, "ground_truth") + texts_from_rows(
            test_rows, "shuffled_input"
        )
        train_meta = (
            [(r, "gt") for r in train_rows] + [(r, "shuf") for r in train_rows]
        )
        test_meta = [(r, "gt") for r in test_rows] + [(r, "shuf") for r in test_rows]
    else:
        key = args.field
        texts_train = texts_from_rows(train_rows, key)
        texts_test = texts_from_rows(test_rows, key)
        train_meta = [(r, key) for r in train_rows]
        test_meta = [(r, key) for r in test_rows]

    manifest_extra: dict[str, Any] = {}

    if args.encoder == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            print(
                "缺少 sentence-transformers，请: pip install sentence-transformers\n"
                "或改用: --encoder tfidf_svd\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from e

        print(f"加载模型 {args.model} …")
        model = SentenceTransformer(args.model)
        print(f"编码 train={len(texts_train)} test={len(texts_test)} …")
        emb_train = model.encode(
            texts_train,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        emb_test = model.encode(
            texts_test,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        manifest_extra["sentence_transformer_model"] = args.model
    else:
        print("使用 TF-IDF + TruncatedSVD（无外部模型下载）…")
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        x_tr = vec.fit_transform(texts_train)
        x_te = vec.transform(texts_test)
        n_svd = min(args.svd_dim, x_tr.shape[1] - 1) if x_tr.shape[1] > 1 else 1
        n_svd = max(2, n_svd)
        svd = TruncatedSVD(n_components=n_svd, random_state=args.pca_seed)
        emb_train = svd.fit_transform(x_tr)
        emb_test = svd.transform(x_te)
        manifest_extra["tfidf"] = {"analyzer": "char_wb", "ngram_range": [2, 5]}
        manifest_extra["truncated_svd_dim"] = n_svd
        manifest_extra["tfidf_feature_count"] = x_tr.shape[1]

    pca = PCA(n_components=2, random_state=args.pca_seed)
    xy_train = pca.fit_transform(emb_train)
    xy_test = pca.transform(emb_test)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, meta: list, xy: np.ndarray) -> None:
        with path.open("w", encoding="utf-8") as f:
            for (r, tag), (x, y) in zip(meta, xy, strict=True):
                if args.field == "both":
                    suffix = "_gt" if tag == "gt" else "_shuf"
                    sid_out = f"{r['sample_id']}{suffix}"
                    field_name = "ground_truth" if tag == "gt" else "shuffled_input"
                else:
                    sid_out = r["sample_id"]
                    field_name = args.field
                rec = {
                    "sample_id": sid_out,
                    "original_sample_id": r["sample_id"],
                    "split": r["split"],
                    "encode_field": field_name,
                    "x2d": float(x),
                    "y2d": float(y),
                    "chain_len": r["chain_len"],
                    "full_embedding_dim": int(emb_train.shape[1]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train_2d.jsonl", train_meta, xy_train)
    write_jsonl(out_dir / "test_2d.jsonl", test_meta, xy_test)

    manifest = {
        "encoder": args.encoder,
        "text_format": args.text_format,
        "encode_field": args.field,
        "pca_seed": args.pca_seed,
        "full_dim": int(emb_train.shape[1]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "train_points": len(train_meta),
        "test_points": len(test_meta),
        "files": {
            "train_2d": str((out_dir / "train_2d.jsonl").relative_to(ROOT)),
            "test_2d": str((out_dir / "test_2d.jsonl").relative_to(ROOT)),
        },
        **manifest_extra,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_npy:
        np.save(out_dir / "embeddings_train.npy", emb_train)
        np.save(out_dir / "embeddings_test.npy", emb_test)
        np.save(out_dir / "pca_components.npy", pca.components_)

    print(
        f"完成。PCA 解释方差比: {pca.explained_variance_ratio_} "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()
