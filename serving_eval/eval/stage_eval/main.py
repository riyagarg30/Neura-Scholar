#!/usr/bin/env python3
import os
import json
import time
import math
import ast
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import faiss
import mlflow
import onnxruntime as ort
from transformers import AutoTokenizer
from sqlalchemy import create_engine

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# MLflow settings (via env or defaults)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_STAGE  = os.getenv("MLFLOW_MODEL_STAGE", "Staging")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

DB_URL        = "postgresql+psycopg2://rg5073:rg5073pass@129.114.27.112:5432/cleaned_meta_data_db"
TABLE_NAME    = "arxiv_chunks_eval_5"
EVAL_DIR      = "eval"
INDEX_DIR     = "indexes"
OUTPUT_DIR    = "eval"
TOP_K         = 10
EMBED_TOKENIZER = "distilbert/distilbert-base-uncased"

# Instead of file paths, point to your registered MLflow models:
MODEL_DETAILS = [
    {
        "column":   "chunk_embedding_768_graph",
        "model_uri":"models:/distilbert_chunks_graph/{}".format(MLFLOW_MODEL_STAGE),
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

engine = create_engine(DB_URL, pool_timeout=30, max_overflow=0)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def recall_at_k(gt: list[str], retrieved: list[str], k: int = TOP_K) -> float:
    if not gt: return 0.0
    return sum(1 for p in retrieved[:k] if p in gt) / len(gt)

def adj_recall_at_k(gt: list[str], retrieved: list[str], k: int = TOP_K) -> float:
    if not gt: return 0.0
    rel = max(1, math.ceil(0.4 * len(gt)))
    return sum(1 for p in retrieved[:k] if p in gt) / rel

def mrr_at_k(gt: list[str], retrieved: list[str], k: int = TOP_K) -> float:
    for rank, p in enumerate(retrieved[:k], start=1):
        if p in gt:
            return 1.0 / rank
    return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# RECORD LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_eval_records(eval_dir: str):
    return (
        [json.loads(l) for l in open(f"{eval_dir}/heldout.jsonl")],
        [json.loads(l) for l in open(f"{eval_dir}/slices.jsonl")],
        [json.loads(l) for l in open(f"{eval_dir}/perturbations.jsonl")],
        [json.loads(l) for l in open(f"{eval_dir}/failures.jsonl")],
    )

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    heldout, slices, perturbs, failures = load_eval_records(EVAL_DIR)
    results = []

    for mdl in MODEL_DETAILS:
        col        = mdl["column"]
        model_uri  = mdl["model_uri"]

        logger.info(f"=== Fetching `{col}` model from MLflow URI `{model_uri}` ===")
        local_artifact = mlflow.artifacts.download_artifacts(model_uri)

        # resolve ONNX file path
        if os.path.isdir(local_artifact):
            files = [f for f in os.listdir(local_artifact) if f.endswith(".onnx")]
            if not files:
                raise FileNotFoundError(f"No .onnx file under {local_artifact}")
            onnx_path = os.path.join(local_artifact, files[0])
        else:
            onnx_path = local_artifact

        idx_file  = os.path.join(INDEX_DIR, f"{col}.index")
        meta_file = os.path.join(INDEX_DIR, f"{col}_meta.jsonl")

        # ── 1) Load / build FAISS index ────────────────────────────────────
        if os.path.exists(idx_file):
            t0 = time.perf_counter()
            index = faiss.read_index(idx_file)
            logger.info("Loaded index in %.2fs", time.perf_counter()-t0)

            if os.path.exists(meta_file):
                df_emb = pd.read_json(meta_file, lines=True)
            else:
                df_emb = pd.read_sql(
                    f"SELECT chunk_id, paper_cited FROM {TABLE_NAME}", con=engine
                )
                df_emb["paper_list"] = df_emb["paper_cited"].str.strip("{}").str.split(",")
                logger.info("Rebuilt metadata from Postgres")
        else:
            t0 = time.perf_counter()
            sql = f"SELECT chunk_id, paper_cited, chunk_data, {col} FROM {TABLE_NAME}"
            df_emb = pd.read_sql(sql, con=engine)
            df_emb["paper_list"] = df_emb["paper_cited"].str.strip("{}").str.split(",")
            df_emb["emb_list"] = df_emb[col].apply(
                lambda v: v if isinstance(v, list) else ast.literal_eval(v)
            )
            embs = np.array(df_emb["emb_list"].tolist(), dtype="float32")
            faiss.normalize_L2(embs)
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs)

            faiss.write_index(index, idx_file)
            df_emb[["chunk_id", "paper_list"]].to_json(meta_file, orient="records", lines=True)
            logger.info("Built & saved index in %.2fs", time.perf_counter()-t0)

        paper_lists = df_emb["paper_list"].tolist()

        # ── 2) ONNX session + tokenizer + sanity check ──────────────────────
        t1 = time.perf_counter()
        ort_sess  = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        tokenizer = AutoTokenizer.from_pretrained(EMBED_TOKENIZER)
        logger.info("Loaded ONNX+tokenizer in %.2fs", time.perf_counter()-t1)

        def encode_query(text: str) -> np.ndarray:
            toks = tokenizer(text, return_tensors="np", max_length=300,
                             padding="max_length", truncation=True)
            out = ort_sess.run(
                None,
                {
                    ort_sess.get_inputs()[0].name: toks["input_ids"].astype("int64"),
                    ort_sess.get_inputs()[1].name: toks["attention_mask"].astype("int64"),
                },
            )[0]
            mask = np.expand_dims(toks["attention_mask"], -1).astype("float32")
            emb = (out * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
            faiss.normalize_L2(emb)
            return emb.astype("float32")

        # sanity-check
        sample = pd.read_sql(
            f"SELECT chunk_data, {col} FROM {TABLE_NAME} LIMIT 1", con=engine
        ).iloc[0]
        offline_vec = np.array(
            sample[col] if isinstance(sample[col], list) else ast.literal_eval(sample[col]),
            dtype="float32"
        )
        offline_vec /= np.linalg.norm(offline_vec)
        online_vec  = encode_query(sample["chunk_data"])[0]
        cos_sim     = float(np.dot(offline_vec, online_vec))
        if cos_sim < 0.90:
            logger.warning("Embedding mismatch for `%s` (cos=%.3f)", col, cos_sim)
        else:
            logger.info("Sanity-check passed (cos=%.3f)", cos_sim)

        # ── Retrieval helper
        def retrieve_papers(q_emb: np.ndarray, top_n: int = TOP_K):
            D, I = index.search(q_emb, top_n * 20)
            scores = {}
            for ds, idxs in zip(D, I):
                for sc, idx in zip(ds, idxs):
                    for pid in paper_lists[idx]:
                        scores[pid] = max(scores.get(pid, -1e9), sc)
            sorted_p = sorted(scores.items(), key=lambda x: -x[1])
            return [p for p, _ in sorted_p][:top_n], I

        # ── 3) Held-out evaluation
        raw_rec, adj_rec, chunk_rec, mrrs = [], [], [], []
        for rec in heldout:
            q_emb, I_ = encode_query(rec["query"]), None
            top_papers, I_ = retrieve_papers(q_emb)
            raw_rec.append(recall_at_k(rec["ground_truth"], top_papers))
            adj_rec.append(adj_recall_at_k(rec["ground_truth"], top_papers))
            mrrs.append(mrr_at_k(rec["ground_truth"], top_papers))

            hits = sum(
                1 for idx in I_[0][: TOP_K * 20]
                if any(pid in rec["ground_truth"] for pid in paper_lists[idx])
            )
            chunk_rec.append(hits / max(1, len(rec["ground_truth"])))

        logger.info(
            "→ Held-out Recall@%d raw|adj|chunk = %.4f | %.4f | %.4f; MRR@%d = %.4f",
            TOP_K, np.mean(raw_rec), np.mean(adj_rec), np.mean(chunk_rec),
            TOP_K, np.mean(mrrs)
        )

        # ── 4) Slice evaluation
        slice_scores = defaultdict(list)
        for rec in slices:
            tops, _ = retrieve_papers(encode_query(rec["query"]))
            slice_scores[rec["slice"]].append(adj_recall_at_k(rec["ground_truth"], tops))
        logger.info("→ Slice count: %d", len(slice_scores))

        # ── 5) Perturbation & failure checks
        perturb_ok = [
            retrieve_papers(encode_query(r["perturbed"]), 1)[0][0] in r["expected_papers"]
            for r in perturbs
        ]
        failure_ok = [
            retrieve_papers(encode_query(r["query"]), 1)[0][0] in r["correct_papers"]
            for r in failures
        ]
        logger.info("→ Perturbation pass rate = %.4f", np.mean(perturb_ok))
        logger.info("→ Failure-mode pass rate = %.4f", np.mean(failure_ok))

        # ── 6) Collect results
        results.append({
            "model":         col,
            "recall@10_raw": np.mean(raw_rec),
            "recall@10_adj": np.mean(adj_rec),
            "chunk_recall":  np.mean(chunk_rec),
            "MRR@10":        np.mean(mrrs),
            "slice_recalls": {s: np.mean(v) for s, v in slice_scores.items()},
            "perturb_acc":   np.mean(perturb_ok),
            "failure_acc":   np.mean(failure_ok),
        })

    # ──────────────────────────────────────────────────────────────────────────
    # SAVE & REPORT
    df_res = pd.DataFrame(results)
    df_res.to_json(os.path.join(OUTPUT_DIR, "model_comparison.jsonl"),
                   orient="records", lines=True)
    logger.info("Done! Results written to %s", os.path.join(OUTPUT_DIR, "model_comparison.jsonl"))
    print(df_res)

if __name__ == "__main__":
    main()
