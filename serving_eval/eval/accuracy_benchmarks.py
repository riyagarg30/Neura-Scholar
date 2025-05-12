# Cell 1: Imports & config
import os, json
import numpy as np
import pandas as pd
import faiss
from sqlalchemy import create_engine

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# Your Postgres connection
DB_URL     = "postgresql+psycopg2://rg5073:rg5073pass@129.114.27.112:5432/cleaned_meta_data_db"
TABLE_NAME = "arxiv_chunks_eval_5"
EVAL_DIR   = "/home/pb/projects/course/sem2/mlops/project/mlops/Neura-Scholar/serving_eval/eval"        # where heldout.jsonl etc live
TOP_K      = 10            # how many papers to return & evaluate

# The embedding columns you’ve stored, e.g.:
MODEL_DETAILS = [
    {
        "column": "chunk_embedding_768",
        "model_path": "/home/pb/projects/course/sem2/mlops/project/mlops/models/distilbert.onnx",
    },
    {
        "column": "chunk_embedding_768_dyn",
        "model_path": "/home/pb/projects/course/sem2/mlops/project/mlops/models/distilbert_dyn.onnx",
    },
    {
        "column": "chunk_embedding_768_graph",
        "model_path": "/home/pb/projects/course/sem2/mlops/project/mlops/models/distilbert_opt.onnx",
    },
    # {
    #     "column": "chunk_embedding_768_static_h",
    #     "model_path": "/home/pb/projects/course/sem2/mlops/project/mlops/models/distilbert_static_heavy.onnx",
    # },
    # {
    #     "column": "chunk_embedding_768_static_m",
    #     "model_path": "/home/pb/projects/course/sem2/mlops/project/mlops/models/distilbert_static_moderate.onnx",
    # },
]

engine = create_engine(DB_URL, pool_timeout=30, max_overflow=0)
os.makedirs("indexes", exist_ok=True)


# Cell 2: Load your eval test records once
heldout = [json.loads(l) for l in open(f"{EVAL_DIR}/heldout.jsonl")]
slices  = [json.loads(l) for l in open(f"{EVAL_DIR}/slices.jsonl")]
perturbs= [json.loads(l) for l in open(f"{EVAL_DIR}/perturbations.jsonl")]
failures=[json.loads(l) for l in open(f"{EVAL_DIR}/failures.jsonl")]


# Cell 3: Helper to compute Recall@K and MRR@K
def recall_at_k(gt, pred, k=TOP_K):
    return int(any(p in gt for p in pred[:k]))

def mrr_at_k(gt, pred, k=TOP_K):
    for rank, p in enumerate(pred[:k], start=1):
        if p in gt:
            return 1.0/rank
    return 0.0


# ────────────────────────────────────────────────────────────────────────────
# Cell 4 — Evaluation loop   (40 % noise in ground‑truth handled)
# ────────────────────────────────────────────────────────────────────────────
import ast, os, time, math
from collections import defaultdict
import onnxruntime as ort
from transformers import AutoTokenizer

# ---------- helpers --------------------------------------------------------
def recall_at_k(gt: list[str], retrieved: list[str], k:int = TOP_K) -> float:
    if not gt:
        return 0.0
    hit = sum(1 for p in retrieved[:k] if p in gt)
    return hit / len(gt)

def adj_recall_at_k(gt: list[str], retrieved: list[str], k:int = TOP_K) -> float:
    if not gt:
        return 0.0
    effective_rel = max(1, math.ceil(0.4 * len(gt)))          # assume 40 % relev.
    hit = sum(1 for p in retrieved[:k] if p in gt)
    return hit / effective_rel

# ---------- main loop ------------------------------------------------------
results        = []
EMBED_TOKENIZER = "distilbert/distilbert-base-uncased"

for mdl in MODEL_DETAILS:
    col       = mdl["column"]
    onnx_path = mdl["model_path"]
    meta_file = f"indexes/{col}_meta.jsonl"
    idx_file  = f"indexes/{col}.index"

    logger.info(f"=== Evaluating `{col}` with ONNX [{onnx_path}] ===")

    # ── 1.  Load / build FAISS index ───────────────────────────────────────
    if os.path.exists(idx_file):
        t0 = time.perf_counter()
        index = faiss.read_index(idx_file)
        logger.info("Loaded index `%s` in %.2fs", idx_file, time.perf_counter()-t0)

        if os.path.exists(meta_file):
            df_emb = pd.read_json(meta_file, lines=True)
        else:
            df_emb = pd.read_sql(
                f"SELECT chunk_id, paper_cited FROM {TABLE_NAME}",
                con=engine
            )
            df_emb["paper_list"] = (
                df_emb["paper_cited"].str.strip("{}").str.split(",")
            )
            logger.info("Meta file missing – reloaded chunk_ids+paper_lists from Postgres")

    else:
        t0  = time.perf_counter()
        sql = f"SELECT chunk_id, paper_cited, chunk_data, {col} FROM {TABLE_NAME}"
        df_emb = pd.read_sql(sql, con=engine)
        df_emb["paper_list"] = (
            df_emb["paper_cited"].str.strip("{}").str.split(",")
        )

        df_emb["emb_list"] = df_emb[col].apply(
            lambda v: v if isinstance(v, list) else ast.literal_eval(v)
        )
        embs = np.array(df_emb["emb_list"].tolist(), dtype="float32")
        faiss.normalize_L2(embs)

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        os.makedirs("indexes", exist_ok=True)
        faiss.write_index(index, idx_file)
        df_emb[["chunk_id", "paper_list"]].to_json(
            meta_file, orient="records", lines=True
        )
        logger.info("Built & saved index `%s` in %.2fs", col, time.perf_counter()-t0)

    chunk_ids   = df_emb["chunk_id"].tolist()
    paper_lists = df_emb["paper_list"].tolist()

    # ── 2.  ONNX session + tokenizer  +  sanity‑check───────────────────────
    t1       = time.perf_counter()
    ort_sess = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider"]
    )
    tokenizer = AutoTokenizer.from_pretrained(EMBED_TOKENIZER)
    logger.info("Loaded ONNX session + tokenizer in %.2fs", time.perf_counter()-t1)

    # --- 2a.  encode_query helper -----------------------------------------
    def encode_query(text: str):
        toks = tokenizer(
            text,
            return_tensors="np",
            max_length=300,
            padding="max_length",
            truncation=True,
        )
        out = ort_sess.run(
            None,
            {
                ort_sess.get_inputs()[0].name: toks["input_ids"].astype("int64"),
                ort_sess.get_inputs()[1].name: toks["attention_mask"].astype("int64"),
            },
        )[0]
        mask = np.expand_dims(toks["attention_mask"], -1).astype("float32")
        emb  = (out * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        faiss.normalize_L2(emb)
        return emb.astype("float32")

    # --- 2b.  *sanity‑check* embedding path -------------------------------
    sample = pd.read_sql(
        f"SELECT chunk_data, {col} FROM {TABLE_NAME} LIMIT 1",
        con=engine,
    ).iloc[0]

    offline_vec = np.array(
        sample[col] if isinstance(sample[col], list) else ast.literal_eval(sample[col]),
        dtype="float32",
    )
    offline_vec = offline_vec / np.linalg.norm(offline_vec)  # ensure unit
    online_vec  = encode_query(sample["chunk_data"])[0]

    cosine_sim  = float(np.dot(offline_vec, online_vec))
    if cosine_sim < 0.90:
        logger.warning(
            "Embedding mismatch for `%s` (cos = %.3f).",
            col,
            cosine_sim,
        )
    else:
        logger.info("Embedding sanity-check passed (cos = %.3f)", cosine_sim)

    # --- retrieval helper --------------------------------------------------
    def retrieve_papers(q_emb, top_n=TOP_K):
        D, I = index.search(q_emb, top_n * 20)        # bigger over‑fetch
        paper2score = {}
        for scores, idxs in zip(D, I):
            for sc, idx in zip(scores, idxs):
                for pid in paper_lists[idx]:
                    paper2score[pid] = max(paper2score.get(pid, -1e9), sc)
        return [p for p, _ in sorted(paper2score.items(), key=lambda x: -x[1])][:top_n], I

    # ── 3.  Held‑out evaluation (paper + chunk recall) ─────────────────────
    raw_rec   = []
    adj_rec   = []
    chunk_rec = []          # NEW
    mrrs      = []

    for rec in heldout:
        q_emb          = encode_query(rec["query"])
        top_papers, I  = retrieve_papers(q_emb)
        raw_rec.append(recall_at_k(rec["ground_truth"], top_papers))
        adj_rec.append(adj_recall_at_k(rec["ground_truth"], top_papers))
        mrrs.append(mrr_at_k(rec["ground_truth"], top_papers))

        # chunk‑level recall ⟶ any retrieved *chunk* from a GT paper?
        top_chunk_hit = sum(
            1
            for idx in I[0][: TOP_K * 20]
            if any(pid in rec["ground_truth"] for pid in paper_lists[idx])
        )
        chunk_rec.append(top_chunk_hit / max(1, len(rec["ground_truth"])))

    logger.info(
        "→ Held‑out Recall@%d (raw|adj|chunk) = %.4f | %.4f | %.4f ,  MRR@%d = %.4f",
        TOP_K,
        np.mean(raw_rec),
        np.mean(adj_rec),
        np.mean(chunk_rec),
        TOP_K,
        np.mean(mrrs),
    )

    # ── 4.  Slice evaluation (adjusted recall) ─────────────────────────────
    slice_scores = defaultdict(list)
    for rec in slices:
        tops, _ = retrieve_papers(encode_query(rec["query"]))
        slice_scores[rec["slice"]].append(
            adj_recall_at_k(rec["ground_truth"], tops)
        )
    logger.info("→ Slice count: %d", len(slice_scores))

    # ── 5.  Perturbation / failure‑mode checks (unchanged) ─────────────────
    perturb_ok = [
        retrieve_papers(encode_query(r["perturbed"]), 1)[0][0] in r["expected_papers"]
        for r in perturbs
    ]
    logger.info("→ Perturbation pass rate %.4f", np.mean(perturb_ok))

    failure_ok = [
        retrieve_papers(encode_query(r["query"]), 1)[0][0] in r["correct_papers"]
        for r in failures
    ]
    logger.info("→ Failure‑mode pass rate %.4f", np.mean(failure_ok))

    # ── 6.  Collect results  ───────────────────────────────────────────────
    results.append(
        {
            "model":           col,
            "recall@10_raw":   np.mean(raw_rec),
            "recall@10_adj":   np.mean(adj_rec),
            "chunk_recall":    np.mean(chunk_rec),
            "MRR@10":          np.mean(mrrs),
            "slice_recalls":   {s: np.mean(v) for s, v in slice_scores.items()},
            "perturb_acc":     np.mean(perturb_ok),
            "failure_acc":     np.mean(failure_ok),
        }
    )

# ---------- save & show ----------------------------------------------------
df_res = pd.DataFrame(results)
os.makedirs("eval", exist_ok=True)
df_res.to_json("eval/model_comparison.jsonl", orient="records", lines=True)
logger.info("Done! Results written to eval/model_comparison.jsonl")
df_res


