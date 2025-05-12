#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Neura‑Scholar batched ONNX FastAPI service with terminal + Prometheus timing
# ─────────────────────────────────────────────────────────────────────────────
import os, time, asyncio, logging
from itertools import chain
from typing import List, Tuple, Callable

import numpy as np
import torch
import onnxruntime as ort
import mlflow, mlflow.onnx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from prometheus_client import (
    CollectorRegistry, Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

# ─────────────── logging ───────────────
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────── env / config ───────────────
PORT = int(os.getenv("PORT", 8000))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

EMBED_URI   = os.getenv("EMBEDDING_MODEL_URI",
                        "models:/distilbert-embedding-onnx-graph-opt/1")
SUMM_URI    = os.getenv("SUMMARIZATION_MODEL_URI",
                        "models:/facebook-bart-large/1")
LOCAL_EMBED = os.getenv("EMBEDDING_MODEL_PATH",
                        "/home/pb/projects/course/sem2/mlops/project/mlops/models/distilbert_opt.onnx")
LOCAL_SUMM  = os.getenv("SUMMARIZATION_MODEL_PATH",
                        "/home/pb/projects/course/sem2/mlops/project/mlops/models/bart_summarize.onnx")

USE_MLFLOW_EMBED = os.getenv("USE_MLFLOW_EMBED", "false").lower() in ("1","true","yes")
USE_MLFLOW_SUMM  = os.getenv("USE_MLFLOW_SUMM", "false").lower() in ("1","true","yes")

# ─────────────── load ONNX models ───────────────
try:
    embed_path = mlflow.onnx.load_model(EMBED_URI) if USE_MLFLOW_EMBED else LOCAL_EMBED
    log.info("Loaded embedding model from MLflow: %s", embed_path)
except Exception as e:
    log.warning("Falling back to local embedding model (%s) – %s", LOCAL_EMBED, e)
    embed_path = LOCAL_EMBED

summ_path = mlflow.onnx.load_model(SUMM_URI) if USE_MLFLOW_SUMM else LOCAL_SUMM
log.info("Loaded summarization model from MLflow: %s", summ_path)

providers  = ["CUDAExecutionProvider"]
embed_sess = ort.InferenceSession(embed_path,  providers=providers)
summ_sess  = ort.InferenceSession(summ_path,   providers=providers)
log.info("Providers in use → %s", embed_sess.get_providers())

# ─────────────── tokenizers ───────────────
EMBED_TOK = os.getenv("EMBEDDING_TOKENIZER_NAME",
                      "distilbert/distilbert-base-uncased")
SUMM_TOK  = os.getenv("SUMMARIZATION_TOKENIZER_NAME",
                      "facebook/bart-large")

embed_tok = Tokenizer.from_pretrained(EMBED_TOK)
summ_tok  = Tokenizer.from_pretrained(SUMM_TOK)
summ_tok_decode     = AutoTokenizer.from_pretrained(SUMM_TOK)    # use for decode

# ─────────────── Prometheus metrics ───────────────
REG     = CollectorRegistry()
REQS    = Counter ("api_requests_total", "", ["ep","method","status"], registry=REG)
LAT     = Histogram("request_seconds", "", ["ep"],                    registry=REG)
BATCH   = Histogram("batch_size",      "", ["model"],                 registry=REG)
STAGES  = Histogram("stage_seconds",   "", ["stage","model"],         registry=REG)
MODEL_SZ= Gauge    ("model_bytes",     "", ["model"],                 registry=REG)

MODEL_SZ.labels("embed").set(os.path.getsize(embed_path))
MODEL_SZ.labels("summ").set(os.path.getsize(summ_path))

# ─────────────── profiling decorator ───────────────
def profile(stage: str, model: str):
    def deco(fn: Callable):
        def wrap(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                d = time.perf_counter() - t0
                STAGES.labels(stage, model).observe(d)
                log.debug("Stage %-12s | %s | %.3f ms", stage, model, d * 1000)
        return wrap
    return deco

# ───────────────────────── EMBEDDING PIPELINE ─────────────────────────
EMBED_OUT = embed_sess.get_outputs()[0].name          # "last_hidden_state"
IDS_NAME  = embed_sess.get_inputs()[0].name           # "input_ids"
ATT_NAME  = embed_sess.get_inputs()[1].name           # "attention_mask"

class EmbedBatcher:
    def __init__(self, max_batch: int = 256, max_wait: float = 0.015):
        self.max_batch, self.max_wait = max_batch, max_wait
        self.q: List[Tuple[List[str], asyncio.Future]] = []
        self.lock, self.event = asyncio.Lock(), asyncio.Event()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    async def loop(self):
        while True:
            await self.event.wait()
            await asyncio.sleep(self.max_wait)
            async with self.lock:
                batch, self.q = self.q, []
                self.event.clear()
            if not batch:
                continue

            texts, futs = zip(*batch)
            flat = list(chain.from_iterable(texts))
            BATCH.labels("embed").observe(len(flat))

            @profile("tokenize", "embed")
            def _tok():
                encs = embed_tok.encode_batch(flat, add_special_tokens=True)
                L = 512
                ids = np.zeros((len(flat), L), dtype=np.int64)
                att = np.zeros_like(ids)
                for i, e in enumerate(encs):
                    l = min(len(e.ids), L)
                    ids[i, :l] = e.ids[:l]
                    att[i, :l] = 1
                return ids, att
            ids, att = _tok()

            @profile("onnx_infer", "embed")
            def _infer(inp_ids, inp_att):
                return embed_sess.run([EMBED_OUT],
                                      {IDS_NAME: inp_ids, ATT_NAME: inp_att})[0]
            outs = _infer(ids, att)

            @profile("gpu_pool", "embed")
            def _pool(h, m):
                with torch.no_grad():
                    h = torch.from_numpy(h).to(self.dev)
                    m = torch.from_numpy(m).to(self.dev).unsqueeze(-1)
                    emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
                    return emb.cpu().numpy().tolist()
            embs = _pool(outs, att)

            idx = 0
            for t, f in batch:
                f.set_result(embs[idx:idx + len(t)])
                idx += len(t)

    async def dispatch(self, txts: List[str]):
        fut = asyncio.get_running_loop().create_future()
        async with self.lock:
            self.q.append((txts, fut))
            self.event.set()
        return await fut

batcher = EmbedBatcher()

# ──────────────────────── SUMMARIZATION PIPELINE ───────────────────────
# inspect model I/O ------------------------------------------------------
inputs  = [i.name for i in summ_sess.get_inputs()]
outputs = [o.name for o in summ_sess.get_outputs()]

# encoder inputs are always first two
ENC_IDS_NAME, ENC_ATT_NAME = inputs[:2]

# some exports include decoder‑side feeds, others do not
DECODER_MODE = len(inputs) >= 4
if DECODER_MODE:
    DEC_IDS_NAME, DEC_ATT_NAME = inputs[2:4]
    log.info("Summ‑pipeline: using **greedy decode loop** (4‑input graph)")
else:
    log.info("Summ‑pipeline: model returns finished sequences (2‑input graph)")

SUMM_OUT = outputs[0]                # "logits"  OR "sequences"
BOS_ID   = summ_tok.token_to_id("<s>")   or 0
EOS_ID   = summ_tok.token_to_id("</s>")  or 2
MAX_GEN  = int(os.getenv("MAX_SUM_LEN", 60))

# ---------- helpers -----------------------------------------------------
@profile("summ_tokenize", "summ")
def _summ_tokenize(txts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encs = summ_tok.encode_batch(txts, add_special_tokens=True)
    L = 1024
    ids = np.zeros((len(txts), L), dtype=np.int64)
    att = np.zeros_like(ids)
    for i, e in enumerate(encs):
        l = min(len(e.ids), L)
        ids[i, :l] = e.ids[:l]
        att[i, :l] = 1
    return ids, att                                  # (B, L)

# ---------- two‑input (model‑does‑it‑all) path --------------------------
@profile("summ_infer", "summ")
def _model_generate(enc_ids, enc_att):
    return summ_sess.run([SUMM_OUT],
                         {ENC_IDS_NAME: enc_ids, ENC_ATT_NAME: enc_att})[0]  # (B,T)

# ---------- four‑input (greedy) path ------------------------------------
@profile("summ_step", "summ")
def _greedy_step(enc_ids, enc_att, dec_ids, dec_att):
    logits = summ_sess.run(
        [SUMM_OUT],
        {
            ENC_IDS_NAME: enc_ids,
            ENC_ATT_NAME: enc_att,
            DEC_IDS_NAME: dec_ids,
            DEC_ATT_NAME: dec_att,
        },
    )[0]                              # (B, T, V)
    return logits[:, -1, :]           # (B, V) last token

def _greedy_decode(enc_ids, enc_att):
    B = enc_ids.shape[0]
    dec_ids = np.full((B, 1), BOS_ID, dtype=np.int64)
    dec_att = np.ones_like(dec_ids, dtype=np.int64)
    finished = np.zeros(B, dtype=bool)

    for _ in range(MAX_GEN):
        next_tok = _greedy_step(enc_ids, enc_att, dec_ids, dec_att).argmax(-1, keepdims=True).astype(np.int64)
        dec_ids  = np.concatenate([dec_ids, next_tok], axis=1)
        dec_att  = np.concatenate([dec_att, np.ones_like(next_tok)], axis=1)
        finished |= (next_tok[:, 0] == EOS_ID)
        if finished.all(): break

    return dec_ids                    # (B, seq_len)

# ---------- final dispatcher --------------------------------------------
VOCAB_SIZE = summ_tok.get_vocab_size()
BOS_ID     = 0
EOS_ID     = 2                 # </s> for BART‑family models

@profile("summ_decode", "summ")
def _decode_sequences(seqs: np.ndarray) -> list[str]:
    outs = []
    for raw in seqs:
        # 1) make numeric, fix weird values
        clean = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) round to nearest int and clamp into vocab range
        seq = np.rint(clean).clip(0, VOCAB_SIZE - 1).astype(np.int64, copy=False)

        # 3) strip BOS/EOS
        if seq[0] == BOS_ID:
            seq = seq[1:]
        if EOS_ID in seq:
            seq = seq[: list(seq).index(EOS_ID)]

        outs.append(
            summ_tok_decode.decode(seq.tolist(), skip_special_tokens=True).strip()
        )
    return outs



def summarize_batch(texts: list[str]) -> list[str]:
    enc_ids, enc_att = _summ_tokenize(texts)

    if DECODER_MODE:
        seqs = _greedy_decode(enc_ids, enc_att)
    else:
        seqs = _model_generate(enc_ids, enc_att)      # already int64

    return _decode_sequences(seqs)


# ─────────────── FastAPI app ───────────────
app = FastAPI()

class EmbedReq(BaseModel): texts: List[str]
class EmbedResp(BaseModel): embeddings: List[List[float]]

class SummReq(BaseModel): texts: List[str]
class SummResp(BaseModel): summaries: List[str]

@app.on_event("startup")
async def _start():
    asyncio.create_task(batcher.loop())
    log.info("Embed worker on %s ready", batcher.dev)

@app.middleware("http")
async def _timer(req: Request, cnext):
    t = time.perf_counter()
    resp = await cnext(req)
    LAT.labels(req.url.path).observe(time.perf_counter() - t)
    return resp

# ——— embed endpoint ———
@app.post("/embed", response_model=EmbedResp)
async def embed(req: EmbedReq):
    REQS.labels("/embed", "POST", "200").inc()
    t0 = time.perf_counter()
    embs = await batcher.dispatch(req.texts)
    log.info("Request /embed batch=%d → %.3f ms",
             len(req.texts), (time.perf_counter() - t0) * 1000)
    return {"embeddings": embs}

# ——— summarization endpoint ———
@app.post("/summarize", response_model=SummResp)
async def summarize(req: SummReq):
    REQS.labels("/summarize", "POST", "200").inc()
    t0 = time.perf_counter()
    summaries = summarize_batch(req.texts)
    log.info("Request /summarize batch=%d → %.3f ms",
             len(req.texts), (time.perf_counter() - t0) * 1000)
    return {"summaries": summaries}

# ——— infra endpoints ———
@app.get("/metrics")
def metrics():
    return Response(generate_latest(REG), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────── run ───────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=PORT, log_level="info")
