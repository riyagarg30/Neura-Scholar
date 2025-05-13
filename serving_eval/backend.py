#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Neura‑Scholar batched FastAPI service (ONNX embeddings + HF summaries)
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
from tokenizers import Tokenizer                        # (for embeddings only)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # ✨ NEW
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

LOCAL_EMBED = os.getenv("EMBEDDING_MODEL_PATH",
                        "/home/pb/projects/course/sem2/mlops/project/mlops/models/distilbert_opt.onnx")

USE_MLFLOW_EMBED = os.getenv("USE_MLFLOW_EMBED", "false").lower() in ("1","true","yes")

# ─────────────── load embedding ONNX ───────────────
try:
    embed_path = mlflow.onnx.load_model(EMBED_URI) if USE_MLFLOW_EMBED else LOCAL_EMBED
    log.info("Loaded embedding model from MLflow: %s", embed_path)
except Exception as e:
    log.warning("Falling back to local embedding model (%s) – %s", LOCAL_EMBED, e)
    embed_path = LOCAL_EMBED

providers  = ["CUDAExecutionProvider"]
embed_sess = ort.InferenceSession(embed_path, providers=providers)
log.info("Providers in use → %s", embed_sess.get_providers())

# ─────────────── tokenizers ───────────────
EMBED_TOK = os.getenv("EMBEDDING_TOKENIZER_NAME",
                      "distilbert/distilbert-base-uncased")
embed_tok = Tokenizer.from_pretrained(EMBED_TOK)

# ── HF summariser (no ONNX) ──
SUMM_MODEL_NAME = os.getenv("SUMMARIZATION_MODEL_NAME", "facebook/bart-large")
SUMM_MAX_LEN    = int(os.getenv("SUMMARIZATION_MAX_LEN", 128))
device = "cuda" if torch.cuda.is_available() else "cpu"

summ_tok   = AutoTokenizer.from_pretrained(SUMM_MODEL_NAME)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(SUMM_MODEL_NAME).to(device)
summ_model.eval()
log.info("Loaded summariser %s on %s", SUMM_MODEL_NAME, device)

# ─────────────── Prometheus metrics ───────────────
REG     = CollectorRegistry()
REQS    = Counter ("api_requests_total", "", ["ep","method","status"], registry=REG)
LAT     = Histogram("request_seconds", "", ["ep"],                    registry=REG)
BATCH   = Histogram("batch_size",      "", ["model"],                 registry=REG)
STAGES  = Histogram("stage_seconds",   "", ["stage","model"],         registry=REG)
MODEL_SZ= Gauge    ("model_bytes",     "", ["model"],                 registry=REG)

MODEL_SZ.labels("embed").set(os.path.getsize(embed_path))
# We skip a size gauge for the HF model (weights live in multiple shards).

# ─────────────── profiling decorator ───────────────
def profile(stage: str, model: str):
    def deco(fn: Callable):
        def wrap(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                STAGES.labels(stage, model).observe(time.perf_counter() - t0)
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
        self.dev = device

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

# ──────────────────────── SUMMARIZATION PIPELINE (HF) ───────────────────────
@profile("summ_tokenize", "summ")
def _summ_tokenize(txts: List[str]):
    return summ_tok(txts,
                    truncation=True,
                    padding=True,
                    max_length=1024,
                    return_tensors="pt").to(device)

@profile("summ_infer", "summ")
@torch.inference_mode()
def _summ_infer(enc):
    return summ_model.generate(**enc,
                               max_length=SUMM_MAX_LEN,
                               num_beams=4,
                               do_sample=False)

@profile("summ_decode", "summ")
def _summ_decode(seqs: torch.Tensor):
    return summ_tok.batch_decode(seqs,
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)

def summarize_batch(texts: List[str]) -> List[str]:
    enc  = _summ_tokenize(texts)
    seqs = _summ_infer(enc)
    return _summ_decode(seqs)

# ─────────────── FastAPI app ───────────────
app = FastAPI()

class EmbedReq(BaseModel): texts: List[str]
class EmbedResp(BaseModel): embeddings: List[List[float]]

class SummReq(BaseModel):  texts: List[str]
class SummResp(BaseModel): summaries: List[str]

@app.on_event("startup")
async def _start():
    asyncio.create_task(batcher.loop())
    log.info("Embed worker on %s ready", batcher.dev)

@app.middleware("http")
async def _timer(req: Request, cnext):
    t0 = time.perf_counter()
    resp = await cnext(req)
    LAT.labels(req.url.path).observe(time.perf_counter() - t0)
    return resp

@app.post("/embed", response_model=EmbedResp)
async def embed(req: EmbedReq):
    REQS.labels("/embed", "POST", "200").inc()
    return {"embeddings": await batcher.dispatch(req.texts)}

@app.post("/summarize", response_model=SummResp)
async def summarize(req: SummReq):
    REQS.labels("/summarize", "POST", "200").inc()
    return {"summaries": summarize_batch(req.texts)}

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
