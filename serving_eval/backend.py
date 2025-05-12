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

os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.27.112:8000"
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

EMBED_URI   = os.getenv("EMBEDDING_MODEL_URI",
                        "models:/distilbert-embedding-onnx/1")
SUMM_URI    = os.getenv("SUMMARIZATION_MODEL_URI",
                        "models:/facebook-bart-large/1")
LOCAL_EMBED = os.getenv("EMBEDDING_MODEL_PATH",
                        "/home/pb/projects/course/sem2/mlops/project/mlops/models/distilbert_graph_opt.onnx")
LOCAL_SUMM  = os.getenv("SUMMARIZATION_MODEL_PATH",
                        "/home/pb/projects/course/sem2/mlops/project/mlops/models/bart_summarize.onnx")
USE_MLFLOW_EMBED  = os.getenv("USE_MLFLOW_EMBED", "false").lower() in ("1","true","yes")
USE_MLFLOW_SUMM  = os.getenv("USE_MLFLOW_SUMM", "false").lower() in ("1","true","yes")


# ─────────────── load ONNX models ───────────────
try:
    embed_path = mlflow.onnx.load_model(EMBED_URI) if USE_MLFLOW_EMBED else LOCAL_EMBED
except:
    log.warning("Failed to load embedding model from MLflow, using local path")
    embed_path = LOCAL_EMBED
summ_path  = mlflow.onnx.load_model(SUMM_URI) if USE_MLFLOW_SUMM else LOCAL_SUMM

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
embed_sess = ort.InferenceSession(embed_path, providers=providers)
summ_sess  = ort.InferenceSession(summ_path,  providers=providers)

log.info("Providers in use → %s", embed_sess.get_providers())

# ─────────────── tokenizers ───────────────
EMBED_TOK = os.getenv("EMBEDDING_TOKENIZER_NAME",
                      "distilbert/distilbert-base-uncased")
embed_tok = Tokenizer.from_pretrained(EMBED_TOK)

# ─────────────── Prometheus metrics ───────────────
REG     = CollectorRegistry()
REQS    = Counter("api_requests_total", "", ["ep","method","status"], registry=REG)
LAT     = Histogram("request_seconds", "", ["ep"], registry=REG)
BATCH   = Histogram("batch_size", "", ["model"], registry=REG)
STAGES  = Histogram("stage_seconds", "", ["stage","model"], registry=REG)
MODEL_SZ= Gauge("model_bytes", "", ["model"], registry=REG)

MODEL_SZ.labels("embed").set(os.path.getsize(embed_path))

# ─────────────── profiling decorator ───────────────
def profile(stage:str, model:str):
    def deco(fn:Callable):
        def wrap(*a, **kw):
            t0=time.perf_counter()
            try: return fn(*a, **kw)
            finally:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                d=time.perf_counter()-t0
                STAGES.labels(stage,model).observe(d)
                log.debug("Stage %-12s | %s | %.3f ms", stage, model, d*1000)
        return wrap
    return deco

# ─────────────── batching worker ───────────────
EMBED_OUT = embed_sess.get_outputs()[0].name     # usually "last_hidden_state"
IDS_NAME  = embed_sess.get_inputs()[0].name      # "input_ids"
ATT_NAME  = embed_sess.get_inputs()[1].name      # "attention_mask"

class EmbedBatcher:
    def __init__(self, max_batch:int=256, max_wait:float=0.015):
        self.max_batch, self.max_wait = max_batch, max_wait
        self.q : List[Tuple[List[str], asyncio.Future]] = []
        self.lock  , self.event = asyncio.Lock(), asyncio.Event()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    async def loop(self):
        while True:
            await self.event.wait()
            await asyncio.sleep(self.max_wait)
            async with self.lock:
                batch, self.q = self.q, []
                self.event.clear()
            if not batch: continue

            texts,futs = zip(*batch)
            flat=list(chain.from_iterable(texts))
            BATCH.labels("embed").observe(len(flat))

            # 1‑tokenize -------------------------------------------------------
            @profile("tokenize","embed")
            def _tok():
                encs = embed_tok.encode_batch(flat, add_special_tokens=True)
                L=512
                ids = np.zeros((len(flat),L), dtype=np.int64)   # MUST be int64
                att = np.zeros_like(ids)
                for i,e in enumerate(encs):
                    l=min(len(e.ids),L)
                    ids[i,:l]=e.ids[:l]; att[i,:l]=1
                return ids,att
            ids,att=_tok()

            # 2‑onnx infer + pool ----------------------------------------------
            @profile("onnx_infer","embed")
            def _infer(inp_ids, inp_att):
                return embed_sess.run([EMBED_OUT],
                        {IDS_NAME:inp_ids, ATT_NAME:inp_att})[0]
            outs=_infer(ids,att)

            @profile("gpu_pool","embed")
            def _pool(h, m):
                with torch.no_grad():
                    h=torch.from_numpy(h).to(self.dev)
                    m=torch.from_numpy(m).to(self.dev).unsqueeze(-1)
                    emb=(h*m).sum(1)/m.sum(1).clamp(min=1)
                    return emb.cpu().numpy().tolist()
            embs=_pool(outs,att)

            # 3‑fan‑out ---------------------------------------------------------
            idx=0
            for t,f in batch:
                f.set_result(embs[idx:idx+len(t)])
                idx+=len(t)

    async def dispatch(self, txts:List[str]):
        fut=asyncio.get_running_loop().create_future()
        async with self.lock:
            self.q.append((txts,fut))
            if len(self.q)>=self.max_batch: self.event.set()
            else: self.event.set()
        return await fut

batcher=EmbedBatcher()

# ─────────────── FastAPI app ───────────────
app=FastAPI()

class Req(BaseModel):  texts:List[str]
class Resp(BaseModel): embeddings:List[List[float]]

@app.on_event("startup")
async def _start():
    asyncio.create_task(batcher.loop())
    log.info("Embed worker on %s ready", batcher.dev)

@app.middleware("http")
async def _timer(req:Request,cnext):
    t=time.perf_counter(); r=await cnext(req)
    LAT.labels(req.url.path).observe(time.perf_counter()-t); return r

@app.post("/embed",response_model=Resp)
async def embed(req:Req):
    REQS.labels("/embed","POST","200").inc()
    t0=time.perf_counter()
    embs=await batcher.dispatch(req.texts)
    log.info("Request /embed batch=%d → %.3f ms",
             len(req.texts),(time.perf_counter()-t0)*1000)
    return {"embeddings":embs}

@app.get("/metrics")
def metrics(): return Response(generate_latest(REG),media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health(): return {"status":"ok"}

# ─────────────── run ───────────────
if __name__=="__main__":
    import uvicorn
    uvicorn.run("backend:app",host="0.0.0.0",port=PORT,log_level="info")
