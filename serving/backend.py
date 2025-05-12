import os
import asyncio
import time
from itertools import chain
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import mlflow
import mlflow.onnx
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoTokenizer
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import logging

# ─────────── Logging ───────────
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ─────────── MLflow & Config ───────────
os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.27.112:8000"
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

EMBED_MODEL_URI     = os.getenv("EMBEDDING_MODEL_URI",     "models:/distilbert-embedding-onnx/1")
SUMMARIZE_MODEL_URI = os.getenv("SUMMARIZATION_MODEL_URI", "models:/bart-summarize-onnx/1")

LOCAL_EMBED_PATH     = os.getenv("EMBEDDING_MODEL_PATH",     "models/distilbert.onnx")
LOCAL_SUMMARIZE_PATH = os.getenv("SUMMARIZATION_MODEL_PATH", "models/bart_summarize.onnx")

USE_MLFLOW = os.getenv("USE_MLFLOW", "false").lower() in ("1", "true", "yes")

# ─────────── Prometheus Metrics ───────────
registry = CollectorRegistry()
MODEL_LOAD_SUCCESS = Counter("model_load_success_total","…",["model_type"],registry=registry)
MODEL_FILE_SIZE    = Gauge(  "model_file_size_bytes","…",["model_type"],registry=registry)
API_REQUESTS       = Counter("api_requests_total","…",["endpoint","method","http_status"],registry=registry)
INFERENCE_DURATION = Histogram("model_inference_seconds","…",["model_type"],registry=registry)
INPUT_SIZE         = Histogram("model_input_size","…",["model_type"],registry=registry)
OUTPUT_SIZE        = Histogram("model_output_size","…",["model_type"],registry=registry)

# ─────────── Load ONNX models ───────────
if USE_MLFLOW:
    embed_onnx_path = mlflow.onnx.load_model(EMBED_MODEL_URI)
    summarize_onnx_path = mlflow.onnx.load_model(SUMMARIZE_MODEL_URI)
else:
    embed_onnx_path = LOCAL_EMBED_PATH
    summarize_onnx_path = LOCAL_SUMMARIZE_PATH

providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
embed_sess     = ort.InferenceSession(embed_onnx_path,     providers=providers)
summarize_sess = ort.InferenceSession(summarize_onnx_path, providers=providers)

# record model-load metrics
MODEL_FILE_SIZE.labels("embed").set(os.path.getsize(embed_onnx_path))
MODEL_LOAD_SUCCESS.labels("embed").inc()
MODEL_FILE_SIZE.labels("summarization").set(os.path.getsize(summarize_onnx_path))
MODEL_LOAD_SUCCESS.labels("summarization").inc()

# ─────────── Tokenizers ───────────
EMBED_TOKENIZER     = os.getenv("EMBEDDING_TOKENIZER_NAME",     "distilbert-base-uncased")
SUMMARIZE_TOKENIZER = os.getenv("SUMMARIZATION_TOKENIZER_NAME", "facebook/bart-large")

embed_tokenizer     = AutoTokenizer.from_pretrained(EMBED_TOKENIZER)
summarize_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZE_TOKENIZER)

# ─────────── FastAPI setup ───────────
app = FastAPI()

class EmbedRequest(BaseModel):
    texts: List[str]
class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
class SummarizeRequest(BaseModel):
    text: str
class SummarizeResponse(BaseModel):
    summary: str

# ─────────── Batcher ───────────
class EmbedBatcher:
    def __init__(self, sess: ort.InferenceSession, tok: AutoTokenizer,
                 max_batch_size=32, max_wait_time=0.01):
        self.sess = sess
        self.tok = tok
        self.max_batch = max_batch_size
        self.wait_time = max_wait_time

        self.queue: List[Tuple[List[str], asyncio.Future]] = []
        self.lock = asyncio.Lock()
        self.trigger = asyncio.Event()

    async def _worker(self):
        while True:
            await self.trigger.wait()
            await asyncio.sleep(self.wait_time)

            async with self.lock:
                batch = self.queue
                self.queue = []
                self.trigger.clear()
            if not batch:
                continue

            texts_list, futures = zip(*batch)
            all_texts = list(chain.from_iterable(texts_list))

            # tokenize (DistilBERT max length = 512)
            enc = self.tok(
                all_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )

            # run ONNX
            output_names = [o.name for o in self.sess.get_outputs()]
            input_feed = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            raw_outs = self.sess.run(output_names, input_feed)[0]  # shape: (batch, seq, hidden)

            # mean-pool over seq dimension
            mask = enc["attention_mask"][..., None]                # (batch, seq, 1)
            summed = (raw_outs * mask).sum(axis=1)                 # (batch, hidden)
            counts = mask.sum(axis=1).clip(min=1e-9)               # (batch, 1)
            all_embs = (summed / counts).tolist()                 # List[List[float]]

            # split back to each original request
            idx = 0
            for texts, fut in batch:
                n = len(texts)
                fut.set_result(all_embs[idx:idx+n])
                idx += n

    async def predict(self, texts: List[str]) -> List[List[float]]:
        fut = asyncio.get_running_loop().create_future()
        async with self.lock:
            self.queue.append((texts, fut))
            self.trigger.set()
        return await fut

# instantiate & start on app startup
embed_batcher = EmbedBatcher(embed_sess, embed_tokenizer, max_batch_size=64, max_wait_time=0.02)

@app.on_event("startup")
async def start_worker():
    asyncio.create_task(embed_batcher._worker())

# ─────────── Helpers ───────────
def run_embedding(texts: List[str]) -> List[List[float]]:
    start = time.time()
    API_REQUESTS.labels("/embed","POST","200").inc()
    INPUT_SIZE.labels("embed").observe(len(texts))

    enc = embed_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    raw_outs = embed_sess.run(
        [o.name for o in embed_sess.get_outputs()],
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    )[0]
    # pool
    mask   = enc["attention_mask"][..., None]
    summed = (raw_outs * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    embs   = (summed / counts).tolist()

    dur = time.time() - start
    INFERENCE_DURATION.labels("embed").observe(dur)
    OUTPUT_SIZE.labels("embed").observe(len(embs))
    return embs

def run_summary(text: str) -> str:
    start = time.time()
    API_REQUESTS.labels("/summarize","POST","200").inc()
    INPUT_SIZE.labels("summarization").observe(1)

    toks = summarize_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=summarize_tokenizer.model_max_length or 1024,
        return_tensors="np"
    )
    output_ids = [[summarize_tokenizer.pad_token_id]]
    inputs = {k: v for k, v in toks.items()}
    for _ in range(128):
        inputs["input_ids"] = np.array(output_ids)
        logits = summarize_sess.run(None, inputs)[0]
        nxt = int(logits[0, -1].argmax())
        if nxt == summarize_tokenizer.eos_token_id:
            break
        output_ids[0].append(nxt)

    summary = summarize_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    dur = time.time() - start
    INFERENCE_DURATION.labels("summarization").observe(dur)
    OUTPUT_SIZE.labels("summarization").observe(len(output_ids[0]))
    return summary

# ─────────── Endpoints ───────────
@app.post("/batch-embed", response_model=EmbedResponse)
async def batch_embed(req: EmbedRequest):
    start = time.time()
    API_REQUESTS.labels("/batch-embed","POST","200").inc()
    INPUT_SIZE.labels("embed").observe(len(req.texts))

    embs = await embed_batcher.predict(req.texts)

    dur = time.time() - start
    INFERENCE_DURATION.labels("embed").observe(dur)
    OUTPUT_SIZE.labels("embed").observe(len(embs))
    return {"embeddings": embs}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    return {"embeddings": run_embedding(req.texts)}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    return {"summary": run_summary(req.text)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

# ─────────── Logging Middleware ───────────
@app.middleware("http")
async def log_requests(request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    elapsed = (time.time() - t0) * 1000
    print(f"{request.method} {request.url.path} completed_in={elapsed:.1f}ms status={resp.status_code}")
    return resp

# ─────────── Run via Uvicorn ───────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
