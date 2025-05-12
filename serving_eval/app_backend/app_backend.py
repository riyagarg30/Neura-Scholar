from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
import httpx, asyncpg, os, re, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("DATABASE_URL", "postgresql://rg5073:rg5073pass@129.114.27.112:5432/cleaned_meta_data_db")
DB_URL = re.sub(r"\+\w+$", "", os.getenv("DATABASE_URL"))
EMBED_URL = os.getenv("EMBED_URL", "http://192.5.86.169/embed")
SUMMARY_URL = os.getenv("SUMMARY_URL", "http:/192.5.86.169/summarize")

class QueryRequest(BaseModel):
    query: str
    top_n: int = 5

class PaperResponse(BaseModel):
    paper_id: str
    paper_name: str
    authors: str
    doi: str
    journal_ref: str

class SearchResponse(BaseModel):
    results: list[PaperResponse]

similarity_gauges = [Gauge(f"cosine_similarity_rank_{i+1}", f"Cosine similarity for rank {i+1}") for i in range(5)]

app = FastAPI()

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.on_event("startup")
async def startup():
    logger.info("Connecting to Postgres…")
    try:
        app.state.pool = await asyncpg.create_pool(DB_URL)
        logger.info("Postgres pool ready")
    except Exception:
        logger.exception("Failed to connect to Postgres")
        raise

@app.on_event("shutdown")
async def shutdown():
    await app.state.pool.close()

async def call_embedding(text: str):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(EMBED_URL, json={"text": text})
            r.raise_for_status()
            return r.json()["embedding"]
    except httpx.HTTPError as exc:
        logger.exception("Embedding service call failed")
        raise HTTPException(status_code=502, detail="embedding service unavailable") from exc

async def call_summary(payload):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(SUMMARY_URL, json=payload)
            r.raise_for_status()
            return r.json()["summary"]
    except httpx.HTTPError as exc:
        logger.exception("Summary service call failed")
        raise HTTPException(status_code=502, detail="summary service unavailable") from exc

@app.post("/search", response_model=SearchResponse)
async def search(req: QueryRequest):
    logger.info("/search POST – query='%s'", req.query)
    try:
        embedding = await call_embedding(req.query)
        async with app.state.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT paper_id, chunk_id, chunk_data, chunk_embedding_768_graph <=> $1::vector AS distance
                FROM arxiv_chunks_eval_5
                ORDER BY distance
                LIMIT 500
                """,
                embedding,
            )
    except Exception:
        logger.exception("Database fetch error")
        raise HTTPException(status_code=500, detail="database error")

    top5 = rows[:5]
    for i, r in enumerate(top5):
        similarity_gauges[i].set(1 - r["distance"])

    papers = {}
    for row in rows:
        pid = row["paper_id"]
        if pid not in papers and len(papers) < req.top_n:
            papers[pid] = row
    if not papers:
        raise HTTPException(status_code=404, detail="no results")

    try:
        async with app.state.pool.acquire() as conn:
            meta_rows = await conn.fetch(
                """
                SELECT id, title, authors, journal_ref, doi, abstract
                FROM paper_metadata
                WHERE id = ANY($1::text[])
                """,
                list(papers.keys()),
            )
    except Exception:
        logger.exception("Metadata fetch error")
        raise HTTPException(status_code=500, detail="metadata lookup error")

    meta_map = {m["id"]: m for m in meta_rows}
    results = []
    for pid, info in papers.items():
        meta = meta_map.get(pid)
        if not meta:
            continue
        summary_input = {
            "query": req.query,
            "abstract": meta["abstract"],
            "chunk": info["chunk_data"],
        }
        summary = await call_summary(summary_input)
        results.append(
            {
                "paper_id": pid,
                "paper_name": meta["title"],
                "authors": meta["authors"],
                "doi": meta["doi"],
                "journal_ref": meta["journal_ref"],
                "summary": summary,
            }
        )
    return {"results": results}

@app.get("/search", response_model=SearchResponse)
async def search_get(q: str = Query(...), n: int = Query(5)):
    return await search(QueryRequest(query=q, top_n=n))