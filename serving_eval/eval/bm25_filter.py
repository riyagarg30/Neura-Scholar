# ── Cell 1: BM25‐Based Ground‐Truth Filtering (Thread‐Isolated, Verbose Logging) ──

import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text, create_engine

engine = create_engine(
    'postgresql+psycopg2://local:password@localhost:5433/mlops_local',
    max_overflow=0,   # disallow “extra” connections beyond pool_size
    pool_timeout=30,  # seconds to wait for an idle connection
)
PARQUET_FILE    = "arxiv_eval_results.parquet"
INDEX_DIR       = "whoosh_index"
TOP_N           = 3       # BM25 top-N chunks to consider
MIN_SCORE       = None    # e.g. 1.0 to enforce a minimum BM25 score
NUM_QUERIES     = 3       # Number of queries to generate per chunk
MAX_WORKERS = 4        # number of threads to use
BATCH_SIZE = 1000     # number of rows to update in each batch
LOG_INTERVAL = 10   # log progress every N chunks


from whoosh.index import open_dir
from whoosh.qparser import QueryParser
                  # ← your index folder
ix        = open_dir(INDEX_DIR)
parser    = QueryParser("chunk_data", schema=ix.schema)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── Prerequisites (define these before running) ─────────────────────────
# DB_URL, INDEX_DIR, TOP_N, MIN_SCORE, MAX_WORKERS, LOG_INTERVAL, BATCH_SIZE
# engine = create_engine(DB_URL)
# logger = logging.getLogger(__name__)
# Ensure handler is set up with appropriate level

# ── Thread initializer to open index & parser & keep searcher alive ────
def init_worker(idx_dir, top_n, min_score):
    global ix, parser, searcher, TOP_N, MIN_SCORE
    ix        = open_dir(idx_dir)
    parser    = QueryParser("chunk_data", schema=ix.schema)
    searcher  = ix.searcher()  # reused by threads
    TOP_N     = top_n
    MIN_SCORE = min_score

# ── Filtering function with profiling ───────────────────────────────────
def filter_one(task):
    chunk_id, cited_str, chunk_data = task

    # 1) Parse the Postgres‐style array manually
    if isinstance(cited_str, str) and cited_str.startswith("{") and cited_str.endswith("}"):
        original = [pid.strip() for pid in cited_str[1:-1].split(",") if pid.strip()]
    else:
        original = []

    # 2) Build your BM25 query
    snippet     = (chunk_data or "")[:200]
    q           = parser.parse(snippet)
    window_size = TOP_N * 10

    unique_pids = []

    try:
        with ix.searcher() as searcher:
            # Only retrieve docnums & scores—don’t force decompress of all stored fields
            hits = searcher.search(q, limit=window_size)
            for h in hits:
                # h.score is fine; to get paper_id we grab just that one field 
                pid = searcher.stored_fields(h.docnum)["paper_id"]
                if MIN_SCORE is not None and h.score < MIN_SCORE:
                    continue
                if pid not in unique_pids:
                    unique_pids.append(pid)
                    if len(unique_pids) >= TOP_N:
                        break
    except Exception as e:
        logger.warning("Chunk %s: BM25 search failed (%s), skipping", chunk_id, e)
        # leave unique_pids empty

    # 3) Filter your original citations
    filtered = [pid for pid in original if pid in unique_pids]
    logger.info("Chunk %s: kept %d/%d citations", chunk_id, len(filtered), len(original))
    return chunk_id, filtered


# ── Main parallel filtering & batched DB update ─────────────────────────
def run_filtering():
    # 1) Load tasks
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT chunk_id, paper_cited, chunk_data
              FROM arxiv_chunks_eval_4
             WHERE paper_cited IS NOT NULL
               AND paper_cited <> ''
        """)).fetchall()
    tasks = [(r.chunk_id, r.paper_cited, r.chunk_data) for r in rows]
    total = len(tasks)
    logger.info("Loaded %d chunks to filter", total)

    # 2) Parallel filtering
    results = []
    processed = 0
    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
        initargs=(INDEX_DIR, TOP_N, MIN_SCORE)
    ) as executor:
        future_to_chunk = {executor.submit(filter_one, t): t[0] for t in tasks}
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                cid, filtered = future.result()
                results.append((cid, filtered))
            except Exception as e:
                logger.error("Error processing chunk %s: %s", chunk_id, e)
            processed += 1

            logger.info("Progress: %d/%d chunks filtered", processed, total)

    # 3) Batched DB updates
    logger.info("Starting batched DB updates in batches of %d", BATCH_SIZE)
    with engine.begin() as conn:
        for i in range(0, total, BATCH_SIZE):
            batch = results[i:i+BATCH_SIZE]
            params = [
                {"pc": "{" + ",".join(filtered) + "}", "cid": cid}
                for cid, filtered in batch
            ]
            t_start = time.monotonic()
            conn.execute(
                text("UPDATE arxiv_chunks_eval_4 SET paper_cited = :pc WHERE chunk_id = :cid"),
                params
            )
            t_end = time.monotonic()
            logger.info(
                "DB batch %d-%d update took %.3fs",
                i+1, min(i+BATCH_SIZE, total), t_end - t_start
            )

    logger.info("Filtering and updates complete for %d chunks", total)

# Run the pipeline
if __name__ == "__main__":
    run_filtering()
