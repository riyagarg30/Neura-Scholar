## Project: Semantic Search Evaluation & Serving

This repository contains the end-to-end tooling and notebooks for building, optimizing, and evaluating a chunk-based semantic search system over an ArXiv corpus. Our goal is to return **top-K papers** in response to a user query by embedding chunks of paper text, indexing them, and collapsing chunk hits into paper-level candidates.

---

## Serving and Evaluation Structure

.
├── eval_data.ipynb # Pulls from Postgres → JSONL/NPZ eval datasets
├── bm25_filter.py # BM25 index creation (prototype; not used)
├── eval_gen.ipynb # Generates robustness & slice test cases
├── onnx_model.ipynb # Converts PyTorch encoder → ONNX & quantizes
├── generate-embeddings-new.ipynb # Batches chunk_data → embeddings via API
└── README.md # This document


---

## Artifact Summaries

### 1. `eval_data.ipynb`  
**What it does:**  
- Connects to `arxiv_chunks_with_metadata` in Postgres.  
- Parses the string columns:  
  - `query` → list of queries per chunk  
  - `paper_cited` → list of ground-truth paper IDs  
  - `categories` → list of ArXiv category tags  
- Flattens into one row per `(chunk, query)` pairing.  
- Splits into:  
  1. **Held-out** (10%) → `eval/heldout.jsonl` (for retrieval metrics).  
  2. **Dev + slices** (10%) → `eval/slices.jsonl`, tagging by first category and year.  
  3. **Perturbations** (~200 bases × 2 variants) → `eval/perturbations.jsonl`.  
  4. **Failure modes** (~50 “hard” cases, e.g. contains “CRISPR”) → `eval/failures.jsonl`.  
  5. **Drift reference** (2 000 random chunks) → `eval/drift_reference.npz`.  

**Why:**  
- Creates **versioned**, reproducible evaluation datasets.  
- Separates concerns: retrieval (held-out), fairness/robustness (slices, perturbations), known failures, and data drift.

---

### 2. `bm25_filter.py` (Not Used)  
**What it does:**  
- Prototype code to build a Whoosh/BM25 index over chunks and pre-filter by text matching.

**Why it exists:**  
- Explored a classic IR baseline before moving fully to vector search.  
- Ultimately superseded by embedding + FAISS pipeline, but kept for reference.

---

### 3. `eval_gen.ipynb`  
**What it does:**  
- Generates the **pytest-style** test assertions for all the JSONL files created by `eval_data.ipynb`.  
- Defines helper functions:  
  - `retrieve_topk_papers(query, k)` collapsing chunk hits → paper scores.  
  - `recall_at_k`, `mrr_at_k`, etc.  
- Runs and logs per-slice Recall/MRR, asserts top-1 correctness on perturbations & failures.

**Why:**  
- Automates offline evaluation gating: any new model version must pass these tests before deployment.

---

### 4. `onnx_model.ipynb`  
**What it does:**  
- Exports your fine-tuned PyTorch bi-encoder (e.g. DistilBERT) into ONNX (opset 17).  
- Applies graph optimizations (ORT_ENABLE_EXTENDED).  
- Performs dynamic and static INT8 quantization (using ONNX Runtime).  
- Benchmarks each variant for size, single-sample latency, batch throughput on CPU.

**Why:**  
- Model-level optimizations reduce inference cost.  
- Quantization unlocks CPU speedups and smaller footprints.  
- Graph optimizations fuse operators for better performance.

---

### 5. `generate-embeddings-new.ipynb`  
**What it does:**  
- Reads in your flattened `(chunk_id, chunk_data)` list from `eval_data.ipynb`.  
- Batches requests to your FastAPI `/embed/batch` endpoint.  
- Saves out:  
  - A FAISS index of all chunk embeddings (`.index` file).  
  - A `chunk_ids.json` mapping FAISS positions → `chunk_id`.  

**Why:**  
- Ensures your offline eval uses exactly the same embedding service you’ll deploy.  
- Decouples embedding generation from in-process calls, so you can benchmark the API path under load.

---

## How to Use

1. **Configure** your Postgres connection in `eval_data.ipynb` (cell 1).  
2. **Run** `eval_data.ipynb` end-to-end → produces `eval/*.jsonl` and `eval/drift_reference.npz`.  
3. **Train / convert** your bi-encoder, then run `onnx_model.ipynb` to produce optimized ONNX files.  
4. **Start** your FastAPI embedding service (with the ONNX model loaded).  
5. **Run** `generate-embeddings-new.ipynb` to build your FAISS index under `eval/`.  
6. **Open** `eval_gen.ipynb` and point it at:  
   - `eval/heldout.jsonl`, `slices.jsonl`, etc.  
   - `eval/chunks.index` + `chunk_ids.json`  
   - Your embedding service client (or load ONNX in-process)  
   Then execute all cells to compute metrics and test pass/fail.

---

## Key Considerations & “Whys”

- **Paper-level retrieval**: chunks map to papers; we collapse chunk scores into per-paper scores via a max-score aggregator.  
- **Reproducibility**: all random seeds fixed; JSONL + NPZ artifacts are versionable.  
- **Modularity**: data prep, model conversion, embedding generation, and evaluation are separate notebooks––you can swap in alternative methods at each stage.  
- **Performance**: ONNX + quantization reduce latency; FAISS FlatIP enables sub-millisecond similarity search on thousands of chunks.  
- **Quality**: slice-based recall, perturbation stability, and known failure tests guard against regressions and biases.  
- **Monitoring**: drift reference embeddings feed an MMD drift detector in production.

---

## Other Parts

- **Automate** this pipeline with Airflow or Argo:  
  - Trigger on new MLflow model registration → run `onnx_model.ipynb` + `eval_gen.ipynb` → post metrics back to MLflow.  
- **Deploy** a staging environment:  
  - Host embedding & summarization APIs on Chameleon.  
  - Run load tests (Locust) and system optimizations (dynamic batching, multi-instance).  
- **Canary** traffic split:  
  - Expose real user queries to the new model, gather click feedback → update slice- and drift-based dashboards.  
- **Closing the loop**:  
  - Persist mis-predicted examples in MinIO + Label Studio → human annotations → continuous retraining.
\
