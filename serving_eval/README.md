## Project: Semantic Search Evaluation & Serving

This repository contains the end-to-end tooling, containers, and notebooks for building, serving, and rigorously evaluating a chunk-based semantic search system over the ArXiv corpus. We embed chunks of paper text, index them in FAISS, collapse chunk hits into paper-level scores, and continuously validate accuracy, robustness, and performance before promoting any model.

---

## Serving & Evaluation Structure

.
├── Dockerfile # Root image: Python + CUDA + MLflow + evaluation
├── backend.py # Entrypoint for the FastAPI embedding service
├── app_backend/ # Service code
│ ├── app_backend.py # • Defines /embed/text and /embed/batch endpoints
│ └── requirements.txt # • Python deps for the service
├── requirements.txt # Root Python deps (mlflow, onnxruntime, faiss, etc.)
│
├── eval/ # All evaluation data, scripts & notebooks
│ ├── test_dataset_generator.ipynb # (was eval_data.ipynb) build JSONL/NPZ eval sets
│ ├── chunk_query_generator.ipynb # generate per-chunk query testcases & slices
│ ├── generate-embedding-new.ipynb # (was generate-embeddings-new.ipynb) batch embedding → FAISS
│ ├── accuracy_benchmarks.ipynb # new: notebook to compute/visualize Recall/MRR, etc.
│ ├── accuracy_benchmarks.py # new: same metrics in script form for CICD
│ ├── performance_benchmark.ipynb # new: notebook to measure latency & throughput
│ ├── bm25_filter.py # BM25 prototype (not used in current pipeline)
│ ├── failures.jsonl # “hard” queries for failure-mode tests
│ ├── heldout.jsonl # held-out retrieval test set
│ ├── perturbations.jsonl # noisy/perturbed queries for robustness tests
│ ├── slices.jsonl # grouped queries by category/year (“slices”)
│ ├── drift_reference.npz # reference embeddings for drift monitoring
│ └── model_comparison.jsonl # output from stage_eval/main.py
│
├── eval/stage_eval/ # Containerized staging‐eval harness
│ ├── Dockerfile # image to run main.py with CUDA + MLflow
│ ├── main.py # pulls model from MLflow, runs evaluation, writes JSONL
│ └── requirements.txt # Python deps for the stage‐eval image
│
├── indexes/ # FAISS indexes & metadata for offline eval
│ ├── chunk_embedding_768.index
│ ├── ...
│
└── onnx_model.ipynb # export & quantize PyTorch bi-encoder → ONNX


---

## Artifact Summaries

### 1. `test_dataset_generator.ipynb`  
*(formerly `eval_data.ipynb`)*  
Builds all of your evaluation datasets from Postgres and writes:  
- **Held-out** retrieval cases → `heldout.jsonl`  
- **Slice tests** (by category/year) → `slices.jsonl`  
- **Perturbations** (noisy rephrasings) → `perturbations.jsonl`  
- **Failure modes** (“hard” queries) → `failures.jsonl`  
- **Drift reference** embeddings → `drift_reference.npz`  

### 2. `chunk_query_generator.ipynb`  
Generates per-chunk queries for both held-out and slice evaluation, and tags each query with its slice key.  

### 3. `generate-embedding-new.ipynb`  
*(renamed from `generate-embeddings-new.ipynb`)*  
Batches your `/embed/batch` FastAPI endpoint to compute all chunk embeddings, then writes:  
- A FAISS index file (`.index`) under `indexes/`  
- JSONL metadata mapping positions → `chunk_id`  

### 4. `onnx_model.ipynb`  
Exports your fine-tuned bi-encoder to ONNX, applies graph optimizations, and runs INT8 quantization. Benchmarks each variant for size, single-sample latency, and batch throughput on CPU.

### 5. `accuracy_benchmarks.ipynb` & `accuracy_benchmarks.py`  
**New!** Compute all retrieval metrics (Recall@K raw/adj, MRR@K, chunk recall), per-slice recalls, plus perturbation & failure-mode pass rates. Notebook for exploration; script for automated CICD gating.

### 6. `performance_benchmark.ipynb`  
**New!** Measures end-to-end query latency and throughput of your embedding service and/or in-process ONNX path, on both CPU and GPU.

### 7. `stage_eval/`  
Contains a minimal container that, on startup:  
1. Pulls the correct ONNX model from MLflow (via `models:/…/<STAGE>`),  
2. Builds/loads FAISS indexes,  
3. Runs the full held-out, slice, perturbation, failure, and chunk-recall evaluations,  
4. Emits `model_comparison.jsonl` under `/eval` for artifact collection.

### 8. `backend.py` & `app_backend/`  
Your production FastAPI service code to serve embeddings:  
- `/embed/text` (single query)  
- `/embed/batch` (batch queries)  
Built on ONNXRuntime with CUDA for real-time throughput.

---

## How to Use

1. **Build/export** your bi-encoder → run `onnx_model.ipynb`.  
2. **Start** the FastAPI embedding service:  
```bash
docker build -t embedding-service .
docker run --gpus all -p 8000:8000 embedding-service
```
3. Generate embeddings for all chunks via generate-embedding-new.ipynb.
4. Run chunk_query_generator.ipynb (or script) to prepare test queries.
5. Local eval: interactively run accuracy_benchmarks.ipynb & performance_benchmark.ipynb.
6. Staging eval:
```bash
cd eval/stage_eval
docker build -t neura-scholar-eval:cuda .
docker run --gpus all \
  -e MLFLOW_TRACKING_URI="http://mlflow:5000" \
  -e MLFLOW_MODEL_STAGE="Staging" \
  neura-scholar-eval:cuda
```
→ writes output /eval/model_comparison.jsonl
7. CI/CD: hook accuracy_benchmarks.py into your pipeline to fail on regression, and let Argo/Airflow run the stage_eval container on every new model promotion.

Key Takeaways

-Data prep, model conversion, embedding gen, and evaluation are fully modular.
-Robustness tests (slices, perturbations, failures) guard against regressions and bias.
-Performance benchmarks ensure real-time viability on GPU/CPU.
-Containerized gating with MLflow integration gives automated “staging” checks before production rollout.

