# Use NVIDIA’s CUDA runtime image (with cuDNN) as the base so that
# ONNXRuntime’s CUDAExecutionProvider will work out of the box.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ───────── Environment defaults ─────────────────────────────────────────────
# MLflow settings (override at runtime with -e)
ENV MLFLOW_TRACKING_URI="http://localhost:5000" \
    MLFLOW_MODEL_STAGE="Production" \
    EVAL_DIR="/eval"

# ───────── Install Python & pip ─────────────────────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
 && ln -s /usr/bin/python3 /usr/bin/python \
 && ln -s /usr/bin/pip3   /usr/bin/pip \
 && rm -rf /var/lib/apt/lists/*

# ───────── Create application directory ─────────────────────────────────────
WORKDIR /app

# ───────── Install Python dependencies ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ───────── Copy application & data ──────────────────────────────────────────
COPY main.py .
# Copy all JSONL files (heldout, slices, perturbations, failures) into /eval
COPY eval/ /eval/

# ───────── Default command ──────────────────────────────────────────────────
CMD ["python", "main.py"]
