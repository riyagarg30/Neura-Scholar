FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 

# ---------- system deps ------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- python deps ------------------------------------------------------
# Pin wheels that have CUDA 12 builds
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Core libs
RUN pip install --no-cache-dir \
        fastapi uvicorn[standard] \
        tokenizers==0.19.* \
        prometheus-client \
        numpy==1.26.* \
        torch==2.2.*+cu122 -f https://download.pytorch.org/whl/torch_stable.html \
        onnxruntime-gpu==1.18.*                 \
        mlflow==2.*                             \
        rich

# ---------- app code ---------------------------------------------------------
WORKDIR /app
COPY backend.py ./backend.py

# Optional: if you keep requirements.txt next to backend.py
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# ---------- runtime env ------------------------------------------------------
# Expose the HTTP port FastAPI serves on
ENV PORT=8000
EXPOSE 8000

# You can set these via --env-file as shown above instead of baking creds
# ENV MLFLOW_TRACKING_URI=http://your‑mlflow‑host:5000
# ENV MLFLOW_TRACKING_USERNAME=…
# ENV MLFLOW_TRACKING_PASSWORD=…

# Default command
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
