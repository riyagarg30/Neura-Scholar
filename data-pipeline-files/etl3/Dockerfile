FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libmupdf-dev \
    build-essential \
    libglib2.0-0 \
    libgl1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    gsutil \
    pymupdf \
    numpy \
    langdetect \
    tqdm

RUN gsutil version

# Default to bash so you can drop into the container interactively
CMD ["bash"]
