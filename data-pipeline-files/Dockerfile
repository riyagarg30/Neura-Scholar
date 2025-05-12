FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libmupdf-dev \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libpq-dev \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    gsutil \
    pymupdf \
    numpy \
    pandas \
    sqlalchemy \
    langdetect \
    tqdm \
    psycopg2-binary \
    jupyterlab \
    ipykernel \
    ipywidgets

# Check gsutil version
RUN gsutil version

CMD ["bash"]
