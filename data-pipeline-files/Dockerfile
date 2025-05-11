# Use the official Python 3.11 slim image as a base
FROM python:3.11-slim

# Ensure system packages are up to date, install curl (for pulling GS SDK key), then install gsutil via pip
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && pip install --no-cache-dir gsutil \
    && rm -rf /var/lib/apt/lists/*

# Verify installation
RUN gsutil version

# Default to bash so you can drop into the container interactively
CMD ["bash"]
