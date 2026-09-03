FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY pipeline.py .

# models/, inputs/, outputs/ are mounted at runtime (see docker/README.md), not baked into the image
RUN mkdir -p models inputs outputs

ENTRYPOINT ["python3", "pipeline.py"]
