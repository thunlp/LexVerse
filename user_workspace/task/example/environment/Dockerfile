FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    fonts-noto-cjk \
    pandoc \
    poppler-utils \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.14.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    python-docx \
    openpyxl \
    python-pptx \
    lxml \
    pillow \
    pypdf \
    reportlab \
    pydantic \
    fastmcp==4.0.11 \
    langchain-core==1.6.7 \
    langchain-text-splitters==1.1.3 \
    sentence-transformers==3.4.1 \
    transformers==4.57.6 \
    huggingface-hub==0.36.2 \
    faiss-cpu==1.15.1 \
    numpy==1.26.4 \
    jieba==0.42.1 \
    filelock \
    pyyaml

ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
