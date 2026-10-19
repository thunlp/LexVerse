FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    libreoffice-writer \
    libreoffice-calc \
    pandoc \
    poppler-utils \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    python-docx \
    openpyxl \
    python-pptx \
    pypdf \
    reportlab \
    pydantic \
    fastmcp \
    jieba \
    filelock \
    pyyaml

WORKDIR /workspace
