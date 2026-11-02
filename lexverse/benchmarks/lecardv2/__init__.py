"""LeCaRDv2 benchmark adapter for full-corpus and candidate-pool legal case retrieval."""
from pathlib import Path
from lexverse.benchmarks.plugin import load_upstream

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
UPSTREAM = load_upstream(CATALOG_PATH)
