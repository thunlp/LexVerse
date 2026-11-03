from pathlib import Path
from lexverse.benchmarks.plugin import load_upstream

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
UPSTREAM = load_upstream(CATALOG_PATH)
