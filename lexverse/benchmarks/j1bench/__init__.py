from pathlib import Path

from lexverse.benchmarks.plugin import load_upstream


UPSTREAM = load_upstream(Path(__file__).with_name("catalog.yaml"))
