<h1 align="center">LexVerse</h1>

<p align="center"><em>A universe where legal agents learn, act, and are evaluated.</em></p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Benchmarks" src="https://img.shields.io/badge/Benchmarks-LexEval%20%7C%20LawBench%20%7C%20J1Bench-243B53">
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-C9A227">
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-supported-benchmarks">Benchmarks</a> ·
  <a href="#-outputs">Outputs</a> ·
  <a href="docs/architecture.html">Architecture</a>
</p>

LexVerse is an evolving runtime for evaluating legal language models and agents.
It brings legal benchmarks into a shared workflow for task preparation,
execution, and result management while preserving their original task formats,
interaction harnesses, and evaluators. We are continuing to integrate more
benchmarks and develop environment capabilities for richer agent interactions.

## ⚖️ Supported Benchmarks

| Benchmark                                           |               Coverage | Interaction                 | Verification                |
| --------------------------------------------------- | ---------------------: | --------------------------- | --------------------------- |
| [LexEval](https://github.com/CSHaitao/LexEval)       |               23 tasks | Single response             | Upstream task evaluator     |
| [LawBench](https://github.com/open-compass/LawBench) |               20 tasks | Single response             | Upstream task evaluator     |
| [J1Bench](https://github.com/FudanDISC/J1Bench)      | CI, CR, KQ, LC, CD, DD | Official multi-role harness | Upstream scenario evaluator |

## 🚀 Quick Start

### 1. Install LexVerse

Use Python 3.10 or newer. Install LexVerse with the dependencies for all three benchmarks:

```bash
python -m pip install -e ".[lexeval,lawbench,j1bench,openai]"
python -m lexverse --help
```

> **Compatibility note**
> One environment can run all these benchmarks. LexEval and LawBench use
> different Rouge packages, so some Rouge-based scores may differ slightly
> from upstream results in a combined installation.

### 2. Create local configuration

Create local configuration files from the tracked `*.example.yaml` templates.

Configure at least one OpenAI-compatible connection in
`configs/secrets.local.yaml`.

The benchmark YAML selects the connection with `model.profile` and the actual
model with `model.name`. J1Bench also has `evaluation.model`, because its
official evaluator calls a scoring model. Model names belong in benchmark
configuration, not in the secrets file.

### 3. Run LexEval

`prepare` resolves the selected upstream cases and freezes them into a
manifest. It does not call the model. `run` performs generation, official
task-level evaluation, and aggregation.

```bash
python -m lexverse prepare \
  --config configs/benchmarks/lexeval.local.yaml \
  --output .lexverse/prepared/lexeval-all.json

LEXEVAL_RUN="runs/lexeval-all-$(date +%Y%m%d-%H%M%S)"
python -m lexverse run \
  --prepared .lexverse/prepared/lexeval-all.json \
  --output-root "$LEXEVAL_RUN"
```

### 4. Run LawBench

```bash
python -m lexverse prepare \
  --config configs/benchmarks/lawbench.local.yaml \
  --output .lexverse/prepared/lawbench-all.json

LAWBENCH_RUN="runs/lawbench-all-$(date +%Y%m%d-%H%M%S)"
python -m lexverse run \
  --prepared .lexverse/prepared/lawbench-all.json \
  --output-root "$LAWBENCH_RUN"
```

If a run is interrupted after model generation, set `execution.resume: true`
in `configs/benchmarks/lawbench.local.yaml`, run `prepare` again, and reuse the
same `--output-root`. Completed trials will be reused instead of calling the
model again.

### 5. Run J1Bench

J1Bench uses gated data. Accept the terms on the
[J1-Eval dataset page](https://huggingface.co/datasets/CimoInkPool/J1-Eval_Dataset/tree/main)
and authenticate once:

```bash
hf auth login
```

The first `prepare` downloads the selected `J1-Eval_<SCENARIO>.jsonl` files to
`.lexverse/datasets/j1bench/`. Later runs reuse this cache and do not modify the
downloaded files.

```bash
python -m lexverse prepare \
  --config configs/benchmarks/j1bench.local.yaml \
  --output .lexverse/prepared/j1bench-all.json

J1BENCH_RUN="runs/j1bench-all-$(date +%Y%m%d-%H%M%S)"
python -m lexverse run \
  --prepared .lexverse/prepared/j1bench-all.json \
  --output-root "$J1BENCH_RUN"
```

J1Bench runs the official multi-role conversation for every case and invokes
the official evaluator once per scenario. It is slower and more expensive than
the two single-response benchmarks. A successful `run` has already completed
evaluation; there is no separate eval command.

## 📂 Outputs

Each `--output-root` contains `manifest.json`, `summary.json`, trial records,
and official evaluation artifacts. LexEval and LawBench report task-level
scores in `evaluation_result.csv`; J1Bench keeps per-case intermediate results
and a scenario-level final result under `trials/<scenario>/verifier/`.

`collect` rebuilds `summary.json` from existing Trial records and the existing
official `evaluation_result.csv`:

```bash
python -m lexverse collect --run-dir "$LEXEVAL_RUN"
```

For a cheaper smoke test, change only `execution.limit` in a local benchmark
YAML. Run `prepare` again after every YAML change; `run` rejects configuration
drift by design.

## 🏗️ Architecture

LexVerse connects benchmark preparation, execution, evaluation, and result
management in one workflow.

[Open the interactive architecture map →](docs/architecture.html)

## 📝 Notes

- Do not edit files under `.lexverse/datasets/j1bench/`.
- Re-run `prepare` whenever a local YAML changes.
- Upstream repositories are pinned in `lexverse/runtime/upstream.py`; patches
  make them relocatable without replacing evaluator logic.
- Benchmark and dataset licenses remain governed by their original sources.

LexVerse is released under the [Apache License 2.0](LICENSE).

## Contact Us

For issues and feature requests, use GitHub Issues. You can also email xieh@tsinghua.edu.cn.
