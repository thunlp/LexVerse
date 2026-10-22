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

| Benchmark | Documentation | Coverage | Interaction | Verification |
| :--- | :--- | ---: | :--- | :--- |
| [LexEval](https://github.com/CSHaitao/LexEval) | [`lexverse/benchmarks/lexeval/README.md`](lexverse/benchmarks/lexeval/README.md) | 23 tasks | Single response | Upstream task evaluator |
| [LawBench](https://github.com/open-compass/LawBench) | [`lexverse/benchmarks/lawbench/README.md`](lexverse/benchmarks/lawbench/README.md) | 20 tasks | Single response | Upstream task evaluator |
| [J1Bench](https://github.com/CimoInkPool/J1-EVAL) | [`lexverse/benchmarks/j1bench/README.md`](lexverse/benchmarks/j1bench/README.md) | CI, CR, KQ, LC, CD, DD | Official multi-role harness | Upstream scenario evaluator |
| [PLawBench](https://github.com/SKYLENAGE-AI/PLawBench) | [`lexverse/benchmarks/plawbench/README.md`](lexverse/benchmarks/plawbench/README.md) | 3 tasks | Single response | Upstream rubric evaluator |
| [DLawBench](https://github.com/SKYLENAGE-AI/DLawBench) | [`lexverse/benchmarks/dlawbench/README.md`](lexverse/benchmarks/dlawbench/README.md) | 4 personas | Official consultation loop | Upstream session evaluator |
| [Legal-World](https://github.com/riotulab/Legal-World) | [`lexverse/benchmarks/legalworld/README.md`](lexverse/benchmarks/legalworld/README.md) | 6 stages | Official stage pipeline | Upstream stage evaluator |
| [SimuCourt](https://github.com/Zhitao-He/AgentsCourt) | [`lexverse/benchmarks/simucourt/README.md`](lexverse/benchmarks/simucourt/README.md) | 6 tasks | Single response | Upstream task evaluator |
| [MASER](https://github.com/FudanDISC/MASER) | [`lexverse/benchmarks/maser/README.md`](lexverse/benchmarks/maser/README.md) | `complaint_generation` | Official Law_Fact runner | Upstream scenario evaluator |
| [LeCaRDv2](https://github.com/THUIR/LeCaRDv2) | [`lexverse/benchmarks/lecardv2/README.md`](lexverse/benchmarks/lecardv2/README.md) | `case_retrieval` | Encoder / LLM rewrite retrieval | Recall@K & ranking evaluator |

## 🚀 Quick Start

### 1. Install LexVerse

Use Python 3.10 or newer. Install LexVerse with the dependencies for the benchmarks you want to run:

```bash
python -m pip install -e ".[lexeval,lawbench,j1bench,simucourt,maser,lecardv2,openai]"
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

The benchmark YAML selects the connection with `models.default.profile` and the
actual model with `models.default.name`. Multi-role and LLM-judged benchmarks
also configure `models.roles` and `models.evaluators`. Model names belong in
benchmark configuration, not in the secrets file.

### 3. Run LexEval

`run --dry-run` resolves the selected upstream cases and checks the
configuration without calling the model. `generate` runs model generation only,
`run` performs generation, official evaluation, and aggregation, and `evaluate`
scores an existing run independently.

```bash
python -m lexverse run lexeval --config configs/benchmarks/lexeval.example.yaml --dry-run
python -m lexverse run lexeval --config configs/benchmarks/lexeval.local.yaml \
  --output-root runs/lexeval-experiment
```

### 4. Run LawBench

```bash
python -m lexverse run lawbench --config configs/benchmarks/lawbench.example.yaml --dry-run
python -m lexverse run lawbench --config configs/benchmarks/lawbench.local.yaml \
  --output-root runs/lawbench-experiment
```

If a run is interrupted, resume it from the frozen manifest snapshot:

```bash
python -m lexverse run --resume runs/lawbench-experiment
```

If generation has completed and only scoring needs to be rerun:

```bash
python -m lexverse evaluate --run-dir runs/lawbench-experiment
```

### 5. Run J1Bench

J1Bench uses gated data. Accept the terms on the
[J1-Eval dataset page](https://huggingface.co/datasets/CimoInkPool/J1-Eval_Dataset/tree/main)
and authenticate once:

```bash
hf auth login
```

The first `run --dry-run` or `run` downloads the selected `J1-Eval_<SCENARIO>.jsonl`
files to `.lexverse/datasets/j1bench/`. Later runs reuse this cache and do not
modify the downloaded files.

```bash
python -m lexverse run j1bench --config configs/benchmarks/j1bench.example.yaml --dry-run
python -m lexverse run j1bench --config configs/benchmarks/j1bench.local.yaml \
  --output-root runs/j1bench-experiment
```

J1Bench runs the official multi-role conversation for every case and invokes
the official evaluator once per scenario. It is slower and more expensive than
the two single-response benchmarks.

### 6. Run SimuCourt, MASER, and LeCaRDv2

```bash
python -m lexverse run simucourt --config configs/benchmarks/simucourt.example.yaml --dry-run
python -m lexverse run simucourt --config configs/benchmarks/simucourt.local.yaml \
  --output-root runs/simucourt-experiment

python -m lexverse run maser --config configs/benchmarks/maser.example.yaml --dry-run
python -m lexverse run maser --config configs/benchmarks/maser.local.yaml \
  --output-root runs/maser-experiment

python -m lexverse run lecardv2 --config configs/benchmarks/lecardv2.example.yaml --dry-run
python -m lexverse run lecardv2 --config configs/benchmarks/lecardv2.local.yaml \
  --output-root runs/lecardv2-experiment
```

Adapter details, configuration modes, and local reproduction results are documented in:
- [`lexverse/benchmarks/maser/README.md`](lexverse/benchmarks/maser/README.md)
- [`lexverse/benchmarks/simucourt/README.md`](lexverse/benchmarks/simucourt/README.md)
- [`lexverse/benchmarks/lecardv2/README.md`](lexverse/benchmarks/lecardv2/README.md)

## 📂 Outputs

Each `--output-root` contains `manifest.json`, `summary.json`, trial records,
and official evaluation artifacts. LexEval and LawBench report task-level
scores in `evaluation_result.csv`; J1Bench and MASER keep per-case intermediate
results and a scenario-level final result under `trials/<scenario>/verifier/`.

`collect` rebuilds `summary.json` from existing Trial records and the existing
official `evaluation_result.csv`:

```bash
python -m lexverse collect --run-dir runs/lexeval-experiment
```

For a cheaper smoke test, adjust `benchmark.default_limit` or each task's
`limit` in a local benchmark YAML (`null` selects all samples). Start a new run
whenever changing models, tasks, or generation parameters.

## 🏗️ Architecture

LexVerse connects benchmark preparation, execution, evaluation, and result
management in one workflow.

[Open the interactive architecture map →](docs/architecture.html)

## 📝 Notes

- Do not edit files under `.lexverse/datasets/j1bench/`.
- Start a new run after modifying a local YAML; `--resume` reads the frozen manifest snapshot.
- Upstream repositories are pinned in each benchmark's `catalog.yaml`; patches make them relocatable without replacing evaluator logic.
- Benchmark and dataset licenses remain governed by their original sources.

LexVerse is released under the [Apache License 2.0](LICENSE).

## Contact Us

For issues and feature requests, use GitHub Issues. You can also email xieh@tsinghua.edu.cn.
