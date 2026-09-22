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

LexVerse is a unified runtime for reproducible evaluation of legal language
models and agents. It provides one execution and result-management pipeline
while preserving each benchmark's upstream task format, interaction harness,
and evaluator.

```text
Config → Prepare → Trials → Official Verifier → Summary
```

The current runtime does not include a legal corpus, retrieval system, MCP data
service, or data-environment SDK.

## ⚖️ Supported Benchmarks

| Benchmark                                           |               Coverage | Interaction                 | Verification                |
| --------------------------------------------------- | ---------------------: | --------------------------- | --------------------------- |
| [LexEval](https://github.com/CSHaitao/LexEval)       |               23 tasks | Single response             | Upstream task evaluator     |
| [LawBench](https://github.com/open-compass/LawBench) |               20 tasks | Single response             | Upstream task evaluator     |
| [J1Bench](https://github.com/FudanDISC/J1Bench)      | CI, CR, KQ, LC, CD, DD | Official multi-role harness | Upstream scenario evaluator |

LexEval and LawBench are evaluated once per task. J1Bench is evaluated once per
scenario and additionally retains its official per-case intermediate results.

## 🚀 Quick Start

### 1. Create the environment

Python 3.10 or newer is required. Conda avoids accidentally using macOS's system Python 3.9.

```bash
conda create -n lexverse python=3.10 -y
conda activate lexverse

which python
python --version
```

`which python` should point inside `.../envs/lexverse/`, and only `(lexverse)`
should appear in the shell prompt.

Install LexVerse and the dependencies declared by all three integrations:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[lexeval,lawbench,j1bench,openai]"
python -m lexverse --help
```

> **Compatibility note**
> LexEval's `rouge` and LawBench's `rouge_chinese` publish the same Python
> import path. A combined environment is suitable for integration testing but
> cannot reproduce both upstream Rouge tokenizers exactly. Use separate
> environments only when strict Rouge parity is required.

### 2. Create local configuration

Copy the tracked templates. The resulting local files are ignored by Git.

```bash
cp configs/secrets.example.yaml configs/secrets.local.yaml
cp configs/benchmarks/lexeval.example.yaml configs/benchmarks/lexeval.local.yaml
cp configs/benchmarks/lawbench.example.yaml configs/benchmarks/lawbench.local.yaml
cp configs/benchmarks/j1bench.example.yaml configs/benchmarks/j1bench.local.yaml
mkdir -p .lexverse/prepared runs
```

Configure at least one OpenAI-compatible connection in
`configs/secrets.local.yaml`:

```yaml
openai:
  provider: openai_compatible
  base_url: https://api.openai.com/v1
  api_key: YOUR_API_KEY
```

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

echo "$LEXEVAL_RUN"
python -m json.tool "$LEXEVAL_RUN/summary.json"
find "$LEXEVAL_RUN" -maxdepth 5 -type f | sort
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

echo "$LAWBENCH_RUN"
python -m json.tool "$LAWBENCH_RUN/summary.json"
find "$LAWBENCH_RUN" -maxdepth 5 -type f | sort
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

echo "$J1BENCH_RUN"
python -m json.tool "$J1BENCH_RUN/summary.json"
find "$J1BENCH_RUN" -maxdepth 8 -type f | sort
```

J1Bench runs the official multi-role conversation for every case and invokes
the official evaluator once per scenario. It is slower and more expensive than
the two single-response benchmarks. A successful `run` has already completed
evaluation; there is no separate eval command.

### 6. Confirm success

Every `summary.json` should contain:

```json
{
  "status": "completed",
  "samples": {
    "failed": 0,
    "missing": []
  }
}
```

## 📂 Outputs

LexEval and LawBench use the common layout below:

```text
runs/<benchmark>-<timestamp>/
├── manifest.json                    # frozen run metadata
├── summary.json                     # LexVerse run summary
├── evaluation_result.csv            # official-format task results
├── verifier/
│   └── predictions/
│       └── <model>/                 # official evaluator input
└── trials/
    └── <task>/
        └── <case>/
            └── results.json             # one Trial execution record
```

LexEval predictions are named `<model>_<task>.jsonl`; LawBench predictions are
named `<task>.json`.

J1Bench places the official verifier under each scenario:

```text
trials/<scenario>/verifier/
├── dialog_history/                   # official evaluator input
├── intermediate/                     # official per-case output
├── final/                            # official scenario output
├── stdout.log
└── stderr.log
```

LexEval and LawBench expose task-level scores rather than official per-case
scores. J1Bench exposes both per-case intermediate results and a scenario-level
final result.

`collect` rebuilds `summary.json` from existing Trial records and the existing
official `evaluation_result.csv`:

```bash
python -m lexverse collect --run-dir "$LEXEVAL_RUN"
```

For a cheaper smoke test, change only `execution.limit` in a local benchmark
YAML. Run `prepare` again after every YAML change; `run` rejects configuration
drift by design.

## 🏗️ Architecture

```text
Config → TaskBundle → Orchestrator → Environment → Official Verifier → Artifacts
```

[Open the interactive architecture map →](docs/architecture.html)

## 📝 Notes

- Do not edit files under `.lexverse/datasets/j1bench/`.
- Re-run `prepare` whenever a local YAML changes.
- Upstream repositories are pinned in `lexverse/runtime/upstream.py`; patches
  make them relocatable without replacing evaluator logic.
- The wheel and source distribution exclude benchmark data, cloned upstream
  repositories, local runs, tests, documentation, and credential files.
- Benchmark and dataset licenses remain governed by their original sources.

LexVerse is released under the [Apache License 2.0](LICENSE). The runtime design
references Harbor's high-level execution patterns without vendoring Harbor
code; attribution details are recorded in [NOTICE](NOTICE).
