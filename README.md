<div align="center">

# LexVerse

*A universe where legal agents learn, act, and are evaluated.*

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Benchmarks](https://img.shields.io/badge/Benchmarks-6-334155)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow)

</div>

LexVerse is a **unified environment** for legal language models and agents, spanning from benchmark evaluation to real-world legal tasks. On the evaluation side, it brings existing legal benchmarks into a shared workflow for task preparation, execution, and result management, while preserving each benchmark's original task formats, interaction harnesses, and evaluators. On the practice side, it connects agents to legal knowledge retrieval, domain skills, and MCP tools, so they can read legal materials, ground their reasoning in authoritative sources, and produce real deliverables.

Researchers can use it to compare models, developers can use it to build legal agents, and legal professionals can use it to test whether AI holds up in practice. **Whoever you are, LexVerse is a place to start.** We are continuing to integrate more benchmarks and expand environment capabilities to support richer agent interactions.

## Supported benchmarks

[LexEval](https://github.com/CSHaitao/LexEval) ·
[LawBench](https://github.com/open-compass/LawBench) ·
[J1Bench](https://github.com/FudanDISC/J1Bench) ·
[PLawBench](https://github.com/SKYLENAGE-AI/PLawBench) ·
[DLawBench](https://github.com/SKYLENAGE-AI/DLawBench) ·
[Legal-world](https://github.com/sii-research/Legal-world)

Model backends include OpenAI-compatible APIs, Transformers and vLLM.

## Quick start

Use Python 3.11 or newer.

```bash
git clone https://github.com/thunlp/LexVerse.git
cd LexVerse
pip install -e ".[lawbench,lexeval,j1bench,plawbench,dlawbench,legalworld,openai]"
```

Configure API connections in `configs/secrets.local.yaml`:

```yaml
openai:
  base_url: https://api.openai.com/v1
  api_key: YOUR_API_KEY
```

Set `profile: openai` in your model configuration to use this connection. Keep credentials local.

### Benchmark evaluation

Copy `configs/benchmarks/<benchmark>.example.yaml` to `<benchmark>.local.yaml`, then configure the model and task selection.

```bash
lexverse benchmark run lawbench
```

The command uses `lawbench.local.yaml` when available, otherwise `lawbench.example.yaml`. Replace `lawbench` with another supported benchmark name.

Results are saved to `runs/benchmarks/<benchmark>/<run_id>/`. Open `summary.json` for completion status and scores.

### User task

Prepare the agent dependencies:

```bash
lexverse environment setup
```

Organize your task and files:

```text
user_workspace/
├── knowledge/              # Documents available for retrieval
└── task/<task_name>/
    ├── task.yaml           # Instructions, model and capability selection
    └── inputs/             # Files provided for this task
```

Configure `task.yaml` using the [task configuration guide](docs/Stage2设计文档.md), then run:

```bash
lexverse task run user_workspace/task/<task_name>/task.yaml
```

For large knowledge sources, optionally build indexes beforehand:

```bash
lexverse environment build --knowledge user_workspace/knowledge --mode hybrid
```

Results are saved to `runs/user_tasks/<task_name>/<run_id>/`. Open `result.json` for the answer and status, and `artifacts/` for generated files.

## Commands

| Command                                            | Purpose                             |
| -------------------------------------------------- | ----------------------------------- |
| `lexverse benchmark run <benchmark>`             | Generate answers and evaluate       |
| `lexverse benchmark generate <benchmark>`        | Generate answers without scoring    |
| `lexverse benchmark evaluate --run-dir <path>`   | Evaluate saved answers              |
| `lexverse benchmark collect --run-dir <path>`    | Rebuild result summaries            |
| `lexverse benchmark catalog <benchmark>`         | List benchmark tasks                |
| `lexverse task run <task.yaml>`                  | Run a user task                     |
| `lexverse task resume <path>`                    | Resume a paused user task           |
| `lexverse task inspect <path>`                   | Inspect task status                 |
| `lexverse environment setup`                     | Install or reuse agent dependencies |
| `lexverse environment build --knowledge <paths>` | Build or reuse knowledge indexes    |

Use `--config <path>` to select a specific benchmark configuration, or `--resume <path>` to resume a benchmark run. Index building supports `--mode keyword`, `vector`, and `hybrid`.

Run any command with `--help` for its options.

## License and contact

LexVerse is licensed under [Apache License 2.0](LICENSE). Upstream code, datasets, skills and services retain their own licenses and usage terms.

For questions, open a GitHub Issue or email xieh@tsinghua.edu.cn.
