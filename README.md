<div align="center">

# LexVerse

*A universe where legal agents learn, act, and are evaluated.*

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![Benchmarks](https://img.shields.io/badge/Benchmarks-6-334155)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow)

</div>

LexVerse is a **unified environment** for legal language models and agents. It supports user-defined legal tasks with knowledge retrieval, domain skills, and MCP tools, helping agents read legal materials and produce deliverables. It also brings existing legal benchmarks into a shared workflow for task preparation, execution, and evaluation, while preserving their original task formats, interaction harnesses, and evaluators.

Researchers can use it to compare models, developers can use it to build legal agents, and legal professionals can use it to test AI in practice. The same capabilities can also be enabled for supported benchmark tasks.

## Supported benchmarks

[LexEval](https://github.com/CSHaitao/LexEval) ·
[LawBench](https://github.com/open-compass/LawBench) ·
[J1Bench](https://github.com/FudanDISC/J1Bench) ·
[PLawBench](https://github.com/SKYLENAGE-AI/PLawBench) ·
[DLawBench](https://github.com/SKYLENAGE-AI/DLawBench) ·
[Legal-world](https://github.com/sii-research/Legal-world)

**Planned integrations:** [LegalAgentBench](https://github.com/CSHaitao/LegalAgentBench) · [AgentsCourt](https://github.com/Zhitao-He/AgentsCourt) · [MASER](https://github.com/FudanDISC/MASER) · [LeCaRDv2](https://github.com/THUIR/LeCaRDv2).

Model backends include OpenAI-compatible APIs, Transformers and vLLM.

## Quick start

Use Python 3.12 or newer.

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

### User task

Prepare the agent dependencies:

```bash
lexverse environment setup
```

Download optional knowledge and Skills separately; setup does not download them. Downloads default to the LexVerse project root's `.lexverse/capabilities/`, regardless of the current directory; use `--local-dir` to override it:

```bash
lexverse environment download knowledge
lexverse environment download skills
```

These commands do not build indexes or install software required by Skill scripts; use `environment build` to prepare indexes as needed.

```bash
lexverse environment build --knowledge user_workspace/knowledge --mode hybrid
```

Initialize a new task with `lexverse task init my_task`, or start from the built-in sandbox example at `user_workspace/task/example/task.yaml`. Task files are organized as follows:

```text
user_workspace/
├── knowledge/              # Documents available for retrieval
└── task/<task_name>/
    ├── task.yaml           # Instructions, model and capability selection
    ├── inputs/             # Files provided for this task
    ├── environment/        # Optional Dockerfile and static sandbox files
    ├── solution/           # Optional reference solve.sh for the oracle agent
    └── tests/              # Optional test.sh verifier
```

Configure instructions, Knowledge, Skills, MCP and Tools in `task.yaml`, then run:

```bash
lexverse task run user_workspace/task/<task_name>/task.yaml
```

Use `--agent` to select a general CLI agent and `--env` to select its sandbox:

```bash
lexverse task run user_workspace/task/example/task.yaml --agent codex --model gpt-5 --env docker
lexverse task run user_workspace/task/example/task.yaml --agent oracle --no-interactive
```

Run `lexverse task agents` to list supported agents, including custom `module:Class` adapters.

Results are saved to `runs/user_tasks/<task_name>/<run_id>/`. Open `result.json` for the answer and status, `trajectory.json` for the standardized ATIF trajectory, and `artifacts/` for generated files.

### Benchmark evaluation

Copy `configs/benchmarks/<benchmark>.example.yaml` to `<benchmark>.local.yaml`, then configure the model and task selection.

```bash
lexverse benchmark run lawbench
```

The command uses `lawbench.local.yaml` when available, otherwise `lawbench.example.yaml`. Replace `lawbench` with another supported benchmark name.

Results are saved to `runs/benchmarks/<benchmark>/<run_id>/`. Open `summary.json` for completion status and scores.

#### Capability enhancement

For LawBench, LexEval and PLawBench, enable `generation.capabilities.enabled` and select Knowledge, Skills and MCP services to run tasks through the LexVerse agent while retaining the benchmark’s original evaluators. Use a tool-calling model through an OpenAI-compatible service.

See the capability settings in the [LawBench](configs/benchmarks/lawbench.example.yaml) and [LexEval](configs/benchmarks/lexeval.example.yaml) examples. After editing your local configuration, run:

```bash
lexverse benchmark run lawbench --config configs/benchmarks/lawbench.local.yaml
```

Configure resource paths before running, and keep evaluation answers out of the knowledge sources.

## Commands

| Command | Purpose |
|---|---|
| `lexverse task init <task_name>` | Initialize a user task folder |
| `lexverse task run <task.yaml>` | Run a user task |
| `lexverse task resume <path>` | Resume a paused user task |
| `lexverse task inspect <path> [--trajectory]` | Inspect task status and ATIF trajectory |
| `lexverse task agents` | List supported general agents |
| `lexverse benchmark run <benchmark>` | Generate answers and evaluate |
| `lexverse benchmark generate <benchmark>` | Generate answers without scoring |
| `lexverse benchmark evaluate --run-dir <path>` | Evaluate saved answers |
| `lexverse benchmark collect --run-dir <path>` | Rebuild result summaries |
| `lexverse benchmark catalog <benchmark>` | List benchmark tasks |
| `lexverse environment setup` | Install or reuse agent dependencies |
| `lexverse environment download knowledge` | Download upstream and LexVerse-hosted knowledge |
| `lexverse environment download skills` | Download or reuse registered Skills |
| `lexverse environment build --knowledge <paths>` | Build or reuse knowledge indexes |

Use `--config <path>` to select a specific benchmark configuration, or `--resume <path>` to resume a benchmark run. Index building supports `--mode keyword`, `vector`, and `hybrid`.

Run any command with `--help` for its options.

## License, attributions, and contact

LexVerse is licensed under [Apache License 2.0](LICENSE). Upstream code, datasets, skills and services retain their own licenses and usage terms.

General agent integration, sandbox environment abstractions, installed CLI agent adapters, and ATIF trajectory models in LexVerse include components derived from [Harbor](https://github.com/laude-institute/harbor) (`v0.20.0`, Apache License 2.0). 

For questions, open a GitHub Issue or email xieh@tsinghua.edu.cn.
