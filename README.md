<div align="center">

# LexVerse

*A universe where legal agents learn, act, and are evaluated.*

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![Benchmarks](https://img.shields.io/badge/Benchmarks-6-334155)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow)

</div>

LexVerse is **a unified environment** for legal language models and agents, spanning from benchmark evaluation to real-world legal tasks. On the evaluation side, it brings existing legal benchmarks into a shared workflow for task preparation, execution, and result management, while preserving each benchmark's original task formats, interaction harnesses, and evaluators. On the practice side, it connects agents to legal knowledge retrieval, domain skills, and MCP tools, so they can read legal materials, ground their reasoning in authoritative sources, and produce real deliverables.

Researchers can use it to compare models, developers can use it to build legal agents, and legal professionals can use it to test whether AI holds up in practice. **Whoever you are, LexVerse is a place to start.** We are continuing to integrate more benchmarks and expand environment capabilities to support richer agent interactions.

## Supported benchmarks

[LexEval](https://github.com/CSHaitao/LexEval) ·
[LawBench](https://github.com/open-compass/LawBench) ·
[J1Bench](https://github.com/FudanDISC/J1Bench) ·
[PLawBench](https://github.com/SKYLENAGE-AI/PLawBench) ·
[DLawBench](https://github.com/SKYLENAGE-AI/DLawBench) ·
[Legal-world](https://github.com/sii-research/Legal-world) ·
[AgentsCourt](https://github.com/Zhitao-He/AgentsCourt) ·
[MASER](https://github.com/FudanDISC/MASER) ·
[LeCaRDv2](https://github.com/THUIR/LeCaRDv2)

## Quick start

Use Python 3.12 or newer.

```bash
git clone https://github.com/thunlp/LexVerse.git
cd LexVerse
```

Install Conda, then create and activate a Python 3.12 environment:

```bash
conda create -n lexverse python=3.12
conda activate lexverse
```

Then install LexVerse in the active environment:

```bash
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

#### Step 1. Prepare dependencies

```bash
lexverse environment setup
```

#### Step 2. Prepare Skills and knowledge (optional)

Skills and knowledge are optional; `setup` does not download them. Downloads are saved under the project root's `.lexverse/capabilities/` by default (`--local-dir` overrides this).

**Download Skills:**

```bash
lexverse environment download skills
```

**Prepare knowledge using either option:**

- Download prebuilt indexes, ready for retrieval:

  ```bash
  lexverse environment download indexes
  # Or download selected collections only:
  lexverse environment download indexes --collections legal_laws legal_cases/JuDGE
  ```
- Download source documents and build indexes locally:

  ```bash
  lexverse environment download knowledge
  # Example: build the legal_laws collection.
  lexverse environment build --knowledge .lexverse/capabilities/knowledge/legal_laws --mode hybrid
  ```

Prebuilt indexes use **hybrid retrieval** (BM25 keyword search + FAISS vector search), with **`BAAI/bge-small-zh-v1.5`** as the embedding model. Queries also require this model. Use the same hybrid configuration in your task; select downloaded collections with IDs such as `collection:legal_laws`. For custom index settings, see [retrieval.example.yaml](configs/knowledge/retrieval.example.yaml).

#### Step 3. Initialize a task

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

#### Step 4. Run the task

Configure instructions, Knowledge, Skills, MCP and Tools in `task.yaml`, then run:

```bash
lexverse task run user_workspace/task/<task_name>/task.yaml
```

Use `--agent` to select a general CLI agent and `--env` to select its execution environment (`local` or `docker`):

```bash
lexverse task run user_workspace/task/example/task.yaml --agent codex --model gpt-5 --env docker
lexverse task run user_workspace/task/example/task.yaml --agent oracle --no-interactive
```

Run `lexverse task agents` to list supported agents, including custom `module:Class` adapters.

Docker runs commands in a non-root container with a read-only root filesystem: `/inputs` and `/skills` are read-only; `/workspace` and `/artifacts` are writable. Local runs with the host user's permissions, without filesystem or network isolation.

Docker allows unrestricted outbound access by default (`network_mode: public`). To enable an allowlist:

```yaml
environment: docker
network_mode: allowlist
allowed_hosts: []  # Optional extra exact hostnames or IPs.
```

#### Step 5. View results

Results are saved to `runs/user_tasks/<task_name>/<run_id>/`. Open `result.json` for the answer and status, `trajectory.json` for the standardized ATIF trajectory, and `artifacts/` for generated files.

### Benchmark evaluation

Copy `configs/benchmarks/<benchmark>.example.yaml` to `<benchmark>.local.yaml`, then configure the model and task selection.

```bash
lexverse benchmark run lawbench
```

The command uses `lawbench.local.yaml` when available, otherwise `lawbench.example.yaml`. Replace `lawbench` with another supported benchmark name.

Results are saved to `runs/benchmarks/<benchmark>/<run_id>/`. Open `summary.json` for completion status and scores.

#### Capability enhancement

For LawBench and LexEval, enable `generation.capabilities.enabled` and select Knowledge, Skills and MCP services to run tasks through the LexVerse agent while retaining the benchmark’s original evaluators. Use a tool-calling model through an OpenAI-compatible service.

See the capability settings in the [LawBench](configs/benchmarks/lawbench.example.yaml) and [LexEval](configs/benchmarks/lexeval.example.yaml) examples. After editing your local configuration, run:

```bash
lexverse benchmark run lawbench --config configs/benchmarks/lawbench.local.yaml
```

Configure resource paths before running, and keep evaluation answers out of the knowledge sources.

## Commands

| Command                                            | Purpose                                            |
| -------------------------------------------------- | -------------------------------------------------- |
| `lexverse task init <task_name>`                 | Initialize a user task folder                      |
| `lexverse task run <task.yaml>`                  | Run a user task                                    |
| `lexverse task resume <path>`                    | Resume a paused user task                          |
| `lexverse task inspect <path> [--trajectory]`    | Inspect task status and ATIF trajectory            |
| `lexverse task agents`                           | List supported general agents                      |
| `lexverse benchmark run <benchmark>`             | Generate answers and evaluate                      |
| `lexverse benchmark generate <benchmark>`        | Generate answers without scoring                   |
| `lexverse benchmark evaluate --run-dir <path>`   | Evaluate saved answers                             |
| `lexverse benchmark collect --run-dir <path>`    | Rebuild result summaries                           |
| `lexverse benchmark catalog <benchmark>`         | List benchmark tasks                               |
| `lexverse environment setup`                     | Install or reuse agent dependencies                |
| `lexverse environment download knowledge`        | Download upstream and LexVerse-hosted knowledge    |
| `lexverse environment download indexes`          | Download and verify prebuilt knowledge collections |
| `lexverse environment download skills`           | Download or reuse registered Skills                |
| `lexverse environment build --knowledge <paths>` | Build or reuse knowledge indexes                   |

Run any command with `--help` for its options.

## License and attribution

LexVerse is licensed under [Apache License 2.0](LICENSE). Upstream code, datasets, skills and services retain their own licenses and usage terms.

Parts of LexVerse are adapted from [Harbor](https://github.com/laude-institute/harbor) (`v0.20.0`, Apache License 2.0).

## Contact

For questions, open a GitHub Issue or email xieh@tsinghua.edu.cn.
