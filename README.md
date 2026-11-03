<div align="center">

# LexVerse

*A universe where legal agents learn, act, and are evaluated.*

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![Benchmarks](https://img.shields.io/badge/Benchmarks-9-334155)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow)

</div>

LexVerse is **a unified framework** for legal language models and agents, spanning from benchmark evaluation to real-world legal tasks. On the evaluation side, it integrates diverse legal benchmarks into a standardized workflow for task preparation, execution, and result management—preserving each benchmark's native formats and evaluators while supporting capability-augmented testing and trajectory-level diagnostics. On the practice side, it provides configurable runtime environments and sandboxes equipped with legal knowledge retrieval, domain skills, and MCP tools, enabling agents to process custom user materials, ground their reasoning in authoritative sources, and produce verifiable deliverables.

Researchers can use it to evaluate models and conduct controlled ablation studies, developers can use it to build and customize legal agents, and legal professionals can bring their own materials to test whether AI holds up in real workflows. **Whoever you are, LexVerse is a place to start.** We are continuing to integrate more benchmarks, expand our framework and runtime environments, and unlock multi-agent workflows and trajectory-based learning.

## Supported benchmark

| Benchmark                                                 | Coverage                                         | Execution                   |
| :-------------------------------------------------------- | :----------------------------------------------- | :-------------------------- |
| [LexEval](https://github.com/CSHaitao/LexEval)             | 23 task types                                    | Static / Vanilla / External |
| [LawBench](https://github.com/open-compass/LawBench)       | 20 task types                                    | Static / Vanilla / External |
| [PLawBench](https://github.com/SKYLENAGE-AI/PLawBench)     | Case analysis, consultation, document generation | Static / Vanilla / External |
| [J1Bench](https://github.com/FudanDISC/J1Bench)            | CI, CR, KQ, LC, CD, DD                           | Native                      |
| [DLawBench](https://github.com/SKYLENAGE-AI/DLawBench)     | 4 client personas                                | Native                      |
| [Legal-world](https://github.com/sii-research/Legal-world) | 6 scenarios across the litigation lifecycle      | Native                      |
| [SimuCourt](https://github.com/Zhitao-He/AgentsCourt)      | 6 task types                                     | Native                      |
| [MASER](https://github.com/FudanDISC/MASER)                | Complaint generation                             | Native                      |
| [LeCaRDv2](https://github.com/THUIR/LeCaRDv2)              | Case retrieval                                   | Static                      |

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
pip install -e ".[lawbench,lexeval,j1bench,plawbench,dlawbench,legalworld,simucourt,maser,lecardv2,openai]"
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
lexverse setup
```

#### Step 2. Prepare Skills and knowledge (optional)

`setup` does not download Skills or knowledge. Downloads are saved under the project root's `.lexverse/capabilities/` by default.

Declare Agent resources in Job `resources` or Task environment settings. Static and native executions do not accept Agent resources. For benchmark evaluation, keep reference answers out of knowledge sources.

**Download Skills:**

```bash
lexverse resources download skills
```

**Prepare knowledge using either option:**

- Download prebuilt indexes, ready for retrieval:

  ```bash
  lexverse resources download indexes
  # Or download selected collections only:
  lexverse resources download indexes --collections legal_laws legal_cases/JuDGE
  ```
- Download source documents and build indexes locally:

  ```bash
  lexverse resources download knowledge
  # Example: build the legal_laws collection.
  lexverse resources build-index --knowledge .lexverse/capabilities/knowledge/legal_laws --mode hybrid
  ```

Prebuilt indexes use **hybrid retrieval** (BM25 keyword search + FAISS vector search), with **`BAAI/bge-small-zh-v1.5`** as the embedding model. Queries also require this model. Use the same hybrid configuration in your task; select downloaded collections with IDs such as `collection:legal_laws`. For custom index settings, see [retrieval.example.yaml](configs/knowledge/retrieval.example.yaml).

#### Step 3. Initialize a task

Task packages define the work, environment and verification. Job configurations independently select models, Agents and execution settings using the interfaces below.

```bash
lexverse task init user_workspace/tasks/contract-review
```

```text
user_workspace/tasks/contract-review/
├── task.toml                 # Environment, artifacts and verifier
├── instruction.md            # Task instructions
├── environment/
│   ├── Dockerfile
│   └── inputs/               # Task materials; copied to /inputs/ when building the image
└── tests/
    ├── Dockerfile            # Dedicated verifier image
    ├── test.sh
    └── grade.py
```

#### Step 4. Configure an experiment

User tasks support vanilla and external Agents. Job configuration is optional and separate from task initialization:

```bash
lexverse job init configs/jobs/contract-review.local.yaml \
  -p user_workspace/tasks/contract-review --agents vanilla,codex \
  --model gpt-5.6-terra --profile openai
lexverse run -p user_workspace/tasks/contract-review --dry-run
```

To override the selected YAML for one run (or provide all required options when no YAML exists):

```bash
lexverse run -p user_workspace/tasks/contract-review \
  --agents vanilla,codex --model gpt-5.6-terra --profile openai --dry-run
```

`--attempts` specifies independent repetitions, `--max-retries` specifies additional retries on execution errors, and `--concurrency` bounds simultaneous Trials.

Run the task with the saved configuration:

```bash
lexverse run -p user_workspace/tasks/contract-review
```

#### Step 5. View results

Results are saved under `runs/user_tasks/<task_id>/<job_id>/` or `runs/benchmark_tasks/<benchmark>/<job_id>/`. Each Job includes its configuration, progress summary, per-Trial results and artifacts, and a comparison report at `verification/report.csv`.

### Benchmark task

Run a benchmark with its example Job configuration:

```bash
lexverse run -b lawbench
```

This uses `configs/jobs/lawbench.local.yaml` if present, otherwise `configs/jobs/lawbench.yaml`.

Override the Agent and model, or select a configuration explicitly:

```bash
lexverse run -b lawbench --agents static --model MODEL --profile openai
lexverse run -c configs/jobs/lawbench.yaml
```

Optional operations:

```bash
# Compare Agents with two independent attempts each
lexverse run -b lawbench --agents vanilla,codex --attempts 2
# Generate answers without scoring
lexverse run -b lawbench --no-verify
# Resume an unfinished Job
lexverse job resume runs/benchmark_tasks/lawbench/<job_id>
```

## Commands

| Command                                                        | Purpose                                                                             |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `lexverse setup`                                             | Prepare Agent dependencies                                                          |
| `lexverse agents list`                                       | List registered Agent choices                                                       |
| `lexverse resources download knowledge/indexes/skills`       | Download resources                                                                  |
| `lexverse resources build-index --knowledge <paths>`         | Build knowledge indexes                                                             |
| `lexverse task init <path>`                                  | Create a Task package                                                               |
| `lexverse task catalog <benchmark>`                          | List Benchmark Tasks and Agent support                                              |
| `lexverse job init <config.yaml> ...`                        | Save a Job configuration                                                            |
| `lexverse job schema`                                        | Print the Job configuration JSON Schema                                             |
| `lexverse run -p <path> / -b <benchmark> / -c <config.yaml>` | Start a Job                                                                         |
| `lexverse job start ...`                                     | Start a Job (same options as `run`)                                               |
| `lexverse job inspect <job_dir>`                             | Read current Job status without writing                                             |
| `lexverse trial inspect <trial_dir> [--trajectory]`          | Inspect one Trial                                                                   |
| `lexverse job resume <job_dir>`                              | Resume unfinished Trials                                                            |
| `lexverse trial resume <trial_dir> --reply <text> / --retry` | Resume one paused or failed Trial                                                   |
| `lexverse job verify <job_dir>`                              | Score saved Benchmark answers or saved User Task artifacts with a separate verifier |
| `lexverse job report <job_dir>`                              | Rebuild derived reports                                                             |

Run any command with `--help` for its options.

## License and attribution

LexVerse is licensed under [Apache License 2.0](LICENSE). Upstream code, datasets, skills and services retain their own licenses and usage terms.

Parts of LexVerse are adapted from [Harbor](https://github.com/laude-institute/harbor) (`v0.20.0`, Apache License 2.0).

## Contact

For questions, open a GitHub Issue or email xieh@tsinghua.edu.cn.
