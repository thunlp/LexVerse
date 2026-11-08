# Job configuration

Benchmarks and user Tasks share Job schema `1.0`. Run the commands below from the project root.

## Select a configuration

| Command | YAML selection |
| --- | --- |
| `lexverse run -b lawbench` | `lawbench.local.yaml`, then `lawbench.yaml` in `configs/jobs/` |
| `lexverse run -p user_workspace/tasks/example` | Same lookup using the `task.toml` ID (`example`) |
| `lexverse run -c configs/jobs/lawbench.yaml` | Only the specified file |

Automatic lookup is relative to the current directory. Files are never merged; only a missing file triggers fallback. CLI runtime options override YAML settings; `-b` or `-p` must match any source already specified in YAML.

## Configuration blocks

| Field | Purpose |
| --- | --- |
| `schema_version` | `"1.0"` |
| `benchmark` / `path` | Benchmark selection and filters, or a Task package path |
| `agents` | Agents to run |
| `models.default` | Tested model |
| `models.roles` / `models.summary` | Supporting role and summary models, where required |
| `models.verifiers` | Scoring models, separate from the tested model |
| `generation` / `verification` | Execution and scoring options |
| `resources` | Agent knowledge, Skills and MCP services; `disabled_tools` removes built-in Vanilla tools (Skills require `read_file`) |
| `concurrency` / `attempts` / `max_retries` | Parallelism, attempts and retries; defaults: `1`, `1`, `0` |

## Examples

| YAML | Default Agent | Notes |
| --- | --- | --- |
| [lawbench.yaml](lawbench.yaml) | Static | Also supports Vanilla and External Agents |
| [lexeval.yaml](lexeval.yaml) | Static | Also supports Vanilla and External Agents |
| [plawbench.yaml](plawbench.yaml) | Static | Separate scoring model; also supports Vanilla and External Agents |
| [j1bench.yaml](j1bench.yaml) | Native | Summary and scoring models |
| [dlawbench.yaml](dlawbench.yaml) | Native | Client role and scoring models |
| [legalworld.yaml](legalworld.yaml) | Native | Simulation role and scoring models |
| [simucourt.yaml](simucourt.yaml) | Native | Separate scoring model |
| [maser.yaml](maser.yaml) | Native | Local vLLM model, client/supervisor roles and scoring model |
| [lecardv2.yaml](lecardv2.yaml) | Native | Direct BM25 retrieval; optional encoder and LLM-based methods |
| [example.yaml](example.yaml) | Vanilla | User Task; also supports External Agents |

Prepare the selected model connections, datasets and execution dependencies before running. See the [project README](../../README.md) for setup.

### Reproducible task sampling

`benchmark.default_limit` takes the first records. For paired random-sample comparisons, generate sample IDs once and save each task's selection explicitly:

```yaml
benchmark:
  name: lawbench
  tasks:
  - id: 1-1
    cases: ['122', '236', '457']
  - id: 1-2
    cases: ['16', '84', '311']
```

Each task's `cases` overrides the global `benchmark.cases`; a configured limit applies after filtering. Sample IDs refer to the loader's original record indices. Use the same IDs in every compared Job.

## Personal configuration

```bash
cp configs/jobs/lawbench.yaml configs/jobs/lawbench.local.yaml
# Edit the copied configuration, then preview it.
lexverse run -b lawbench --dry-run
```

`*.local.yaml` files are Git-ignored. Store API credentials in `configs/secrets.local.yaml` and reference their profile names through `models.*.profile`; adjust the example profiles to your connections. YAML-relative paths resolve from the YAML directory; CLI-relative paths resolve from the current directory.
