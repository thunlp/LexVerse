# DLawBench Benchmark

In real-world legal practice, a consultation rarely begins with a clean, chronological summary of facts. A client may arrive anxious, guarded, or convinced of a one-sided legal theory that omits unfavorable details. Before offering legal conclusions, a lawyer must conduct an interactive intake interview—probing for missing documents, testing the client's assumptions, separating subjective beliefs from verifiable facts, and synthesizing a grounded legal memorandum.

**DLawBench** (*Evaluating LLMs Through Multi-Turn Legal Consultation*, [arXiv:2606.13931](https://arxiv.org/abs/2606.13931)) benchmarks this diagnostic workflow across **Chinese law (`CN`) and U.S. law (`US`)**. Each case pairs a **Client Belief View** (colloquial, incomplete, or doctrinally misframed statements used by the simulated client) with a hidden **Court Record View** (calibrated findings and legal reasoning from the actual court opinion, reserved for evaluation). Every case is replayed across four interpersonal client personas—**Cooperative (`P_CS`)**, **Dependent (`P_CD`)**, **Withdrawn (`P_WS`)**, and **Adversarial (`P_WD`)**—and scored across three capability dimensions: **Information Gathering**, **Legal Reasoning**, and **Claim Support**.

---

## 1. Overview

- **Benchmark ID**: `dlawbench`
- **Upstream Repository**: [`SKYLENAGE-AI/DLawBench`](https://github.com/SKYLENAGE-AI/DLawBench) (commit `9f6325f29f07b296fac110a6e1880bdd5d9ca538`)
- **Dataset**: `data/case_backgrounds/*.jsonl` (60 public release cases: 30 Chinese-law + 30 U.S.-law × 4 personas = **240 consultation tasks**)
- **Tasks (Client Personas)**:
  - `P_CS` (*Cooperative*): Calm, organized client who answers clearly and provides requested details.
  - `P_CD` (*Dependent*): Anxious client who lacks legal knowledge and relies heavily on the lawyer's guidance.
  - `P_WS` (*Withdrawn*): Guarded, reluctant client who gives minimal replies unless asked targeted questions.
  - `P_WD` (*Adversarial*): Assertive client who pushes back on questioning and insists on their own framing.
- **Metrics**:
  - **Information Gathering**: `fact_coverage` (share of annotated facts surfaced in the memo), `inquiry` (share of expert-specified follow-up questions asked), and `elicitation` (`(fact_coverage + inquiry) / 2`).
  - **Legal Reasoning**: `fact_resolution` (share of all annotated facts both covered and calibrated against the court-record view), `issue_resolution` (share of expert issue rubrics addressed), and `resolution` (`(fact_resolution + issue_resolution) / 2`, the primary leaderboard metric).
  - **Claim Support**: `fidelity` (`1 - ungrounded_claims / total_claims`, measuring whether memo claims are grounded in the dialogue or case facts).

---

## 2. How to Run

### 2.1 Installation

```bash
python -m pip install -e ".[dlawbench,openai,test]"
python -m lexverse catalog dlawbench
```

### 2.2 Configuration

A DLawBench run binds three model roles:
1. **Candidate Lawyer (`models.default` or `models.roles.lawyer`)**: The model under evaluation, which calls `talk_to_client(message)` for up to `generation.max_turns` (default `10`) turns before submitting `write_memo(content)`.
2. **Client Simulator (`models.roles.client`)**: Role-plays the client under the assigned persona (`P_CS`, `P_CD`, `P_WS`, or `P_WD`) using only the case's client-belief entries.
3. **Evaluator (`models.evaluators`)**: Supports `evaluation.mode: single` (one judge model) or `evaluation.mode: panel` (three judge models with automatic same-vendor recusal and median/mean aggregation).

Copy `configs/benchmarks/dlawbench.example.yaml` to `configs/benchmarks/dlawbench.local.yaml`:

```yaml
schema_version: 1.0

benchmark:
  name: dlawbench
  tasks:
    - {id: "P_CS", limit: 5}   # Set limit to null for all 60 cases per persona
    - {id: "P_CD", limit: 5}
    - {id: "P_WS", limit: 5}
    - {id: "P_WD", limit: 5}

models:
  default:
    provider: vllm
    name: CSHaitao/LegalOne-8B
    load_parameters:
      dtype: bfloat16
      tensor_parallel_size: 1
      max_model_len: 32768
      trust_remote_code: false
    input: {format: chat}
    generation_parameters:
      temperature: 0
      max_tokens: 8192
  roles:
    client:
      provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters:
        temperature: 0
        max_tokens: 2048
  evaluators:
    - provider: openai_compatible
      profile: openai
      name: gpt-5.1-2025-11-13
      generation_parameters:
        max_tokens: 32768

generation:
  concurrency: 5
  max_turns: 10
  timeout_sec: 1800

evaluation:
  mode: single
  concurrency: 4
  max_attempts: 3
  timeout_sec: 1800
```

### 2.3 Execution Commands

```bash
# Preflight check
python -m lexverse run dlawbench --config configs/benchmarks/dlawbench.local.yaml --dry-run

# Run multi-turn consultation and rubric evaluation
python -m lexverse run dlawbench --config configs/benchmarks/dlawbench.local.yaml --output-root runs/dlawbench-experiment

# Resume, re-evaluate, or rebuild summary tables
python -m lexverse run --resume runs/dlawbench-experiment
python -m lexverse evaluate --run-dir runs/dlawbench-experiment
python -m lexverse collect --run-dir runs/dlawbench-experiment
```

---

## 3. Upstream Adaptation & Rationale

LexVerse executes the native `dlawbench` package (`agent_environment.py`, `client_npc.py`, `evaluate.py`, `panel_evaluate.py`, and `scripts/metrics.py`) in an isolated subprocess per Trial without modifying the upstream repository:

1. **Strict Perspective Isolation (`tasks.py` & `environment.py`)**:
   - **Context**: Each record in `data/case_backgrounds/*.jsonl` contains the opening statement, paired `client_belief` and `court_record` facts, and expert rubrics (`expected_inquiries` and `memo_expectations`).
   - **Adaptation**: `DLawBenchImporter` exposes only `row["opening"]` in `TaskInput`. During generation, `DLawBenchEnvironment` invokes upstream `AgentEnvironment(case, persona, "lawyer", "client")`, which passes only the opening statement and jurisdiction to the lawyer agent and only `client_belief` entries to the client simulator. The `court_record` facts and rubrics remain hidden until the evaluation subprocess runs.
2. **Strict Judge Output Validation (`environment.py:validate_response`)**:
   - **Context**: Upstream `evaluate.py` issues two judge prompts per session: (1) Discovery, Accuracy, and Grounding (DAG) over annotated facts and memo claims, and (2) binary `0/1` scores for `memo_expectation_scores` and `expected_action_scores`. If an LLM judge omits a fact ID or miscounts `total_claims`, upstream `metrics.py` can silently miscompute denominators.
   - **Adaptation**: `validate_response` verifies that `covered` and `missed` partition the exact set of case fact IDs, that `correct` and `incorrect` partition `covered`, that `total_claims == len(grounded_claims) + len(ungrounded_claims)`, and that rubric score vectors match the exact rubric lengths, retrying up to `evaluation.max_attempts` on malformed judge JSON.
3. **Single-Judge and 3-Judge Panel Modes (`plugin.py` & `evaluation.py`)**:
   - **Context**: The paper leaderboard uses a 3-judge panel (`gpt-5.1-2025-11-13`, `claude-opus-4-6-20260205`, `gemini-3.1-pro-preview`) with same-vendor recusal, while upstream also provides `evaluate.evaluate_session` for single-judge evaluation.
   - **Adaptation**: LexVerse supports both `evaluation.mode: single` (1 evaluator model) and `evaluation.mode: panel` (3 evaluator models) and aggregates results via upstream `scripts/metrics.py:build_summary_rows` across personas and jurisdictions (`overall` and `by_jurisdiction`).

---

## 4. Reference Results (`lexverse run`) & Implementation Notes

The table below reports the results obtained by running `python -m lexverse run dlawbench` on two open-weight models from Table 4 of the paper—**`Kimi-K2.6`** (`moonshotai/Kimi-K2.6`) and **`CSHaitao/LegalOne-8B`** (run locally via `vllm` on an NVIDIA A100 GPU)—across all 4 client personas (`P_CS`, `P_CD`, `P_WS`, `P_WD`), alongside the paper's reference numbers:

| Open-Weight Model | Source | Fact Cov. | Inquiry | **Elicitation** | Fact Res. | Issue Res. | **Resolution** | **Fidelity** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Kimi-K2.6`** (`moonshotai/Kimi-K2.6`) | Paper Table 4 | 0.730 | 0.376 | 0.553 | 0.489 | 0.358 | 0.424 | 0.868 |
| | **`lexverse run dlawbench`** | **0.725** | **0.366** | **0.545** | **0.482** | **0.338** | **0.410** | **0.895** |
| *DeepSeek-V3.2-Thinking* (`deepseek-ai/DeepSeek-V3.2`) | Paper Table 4 | 0.710 | 0.308 | 0.509 | 0.420 | 0.219 | 0.320 | 0.898 |
| *DeepSeek-R1* (`deepseek-ai/DeepSeek-R1-0528`) | Paper Table 4 | 0.671 | 0.292 | 0.482 | 0.377 | 0.166 | 0.272 | 0.842 |
| *Olmo-3.1-32B-Instruct* (`allenai/Olmo-3.1-32B-Instruct`) | Paper Table 4 | 0.486 | 0.123 | 0.304 | 0.196 | 0.054 | 0.125 | 0.618 |
| **`CSHaitao/LegalOne-8B`** | Paper Table 4 | 0.202 | 0.095 | 0.148 | 0.102 | 0.050 | 0.076 | 0.268 |
| | **`lexverse run dlawbench`** | **0.108** | **0.110** | **0.109** | **0.057** | **0.048** | **0.052** | **0.127** |

### Persona Breakdown (`Kimi-K2.6` via `lexverse run dlawbench`)

| Persona Task | Style Description | `fact_coverage` | `inquiry` | `elicitation` | `fact_resolution` | `issue_resolution` | `resolution` | `fidelity` |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P_CS` | Cooperative | 0.633 | 0.416 | 0.525 | 0.457 | 0.275 | 0.366 | 0.889 |
| `P_CD` | Dependent | 0.763 | 0.380 | 0.572 | 0.523 | 0.308 | 0.416 | 0.935 |
| `P_WS` | Withdrawn | 0.733 | 0.258 | 0.496 | 0.510 | 0.383 | 0.447 | 0.835 |
| `P_WD` | Adversarial | 0.770 | 0.409 | 0.590 | 0.437 | 0.383 | 0.410 | 0.920 |

### Implementation & Protocol Notes

- **Public Release Subset (60 Cases vs. 461 Full Paper Corpus)**: As documented in `configs/benchmarks/dlawbench.example.yaml`, the pinned public GitHub release (`9f6325f2`) includes 60 representative cases (30 `CN` + 30 `US`, 240 total persona-case cells), whereas Table 4 of the paper aggregates over the authors' 461-case full corpus (264 `CN` + 197 `US`).
- **Single-Judge (`gpt-5.1`) vs. 3-Judge Panel**: Our validation run uses `evaluation.mode: single` with `gpt-5.1` (the first judge in the paper's panel) and `gpt-4o-2024-08-06` as the client simulator. As analyzed in Appendix F.2 (Table 22) of the DLawBench paper, single-judge and mean/median panel aggregations exhibit a typical absolute difference of `0.035–0.040` on `Elicitation` and `Resolution`.
- **Tool-Call Formatting in Domain-Specialized Models (`LegalOne-8B`)**: Upstream `AgentEnvironment` requires the lawyer model to emit strict JSON tool calls (`{"tool": "talk_to_client", "message": "..."}` or `{"tool": "write_memo", "content": "..."}`) and to invoke `write_memo` before exhausting the 10-turn budget. Consistent with Section 3.1 of the paper, 8B domain-specialized models frequently fail to transition from dialogue to `write_memo` on time under `P_WS` and `P_CS` (producing empty memos that are scored as `0.0` by `scripts/metrics.py`).
