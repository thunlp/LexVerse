# SimuCourt Benchmark

Judicial adjudication takes different forms depending on the trial stage and the branch of law. In a **first-instance trial**, a judge weighs factual allegations, competing claims, and defense arguments to establish primary liability or guilt. In a **second-instance appellate review**, the judge examines the lower court's findings of fact, statutory citations, and sentencing or civil relief against the appellant's grounds for appeal—deciding whether to affirm or modify the judgment.

**SimuCourt** (introduced in *AgentsCourt: Building Judicial Decision-Making Agents with Court Debate Simulation and Legal Knowledge Augmentation*, [EMNLP 2024 Findings](https://aclanthology.org/2024.findings-emnlp.939/)) benchmarks this judicial decision-making capability across **criminal, civil, and administrative** proceedings at both first-instance and second-instance levels. In LexVerse, `simucourt` evaluates single-turn judicial decision generation across all 419 public cases using the upstream repository's native LLM extraction and scoring pipeline.

---

## 1. Overview

- **Benchmark ID**: `simucourt`
- **Upstream Repository**: [`Zhitao-He/AgentsCourt`](https://github.com/Zhitao-He/AgentsCourt) (commit `7c14d6960abfdfdd7366250905da8dbcb416d6d6`)
- **Dataset**:
  - `data/SimuCourt(1).json`: **239 first-instance cases** (80 criminal, 79 civil, 80 administrative)
  - `data/SimuCourt(2).json`: **180 second-instance cases** (60 criminal, 60 civil, 60 administrative)
- **Tasks (6)**: `first_criminal`, `first_civil`, `first_administrative`, `second_criminal`, `second_civil`, `second_administrative`
- **Expected Output Format**: Three labeled sections: `[法官分析]` (Judicial Analysis), `[审判依据]` (Statutory Basis), and `[最终判决]` (Final Judgment)
- **Metrics**:
  - **Criminal**: `charge_exact`, `term_exact`, `fine_exact`, `law_micro_f1`
  - **Civil & Administrative**: `law_micro_f1`, `judgment_micro_f1`

---

## 2. How to Run

### 2.1 Installation

```bash
python -m pip install -e ".[simucourt,openai,test]"
python -m lexverse catalog simucourt
```

### 2.2 Configuration

In the upstream SimuCourt pipeline, evaluation requires an independent model in `models.evaluators`:
1. **Structural Extraction (`judge_result_process.py`)**: Prompts an LLM to parse the candidate model's free-text `[审判依据]` and `[最终判决]` sections into structured Python lists and dictionaries.
2. **Metric Scoring (`evaluation_*.py`)**: Statutory citations (`evaluation_laws.py`) and criminal holdings (`evaluation_crimial.py`) are scored deterministically from the extracted dictionary, while civil and administrative holdings (`evaluation_other.py`) prompt an LLM judge to count matched key points.

Copy `configs/benchmarks/simucourt.example.yaml` to `configs/benchmarks/simucourt.local.yaml` and configure your candidate model (`models.default`) and evaluator (`models.evaluators`):

```yaml
schema_version: 1.0

benchmark:
  name: simucourt
  tasks:
    - {id: "first_criminal", limit: 1}        # Set limit to null for all 419 cases
    - {id: "first_civil", limit: 1}
    - {id: "first_administrative", limit: 1}
    - {id: "second_criminal", limit: 1}
    - {id: "second_civil", limit: 1}
    - {id: "second_administrative", limit: 1}

models:
  default:
    provider: openai_compatible
    profile: openai
    name: gpt-4o-mini
    generation_parameters:
      temperature: 0
      max_tokens: 4096
  evaluators:
    - provider: openai_compatible
      profile: openai
      name: gpt-4o-mini
      generation_parameters:
        temperature: 0
        max_tokens: 4096

generation:
  concurrency: 2

evaluation:
  concurrency: 2
  max_attempts: 3
```

### 2.3 Execution Commands

```bash
# Preflight check
python -m lexverse run simucourt --config configs/benchmarks/simucourt.local.yaml --dry-run

# Run generation, structural extraction, and scoring
python -m lexverse run simucourt --config configs/benchmarks/simucourt.local.yaml --output-root runs/simucourt-experiment

# Resume, re-evaluate, or rebuild summary tables
python -m lexverse run --resume runs/simucourt-experiment
python -m lexverse evaluate --run-dir runs/simucourt-experiment
python -m lexverse collect --run-dir runs/simucourt-experiment
```

---

## 3. Upstream Adaptation & Rationale

The upstream [`Zhitao-He/AgentsCourt`](https://github.com/Zhitao-He/AgentsCourt) scripts (`judge_analysis_record.py`, `judge_result_process.py`, `evaluation_laws.py`, `evaluation_crimial.py`, `evaluation_other.py`) were written as standalone scripts with top-level file I/O, `eval()` calls, and second-instance-only templates. LexVerse adapts them via `patches/0001-native-callables.patch`:

1. **First-Instance Prompt & Scorer Support (`single_candidate_prompt.py`, `judge_result_process.py`, `evaluation_other.py`)**:
   - **Context**: Upstream `judge_analysis_record.py` only defines `CASE_TEMPLATE2` (for second-instance cases), `evaluation_other.py` hardcodes `current_case["一审法院审判结果"]` inside its loop, and `judge_result_process.py` only includes second-instance criminal extraction examples (`"罪名": "上诉人犯XX罪"`, whereas first-instance gold labels use `"被告人犯XX罪"`).
   - **Rationale**: Our patch adds a symmetric `FIRST_TEMPLATE` mapping the fields in `SimuCourt(1).json`, switches `evaluation_other.py` between `CASE_TEMPLATE1` and `CASE_TEMPLATE2` by trial stage, and uses `"被告人犯"` in the first-instance criminal extraction prompt so all 6 task categories run cleanly.
2. **Callable Module Wrapping & Safe Literal Parsing (`judge_result_process.py`)**:
   - **Context**: Upstream `judge_result_process.py` runs at import time, sleeps 10 seconds per case, filters cases against a hardcoded 3-case debug list (`error_list1`), and calls `eval()` on raw LLM strings.
   - **Rationale**: Removing `error_list1` and `time.sleep(10)` enables full-dataset evaluation, and replacing `eval()` with `json.loads()` / `ast.literal_eval()` ensures safe parsing.
3. **Reference Isolation (`tasks.py`)**:
   - **Context**: The upstream `evaluation_*.py` scripts read an unreleased `judge_result.json` file.
   - **Rationale**: `SimuCourtImporter` deterministically constructs the exact reference dictionary expected by the native scorers from the public fields in `SimuCourt(1).json` and `SimuCourt(2).json` while keeping ground-truth references strictly isolated from candidate prompts.

---

## 4. Reference Results (`lexverse run`) & Implementation Notes

The table below shows the results obtained directly by running `python -m lexverse run simucourt` across all **180 second-instance cases** (`second_criminal`: 60, `second_civil`: 60, `second_administrative`: 60), alongside the single-model baseline numbers reported in Table 5 of the paper:

| Model / Source | Cases | Statutory Basis (`law_micro_f1`) | Civil & Admin Holding (`judgment_micro_f1`) | Criminal Charge (`charge_exact`) | Criminal Term (`term_exact`) | Criminal Fine (`fine_exact`) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| *Paper Table 5: GPT-3.5* | 180 | 0.186 | 0.365 | 0.716 | 0.166 | 0.516 |
| *Paper Table 5: GPT-4* | 180 | 0.228 | 0.409 | 0.800 | 0.183 | 0.533 |
| *Paper Table 5: LaWGPT* | 180 | 0.227 | 0.422 | 0.850 | 0.133 | 0.483 |
| **`lexverse run simucourt` (`gpt-6-luna`)** | **180** | **0.453** | **0.740** | **0.350** | **0.217** | **0.450** |

### Implementation & Protocol Notes

- **Exact-String Matching in Criminal Scoring (`charge_exact`, `fine_exact`)**: Upstream `evaluation_crimial.py` compares extracted dictionary values against gold strings using literal `==`. In upstream `judge_result_process.py`, Example 4 of `prompt_crime_result` includes the appellant's name inside the charge string (`{"罪名": "上诉人陈其洋犯故意伤害罪"}`) and omits the `"罚金"` key when no fine is imposed (`None`), whereas `SimuCourt(2).json` omits the appellant's name in 56 of 60 cases (`"上诉人犯..."`) and encodes no-fine cases as `"无"`. Because LexVerse preserves the unmodified upstream `==` scorer, predictions that include the appellant's name or omit the fine key are counted as mismatches under strict scoring (normalizing those two surface conventions corresponds to `0.950` on `charge_exact` and `0.800` on `fine_exact`).
- **Context Length Requirements**: The upstream 3-shot system prompt (`judge_analysis_record2.txt`) is ~4,980 tokens long. Combined with the case facts, models evaluated under the standard 3-shot prompt require at least an 8K context window to emit all three required section headers (`[法官分析]`, `[审判依据]`, `[最终判决]`).
