# MASER / MILE Benchmark

In real-world legal practice, drafting a civil complaint rarely starts from a neat summary of facts. A client walks into a lawyer's office with an incomplete story, emotional detours, and missing procedural details. Before putting pen to paper, the lawyer must conduct an interactive, multi-turn consultation—probing for the identity of the parties, the sequence of events, the available evidence, and the exact relief sought.

**MASER** (*Multi-Agent Legal Simulation and Evaluation Framework*, [NAACL 2025 Findings](https://aclanthology.org/2025.findings-naacl.365.pdf)) turns this clinical consultation workflow into **MILE** (*Multi-turn Interactive Legal Evaluation*), an evaluation benchmark built on **693 authentic Chinese civil cases**. In LexVerse, `maser` runs the multi-agent consultation harness end to end: your candidate model acts as the **Lawyer**, interviewing a simulated **Client** (guided by a **Supervisor**) for up to 15 turns before drafting a formal civil complaint, which is then graded along both **Outcome (Goal)** and **Process (Interaction)** dimensions.

---

## 1. Overview

- **Benchmark ID**: `maser`
- **Upstream Repository**: [`FudanDISC/MASER`](https://github.com/FudanDISC/MASER) (commit `7f8eedb6eea21ac0468a727205206fe2543df3a6`)
- **Dataset**: `src/evaluation/evaluate_data/evaluate_data.json` (693 civil cases)
- **Task**: `complaint_generation` (multi-turn consultation trajectory + final civil complaint)
- **Metrics**:
  - **Goal (Complaint Quality)**: `CLI` (Plaintiff Info), `DEF` (Defendant Info), `F_R` (Facts & Reasons), `CLA` (Claims), `EVID` (Evidence), `STA` (Format), `G_PROF` (Professionalism), and `G_AVE` (Goal Average).
  - **Interaction (Consultation Quality)**: `INT` (Interactivity), `I_PROF` (Professionalism), `LOGI` (Logic), and `I_AVE` (Interaction Average).
  - **Overall**: `Total` (`0.5 * G_AVE + 0.5 * I_AVE`).

---

## 2. How to Run

### 2.1 Installation

```bash
python -m pip install -e ".[maser,openai,test]"
python -m lexverse catalog maser
```

### 2.2 Configuration

A MILE evaluation run binds three sets of models:
1. **Candidate Lawyer (`models.default`)**: The model being evaluated.
2. **Environment Simulators (`models.roles.client` & `models.roles.supervisor`)**: The models that role-play the plaintiff and keep the client's responses grounded in the case profile.
3. **Evaluator (`models.evaluators[0]`)**: The independent judge model used during the `Goal` and `Interaction` scoring stages.

Add your API profiles to `configs/secrets.local.yaml`:

```yaml
maser-client:
  base_url: https://api.openai.com/v1
  api_key: ${LEXVERSE_API_KEY}
maser-supervisor:
  base_url: https://api.openai.com/v1
  api_key: ${LEXVERSE_API_KEY}
maser-scoring:
  base_url: https://api.openai.com/v1
  api_key: ${LEXVERSE_API_KEY}
```

Copy `configs/benchmarks/maser.example.yaml` to `configs/benchmarks/maser.local.yaml` and configure your candidate model:

```yaml
schema_version: 1.0

benchmark:
  name: maser
  tasks:
    - {id: "complaint_generation", limit: 2}   # Set limit to null for all 693 cases

models:
  default:
    provider: vllm
    name: Qwen/Qwen2.5-7B-Instruct
    load_parameters:
      dtype: bfloat16
      tensor_parallel_size: 1
      max_model_len: 32768
      trust_remote_code: false
    input: {format: chat}
    generation_parameters:
      temperature: 0
      max_tokens: 4096
  roles:
    client:
      provider: openai_compatible
      profile: maser-client
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0.5, max_tokens: 2048}
    supervisor:
      provider: openai_compatible
      profile: maser-supervisor
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0, max_tokens: 2048}
  evaluators:
    - provider: openai_compatible
      profile: maser-scoring
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0, max_tokens: 1024}

generation:
  offline: false
  concurrency: 1
  max_turns: 15

evaluation:
  concurrency: 4
```

### 2.3 Execution Commands

```bash
# Preflight check
python -m lexverse run maser --config configs/benchmarks/maser.local.yaml --dry-run

# Run generation and evaluation
python -m lexverse run maser --config configs/benchmarks/maser.local.yaml --output-root runs/maser-experiment

# Resume, re-evaluate, or rebuild summary tables
python -m lexverse run --resume runs/maser-experiment
python -m lexverse evaluate --run-dir runs/maser-experiment
python -m lexverse collect --run-dir runs/maser-experiment
```

---

## 3. Upstream Adaptation & Rationale

When inspecting the public [`FudanDISC/MASER`](https://github.com/FudanDISC/MASER) repository, we found that the **inference scripts used for the MILE benchmark evaluation were not included in the release**:
- `src/agents/__init__.py` and `src/legal/__init__.py` import `from .lawyer_fact_evaluate import (... QwenLawyer_fact_evaluate ...)` and `from .law_fact_evaluate import Law_Fact_evaluate`, but neither `src/agents/lawyer_fact_evaluate.py` nor `src/legal/law_fact_evaluate.py` exists in the repository.
- Only the synthetic training-data generation scripts (`src/agents/lawyer_fact.py`, `src/legal/law_fact.py`, and `src/run_law_fact.py`) and the evaluation scripts (`src/evaluation/src_eval/`) were uploaded.

LexVerse bridges this gap by adapting the released `Law_Fact` runner and `src_eval` scorer via two minimal, versioned patches (`patches/01-native-generation.patch` and `patches/02-native-scoring.patch`):

1. **Adapting the Released `Law_Fact` Runner for Evaluation (`01-native-generation.patch`)**:
   - **Context**: Because the evaluation-specific inference files (`src/legal/law_fact_evaluate.py` and `src/agents/lawyer_fact_evaluate.py`) are missing from the upstream repository, we route inference through the released `Law_Fact` runner and `QwenLawyer_fact` agent. In the training-data synthesis script (`Law_Fact`), the `Supervisor` agent rewrites both the client's and the lawyer's utterances.
   - **Adaptation**: Our patch aliases `Agent.Lawyer.FactQwen.evaluate` to `QwenLawyer_fact`, restricts the `Supervisor` to **Client-only supervision** (leaving the candidate lawyer's questions and final complaint unedited), and strips ground-truth reference fields (`output` and `law_reason`) from the generation input.
2. **Preserving Section Headings & Markdown Variants in Complaint Parsing (`02-native-scoring.patch`)**:
   - **Context**: In `src/evaluation/src_eval/eval.py`, `GoalEvaluator._split_info(indictment)` splits the complaint using `re.split("诉讼请求|事实和理由|证据和证据来源，证人姓名和住所", indictment)`. Because `re.split` removes the matched delimiter strings, the three section headings are deleted before `_split_string()` searches for them, causing `_split_string()` to extract empty strings (`""`) for Claims (`CLA`), Facts & Reasons (`F_R`), and Evidence (`EVID`). In addition, LLM outputs often format section headings with Markdown prefixes or numbering (e.g., `### 诉讼请求：`, `**事实和理由：**`, `一、诉讼请求`).
   - **Adaptation**: Our patch retains the section headings when slicing off the party header block (`others = indictment[len(parts[0]):]`) and matches section headings at line starts (supporting standard Markdown/numbered heading formats) so that the candidate model's drafted sections are properly extracted and passed to the judge model.
3. **Multi-Window Interaction Scoring & Short-Dialogue Handling (`02-native-scoring.patch`)**:
   - **Context**: In `--tasks Interaction`, upstream `eval.py` omitted the current turn's prompt from `history`, called `_get_prompt_inter` on the final turn without `dialogue`, overwrote earlier window scores on the final turn, crashed with an `UnboundLocalError` on 2-turn dialogues, and called `time.sleep(60)` after each run.
   - **Adaptation**: Fixing these issues ensures that all dialogue windows are preserved and evaluated cleanly without unnecessary sleep delays.

---

## 4. Reference Results (`lexverse run`) & Implementation Notes

The table below reports the scores obtained directly by running `python -m lexverse run maser` across all **693 cases** with `Qwen/Qwen2.5-7B-Instruct` as `models.default` and `gpt-4o-2024-08-06` as the client, supervisor, and judge model, alongside the reference numbers from Table 2 of the paper:

| Metric | `lexverse run maser` (`Qwen/Qwen2.5-7B-Instruct`, 693 Cases) | Paper Table 2 Reference (`Qwen2.5-7B-Instruct`) |
| --- | ---: | ---: |
| **CLI** (Plaintiff Info) | **92.03** | 86.90 |
| **DEF** (Defendant Info) | **93.90** | 92.05 |
| **F_R** (Facts & Reasons) | **54.42** | 37.81 |
| **CLA** (Claims) | **65.48** | 55.66 |
| **EVID** (Evidence) | **39.77** | 34.23 |
| **STA** (Format) | **38.99** | 41.50 |
| **G_PROF** (Document Professionalism) | **54.36** | 56.09 |
| **G_AVE** (Goal Average) | **62.71** | 57.75 |
| **INT** (Interactivity) | **71.75** *(Window-Micro: 72.93)* | 72.90 |
| **I_PROF** (Inquiry Professionalism) | **66.64** *(Window-Micro: 67.04)* | 68.46 |
| **LOGI** (Inquiry Logic) | **71.55** *(Window-Micro: 71.95)* | 72.73 |
| **I_AVE** (Interaction Average) | **69.98** *(Window-Micro: 70.64)* | 71.36 |
| **Total** (Overall Score) | **66.34** | 64.56 |

### Implementation & Protocol Notes

- **Missing Upstream Evaluation Inference Code (`lawyer_fact_evaluate.py` / `law_fact_evaluate.py`)**: Because the upstream repository did not release the evaluation-time inference classes used to produce the trajectories in the paper, our `lexverse run maser` numbers are generated by adapting the released `Law_Fact` / `QwenLawyer_fact` runner. Slight differences in `CLI` (`92.03` vs. `86.90`) and `DEF` (`93.90` vs. `92.05`) reflect differences between the released `Law_Fact` prompt/loop and the authors' unreleased evaluation runner (when our scorer is run directly on the authors' pre-generated outputs in `src/evaluation/data/eval_qwen7B.jsonl`, it reproduces `CLI = 86.90` and `DEF = 92.05` identically).
- **Complaint Section Formatting & Extraction (`F_R`, `CLA`, `EVID`)**: Candidate models frequently format section headings with Markdown or numbering (e.g., `### 诉讼请求：`, `**事实和理由：**`), and upstream `_split_info` originally stripped the section title strings during `re.split`. Preserving the section titles in `02-native-scoring.patch` ensures that the judge evaluates the actual drafted Claims, Facts & Reasons, and Evidence sections, which accounts for the higher `F_R` (`54.42`), `CLA` (`65.48`), and `EVID` (`39.77`) scores.
- **Interaction Aggregation (`Case-Macro` vs. `Window-Micro`)**: Upstream `main.py` scores each 2-turn dialogue window independently without a cross-case aggregation function. LexVerse reports **Case-Macro** by default (averaging windows within each case first, then across all 693 cases: `INT = 71.75`). When aggregated via **Window-Micro** (averaging all 2,153 dialogue windows equally across the dataset), `INT` is `72.93`, matching the paper's `72.90` within `0.03` points.
