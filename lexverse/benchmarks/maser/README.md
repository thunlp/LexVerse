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
lexverse task catalog maser
```

### 2.2 Configuration

A MILE evaluation run binds three sets of models:
1. **Candidate Lawyer (`models.default`)**: The model being evaluated.
2. **Environment Simulators (`models.roles.client` & `models.roles.supervisor`)**: The models that role-play the plaintiff and keep the client's responses grounded in the case profile.
3. **Evaluator (`models.verifiers[0]`)**: The independent judge model used during the `Goal` and `Interaction` scoring stages.

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

Copy `configs/jobs/maser.yaml` to `configs/jobs/maser.local.yaml` and configure your candidate model:

```yaml
schema_version: '1.0'
benchmark:
  name: maser
  tasks:
  - id: complaint_generation
    limit: 2
models:
  default:
    provider: vllm
    name: Qwen/Qwen2.5-7B-Instruct
    load_parameters:
      dtype: bfloat16
      tensor_parallel_size: 1
      max_model_len: 32768
      trust_remote_code: false
    input:
      format: chat
    generation_parameters:
      temperature: 0
      max_tokens: 4096
  roles:
    client:
      provider: openai_compatible
      profile: maser-client
      name: gpt-4o-2024-08-06
      generation_parameters:
        temperature: 0.5
        max_tokens: 2048
    supervisor:
      provider: openai_compatible
      profile: maser-supervisor
      name: gpt-4o-2024-08-06
      generation_parameters:
        temperature: 0
        max_tokens: 2048
  verifiers:
  - provider: openai_compatible
    profile: maser-scoring
    name: gpt-4o-2024-08-06
    generation_parameters:
      temperature: 0
      max_tokens: 1024
generation:
  offline: false
  max_turns: 15
agents:
- name: native
verification:
  concurrency: 4
concurrency: 1
```

### 2.3 Execution Commands

```bash
# Preflight check
lexverse run -b maser -c configs/jobs/maser.local.yaml --dry-run

# Run generation and evaluation
lexverse run -b maser -c configs/jobs/maser.local.yaml --output-dir runs/maser-experiment

# Resume, re-evaluate, or rebuild summary tables
lexverse job resume runs/maser-experiment
lexverse job verify runs/maser-experiment
lexverse job report runs/maser-experiment
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
   - **Final-Turn Instruction (deviates from upstream)**: The lawyer's system prompt says "当你没有任何问题要询问原告时……加上<询问结束>，否则你需要继续询问原告", and upstream `Law_Fact` appends the complaint-drafting request (`draft_indictment`) after the last turn without revoking it. Some models therefore keep asking questions instead of writing the complaint (all Goal metrics then collapse), or wrap the complaint in chat preamble and a trailing question. The patch prefixes `draft_indictment` with "询问阶段已经结束，不能再向原告提问。请直接输出起诉状，不要在起诉状前后添加说明或问题。" Trajectories generated before this change are not comparable on Goal metrics.
2. **Preserving Section Headings & Markdown Variants in Complaint Parsing (`02-native-scoring.patch`)**:
   - **Context**: In `src/evaluation/src_eval/eval.py`, `GoalEvaluator._split_info(indictment)` splits the complaint using `re.split("诉讼请求|事实和理由|证据和证据来源，证人姓名和住所", indictment)`. Because `re.split` removes the matched delimiter strings, the three section headings are deleted before `_split_string()` searches for them, causing `_split_string()` to extract empty strings (`""`) for Claims (`CLA`), Facts & Reasons (`F_R`), and Evidence (`EVID`). In addition, LLM outputs often format section headings with Markdown prefixes or numbering (e.g., `### 诉讼请求：`, `**事实和理由：**`, `一、诉讼请求`).
   - **Adaptation**: Our patch retains the section headings when slicing off the party header block (`others = indictment[len(parts[0]):]`) and matches section headings at line starts (supporting standard Markdown/numbered heading formats) so that the candidate model's drafted sections are properly extracted and passed to the judge model.
   - **Evidence Section Extraction**: Upstream `_eval_fact_claim_evidence` keeps only `sections['证据'].split('\n\n')[0]`, i.e. the text before the first blank line. When a model leaves a blank line after the heading (e.g. `## 证据和证据来源，证人姓名和住所` followed by an empty line, common in Markdown output), the judge receives only the heading and `EVID` collapses to the minimum score. Models also abbreviate the heading (`证据和证据来源`, `证据和证据来源、证人姓名和住所`, `证人姓名及住所`), which upstream does not recognise, so the evidence list is merged into Facts & Reasons. The patch accepts these heading variants (plus `事实与理由`), strips the heading while keeping same-line content, and cuts the evidence section at the closing `此致` instead of the first blank line.
   - **Party-Info Split**: Upstream `_split_info` cuts the party block at the first *substring* `诉讼请求`/`事实和理由`/`证据和证据来源，证人姓名和住所` anywhere in the text. If a preamble before the complaint mentions e.g. "诉讼请求", the party block becomes just that preamble and the rule-based `CLI`/`DEF` drop to 0. The patch splits at the first line-start section heading (same heading rules as above) and falls back to the upstream substring split when no heading line exists.
   - **Empty Sections**: When the Claims, Facts & Reasons, or Evidence section is missing or contains only its heading (e.g. the model never wrote a complaint), the judge still received an empty `generated` field, and `gpt-4o-2024-08-06` frequently returned 10/10 for it. The patch records the minimum score (1) for an empty section without calling the judge.
3. **Multi-Window Interaction Scoring & Short-Dialogue Handling (`02-native-scoring.patch`)**:
   - **Context**: In `--tasks Interaction`, upstream `eval.py` omitted the current turn's prompt from `history`, called `_get_prompt_inter` on the final turn without `dialogue`, overwrote earlier window scores on the final turn, crashed with an `UnboundLocalError` on 2-turn dialogues, and called `time.sleep(60)` after each run.
   - **Adaptation**: Fixing these issues ensures that all dialogue windows are preserved and evaluated cleanly without unnecessary sleep delays.
   - **Judge Retries**: Judge calls are retried up to `verification.max_attempts` times. Rate-limit (429) responses wait for `Retry-After`, and connection/timeout/5xx errors back off exponentially; invalid JSON responses are retried immediately as before. Each scoring subprocess runs up to 10 judge calls in parallel, so on token-limited deployments (e.g. 75k TPM) keep `verification.concurrency` low and raise `max_attempts`.

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
- **Later Fixes and Reference Numbers**: The reference table above was produced before the evidence-section, party-info split, and empty-section fixes in `02-native-scoring.patch` and before the final-turn instruction in `01-native-generation.patch`. Qwen2.5-7B usually writes evidence directly under the heading and starts the complaint without a preamble, so the earlier extraction mostly worked for it, but `EVID`/`F_R`/`CLI`/`DEF` may shift on re-scoring, and Goal metrics may shift on re-generation.
- **Interaction Aggregation (`Case-Macro` vs. `Window-Micro`)**: Upstream `main.py` scores each 2-turn dialogue window independently without a cross-case aggregation function. LexVerse reports **Case-Macro** by default (averaging windows within each case first, then across all 693 cases: `INT = 71.75`). When aggregated via **Window-Micro** (averaging all 2,153 dialogue windows equally across the dataset), `INT` is `72.93`, matching the paper's `72.90` within `0.03` points.
