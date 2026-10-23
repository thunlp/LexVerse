# J1-EVAL (J1Bench) Benchmark

In realistic legal practice, an AI assistant rarely operates in a single-turn vacuum. A junior associate answering public legal inquiries must clarify ambiguous questions across turns; a litigator drafting a civil complaint or statement of defense must interview the client to uncover missing facts and evidence; and a presiding judge in a civil or criminal hearing must manage multi-party courtroom procedures before rendering a reasoned judgment.

**J1-EVAL (`J1Bench`)** (*Dynamic Evaluation of Large Language Models on Chinese Legal Multi-Role Scenarios*, [arXiv:2507.04037](https://arxiv.org/abs/2507.04037)) evaluates LLMs across **six dynamic multi-agent legal scenarios** (`KQ`, `LC`, `CD`, `DD`, `CI`, and `CR`, totaling **508 cases**) by pairing the candidate model (`trainee`, `lawyer`, or `judge`) with interactive environment agents (`general_public`, `specific_character`, `plaintiff`, `defendant`, and `procurator`) and rubric-based evaluators.

---

## 1. Overview

- **Benchmark ID**: `j1bench`
- **Upstream Repository**: [`FudanDISC/J1Bench`](https://github.com/FudanDISC/J1Bench) (commit `61a6ea5515e11229c13353e73b0629e39dcc6076`)
- **Dataset**: [`CimoInkPool/J1-Eval_Dataset`](https://huggingface.co/datasets/CimoInkPool/J1-Eval_Dataset) (revision `5fbc73ce3de1bbce474eed2dab27b4d01cedfd7f`, gated HuggingFace dataset; **508 cases** across 6 scenarios)
- **Scenarios & Metrics**:
  - `KQ` (*Legal Knowledge QA*, 98 cases): Multi-turn legal knowledge Q&A between `trainee` (evaluated) and `general_public`. Metrics: `BIN` (binary accuracy), `NBIN` (non-binary completeness), `AVE` (`(BIN + NBIN) / 2`).
  - `LC` (*Legal Consultation*, 62 cases): Multi-turn client consultation between `trainee` (evaluated) and `general_public`. Metrics: `BIN`, `NBIN`, `AVE`.
  - `CD` (*Civil Complaint Drafting*, 93 cases): Client intake and complaint drafting by `lawyer` (evaluated) with `specific_character`. Metrics: `PLA` (plaintiff info), `DEF` (defendant info), `CLA` (claims), `FAC` (facts), `EVI` (evidence), `DOC` (`0.2 * (PLA + DEF + CLA + FAC + EVI)`), `FOR` (format compliance), `AVE` (`(DOC + FOR) / 2`).
  - `DD` (*Defense Statement Drafting*, 93 cases): Client intake and defense drafting by `lawyer` (evaluated) with `specific_character`. Metrics: `RES` (respondent info), `DEF` (defense arguments), `EVI` (evidence), `DOC` (`(RES + DEF + EVI) / 3`), `FOR`, `AVE`.
  - `CI` (*Civil Trial*, 93 cases): Multi-party civil courtroom simulation presided over by `judge` (evaluated) with `plaintiff` and `defendant`. Metrics: `REA` (reasoning), `LAW` (statute citation), `JUD` (judgment accuracy), `PFS` (procedural fidelity score).
  - `CR` (*Criminal Trial*, 69 cases): Multi-party criminal courtroom simulation presided over by `judge` (evaluated) with `procurator`, `defendant`, and `lawyer`. Metrics: `CRI` (charge accuracy), `SEN` (sentence accuracy), `FINE` (fine accuracy), `REA` (reasoning), `LAW` (statute citation), `PFS` (procedural fidelity score).

---

## 2. How to Run

### 2.1 Installation & Dataset Access

Because [`CimoInkPool/J1-Eval_Dataset`](https://huggingface.co/datasets/CimoInkPool/J1-Eval_Dataset) is a gated HuggingFace dataset, accept the dataset terms on HuggingFace and configure `HF_TOKEN` (or `huggingface-cli login`) before running:

```bash
python -m pip install -e ".[j1bench,openai,test]"
export HF_TOKEN="hf_..."
python -m lexverse catalog j1bench
```

### 2.2 Configuration

A J1Bench run binds four model scopes:
1. **Candidate Model (`models.default`)**: Assigned to the scenario's evaluated role (`trainee` in `KQ`/`LC`, `lawyer` in `CD`/`DD`, and `judge` in `CI`/`CR`).
2. **Environment Simulators (`models.roles.*`)**: Role-play the counterpart agents (`general_public`, `specific_character`, `plaintiff`, `defendant`, `procurator`; default `gpt-4o-2024-11-20` in upstream).
3. **Topic State Tracker (`models.summary`)**: Tracks covered vs. remaining topics after each turn in `KQ` and `LC` (`gpt-4o-2024-08-06` in upstream `utils_func.py`).
4. **Evaluator (`models.evaluators[0]`)**: Grades open-ended rubrics across all 6 scenarios (`gpt-4o-2024-11-20` in upstream `src/Eval/eval.py`).

Copy `configs/benchmarks/j1bench.example.yaml` to `configs/benchmarks/j1bench.local.yaml`:

```yaml
schema_version: 1.0

benchmark:
  name: j1bench
  tasks:
    - {id: "KQ", limit: 2}   # Set limit to null for full 508-case evaluation
    - {id: "LC", limit: 2}
    - {id: "CD", limit: 2}
    - {id: "DD", limit: 2}
    - {id: "CI", limit: 2}
    - {id: "CR", limit: 2}

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
  roles:
    general_public:
      provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0}
    specific_character:
      provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0}
    plaintiff:
      provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0}
    defendant:
      provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0}
    procurator:
      provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0}
  summary:
    provider: openai_compatible
    profile: openai
    name: gpt-4o-2024-08-06
    generation_parameters:
      temperature: 0
      max_tokens: 4096
  evaluators:
    - provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters: {temperature: 0}

generation:
  concurrency: 4
  timeout_sec: 600
  model_request_timeout_sec: 60
  max_conversation_turn: 12
```

### 2.3 Execution Commands

```bash
# Preflight check
python -m lexverse run j1bench --config configs/benchmarks/j1bench.local.yaml --dry-run

# Run multi-agent simulation and evaluation
python -m lexverse run j1bench --config configs/benchmarks/j1bench.local.yaml --output-root runs/j1bench-experiment

# Resume, re-evaluate, or rebuild summary tables
python -m lexverse run --resume runs/j1bench-experiment
python -m lexverse evaluate --run-dir runs/j1bench-experiment
python -m lexverse collect --run-dir runs/j1bench-experiment
```

---

## 3. Upstream Adaptation & Rationale

LexVerse runs upstream `src/run.py` and `src/Eval/bench/*/*.py` in isolated per-Trial subprocesses and applies 8 targeted patches (`lexverse/benchmarks/j1bench/patches/001..008`) to resolve hardcoded paths, environment bindings, and bug fixes in the public repository (`commit 61a6ea55`):

1. **Relocatable Asset & Case Database Paths (`001-relocatable-paths.patch` & `003-eval-relocatable-paths.patch`)**:
   - **Context**: Upstream agent and evaluator scripts hardcode `/root/J1Bench/src/agents/profiles.json` and `/root/J1Bench/src/data/case/J1-Eval_*.jsonl`.
   - **Adaptation**: Resolves `profiles.json` via `J1BENCH_ROOT` and dataset files via `J1BENCH_CASE_DATABASE` so runs and evaluations execute portably from any working directory.
2. **Configurable Model Names, Optional Engines & Scoring Parameters (`002-honor-model-name.patch`, `004-utils-honor-env-vars.patch`, `007-optional-engines.patch`, `008-role-connections.patch`)**:
   - **Context**: Upstream `src/engine/gpt_4o_1120.py` ignores `openai_model_name` in `get_response()`, `src/engine/__init__.py` eagerly imports optional local engine dependencies (`vllm`, `dashscope`, `together`), and `src/utils/utils_func.py` hardcodes `gpt-4o-2024-08-06` for both topic state tracking (`KQ`/`LC`) and evaluation scoring.
   - **Adaptation**: Passes role-specific `openai_model_name`, `generation_parameters`, and API keys through CLI/environment bindings, separates `J1BENCH_SUMMARY_MODEL` (`models.summary`) from `J1BENCH_SCORING_MODEL` (`models.evaluators[0]`), and makes non-OpenAI engine imports lazy.
3. **Evaluator Zero-Score Aggregation & Mixed-Party Profile Formatting (`005-cr-zero-score-aggregation.patch` & `006-ci-mixed-parties.patch`)**:
   - **Context**: In `src/Eval/bench/CR/CR.py`, when `data["PFS"] == 0` (the judge failed to output a valid judgment), `get_final_score` threw a `TypeError` by subscripting `data["PFS"]["STA"]` on an integer and left `rea`/`law` unbound; in `src/Eval/bench/DD/DD.py`, unfinished cases used inconsistent keys (`RES_score`, `DEF_score`, `evi_score`, `FOR`) relative to `get_final_score`; and in `src/agents/plaintiff_CI.py` and `defendant_CI.py`, mixed natural-person vs. company party profiles used mismatched placeholder keys (`{plaintiff_person_name}` vs. `{plaintiff_name}`).
   - **Adaptation**: Guards `data["PFS"] != 0` and initializes `rea = 0` / `law = 0` in `CR.py`, aligns unfinished-case metric keys in `DD.py`, and fixes the mixed-party profile placeholders in `plaintiff_CI.py` and `defendant_CI.py`.

---

## 4. Reference Results (`lexverse run`) & Implementation Notes

The tables below compare the full **508-case (`KQ: 98, LC: 62, CD: 93, DD: 93, CI: 93, CR: 69`)** evaluation of the open-source model **`Qwen/Qwen2.5-7B-Instruct`** (`Qwen2.5-it. 7B`, run locally via `vllm` on an NVIDIA A100 GPU) against the official reference numbers in Table 3 (main results) and Tables 6–9 (sub-metric breakdowns) of the J1-EVAL paper ([arXiv:2507.04037](https://arxiv.org/abs/2507.04037), all scores in `%`):

### 4.1 Main Results Across All 6 Scenarios — Paper Table 3

> **Note**: In `CR`, **`III-4` (`(SEN + FINE) / 2`)** is the normalized logarithmic penalty deviation, where **lower values (`↓`) indicate better performance** (`0%` = exact match, `100%` = maximum deviation or failed judgment).

| Model | Source | `KQ` `I-1` (`BIN`) | `KQ` `I-2` (`NBIN`) | `LC` `I-1` (`BIN`) | `LC` `I-2` (`NBIN`) | `CD` `II-1` (`FOR`) | `CD` `II-2` (`DOC`) | `DD` `II-1` (`FOR`) | `DD` `II-2` (`DOC`) | `CI` `III-1` (`PFS`) | `CI` `III-2` (`JUD`) | `CI` `III-5` (`REA`) | `CI` `III-6` (`LAW`) | `CR` `III-1` (`PFS`) | `CR` `III-3` (`CRI`) | `CR` `III-4` (`↓`) | `CR` `III-5` (`REA`) | `CR` `III-6` (`LAW`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *GPT-4o* | Paper Table 3 | 69.7 | 62.7 | 55.9 | 45.1 | 58.9 | 89.9 | 45.4 | 84.8 | 87.0 | 28.5 | 50.2 | 13.9 | 49.3 | 60.9 | 38.5 | 42.8 | 27.0 |
| *Claude-3.7* | Paper Table 3 | 68.4 | 60.7 | 52.4 | 49.6 | 68.8 | 89.2 | 38.7 | 68.8 | 65.8 | 21.2 | 40.2 | 17.5 | 56.9 | 88.4 | 15.7 | 72.8 | 41.1 |
| *DeepSeek-V3-671B* | Paper Table 3 | 67.1 | 60.7 | 54.5 | 49.7 | 84.7 | 63.6 | 57.2 | 76.3 | 24.0 | 12.4 | 19.5 | 7.6 | 42.4 | 76.8 | 33.9 | 57.2 | 32.3 |
| *Qwen3-32B* | Paper Table 3 | 69.6 | 66.2 | 58.3 | 47.2 | 91.2 | 84.4 | 46.7 | 63.5 | 52.7 | 17.0 | 34.1 | 15.4 | 26.8 | 49.3 | 58.3 | 40.0 | 22.9 |
| *Qwen3-8B* | Paper Table 3 | 66.2 | 57.1 | 55.0 | 36.9 | 74.2 | 72.7 | 17.6 | 22.8 | 9.9 | 5.8 | 14.2 | 7.2 | 16.3 | 26.1 | 67.4 | 19.4 | 16.7 |
| **`Qwen2.5-it. 7B`** | **Paper Table 3** | **67.2** | **54.9** | **52.6** | **31.3** | **89.5** | **81.4** | **47.1** | **59.7** | **15.6** | **7.3** | **14.1** | **7.5** | **26.4** | **49.3** | **54.5** | **37.5** | **25.4** |
| | **Upstream Repo (`qwen2.5-7B_final.json`)** | **67.2** | **54.9** | **52.6** | **31.3** | **89.5** | **81.4** | **47.1** | **59.7** | **21.2** | **11.6** | **21.0** | **8.5** | **16.0** | **66.7** | **38.2** | **48.7** | **33.4** |
| | **`lexverse run j1bench` (Full 508)** | **66.5** | **40.5** | **53.1** | **36.3** | **92.5** | **85.6** | **44.5** | **63.6** | **30.0** | **22.3** | **28.7** | **7.9** | **16.4** | **82.6** | **21.0** | **58.4** | **35.6** |

### 4.2 Document Drafting (`CD`, `DD`) & Penalty Deviation (`CR`) Sub-Metrics — Paper Tables 6, 7 & 9

| Model | Source | `CD` `PLA` | `CD` `DEF` | `CD` `CLA` | `CD` `EVI` | `CD` `FAC` (`F&R`) | `DD` `RES` | `DD` `DEF` | `DD` `EVI` | `CR` `SEN` (`↓`) | `CR` `FINE` (`↓`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *GPT-4o* | Paper Tables 6/7/9 | 92.7 | 96.3 | 97.3 | 77.7 | 85.3 | 92.5 | 86.6 | 75.3 | 36.4 | 40.6 |
| *Qwen3-32B* | Paper Tables 6/7/9 | 74.0 | 94.3 | 98.5 | 76.0 | 79.4 | 73.4 | 79.3 | 37.7 | 59.5 | 57.1 |
| **`Qwen2.5-it. 7B`** | **Paper Tables 6/7/9** | **84.2** | **94.1** | **91.8** | **52.3** | **84.7** | **69.3** | **73.6** | **36.3** | **56.0** | **53.0** |
| | **Upstream Repo (`qwen2.5-7B_final.json`)** | **84.2** | **94.1** | **91.8** | **52.3** | **84.7** | **69.2** | **73.5** | **36.3** | **40.3** | **36.1** |
| | **`lexverse run j1bench` (Full 508)** | **88.1** | **93.4** | **95.8** | **64.2** | **86.6** | **77.9** | **72.0** | **40.8** | **20.5** | **21.4** |

### Implementation & Protocol Notes

- **Why `CI` and `CR` Scores in `src/Eval/final_result/` and `lexverse run` Are Higher Than Paper Table 3**:
  1. **Rule-Based Evaluator in the Public Repository vs. LLM-Checklist Evaluator in the Paper**: The upstream GitHub repository (`commit 61a6ea55`) includes the authors' own pre-computed evaluation outputs under `src/Eval/final_result/{KQ,LC,CD,DD,CI,CR}/qwen2.5-7B_final.json`. While `KQ`, `LC`, `CD`, and `DD` in `qwen2.5-7B_final.json` are identical to Paper Table 3 (`67.2, 54.9, 52.6, 31.3, 89.5, 81.4, 47.1, 59.7`), the authors' own `qwen2.5-7B_final.json` for `CI` (`COM=21.2, VER=11.6, REA=21.0, LAW=8.5`) and `CR` (`COM=16.0, CRI=66.7, VER=38.2, REA=48.7, LAW=33.4`) already match our `lexverse run` numbers (`CR` `PFS=16.4, LAW=35.6`) much more closely than Paper Table 3. This is because the paper (Appendix D.3) used an LLM-as-Judge procedural checklist (`UNI = 40.9%` on `CI` and `69.6%` on `CR` in `qwen2.5-7B_final.json`), whereas the open-sourced `src/Eval/bench/CI/CI.py` and `CR/CR.py` scripts replace the LLM checklist with a rule-based substring matcher (`get_PFS`) that sets `UNI = 1` whenever the judge speaks at least once without merging all stage headers into a single turn.
  2. **Mixed-Party Profile Fix (`006-ci-mixed-parties.patch`)**: In the unpatched upstream `src/agents/plaintiff_CI.py` and `defendant_CI.py`, 30 of the 93 `CI` cases (where one or both parties are companies rather than natural persons) had mismatched placeholder keys (`{plaintiff_person_name}` vs. `{plaintiff_name}`), leaving party identity fields blank in the agent system prompts. With `006-ci-mixed-parties.patch` fixing those 30 cases, `CI` trial completion and judgment scores improve from `21.2 / 11.6 / 21.0` to `30.0 / 22.3 / 28.7`.
- **Single-Type Cases & Law-Topic Matching in `KQ` and `LC`**: In upstream `src/Eval/bench/KQ/KQ.py` and `LC/LC.py`, cases whose `topic_list` contains only non-binary topics (such as `KQ-1`) or only binary topics (such as `LC-1`) record `0` for the absent question type (`total_binary_scores == 0` or `total_nonbinary_scores == 0`), which is averaged into `get_final_score()`. In addition, the open-sourced `KQ.py` checks `if type == 'ended_long_law':` whereas `J1-Eval_KQ.jsonl` labels statutory questions as `'open-ended_law'`, causing statutory sub-questions in `KQ` to be skipped by the open-source evaluator (explaining `KQ` `I-2 = 40.5%` vs. `54.9%`, while `KQ` `I-1 = 66.5%` vs. `67.2%` and `LC` `I-1 = 53.1%` vs. `52.6%` match within `0.7` points).
