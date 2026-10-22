# J1-EVAL (J1Bench) Benchmark

In realistic legal practice, an AI assistant rarely operates in a single-turn vacuum. A junior associate answering public legal inquiries must clarify ambiguous questions across turns; a litigator drafting a civil complaint or statement of defense must interview the client to uncover missing facts and evidence; and a presiding judge in a civil or criminal hearing must manage multi-party courtroom procedures before rendering a reasoned judgment.

**J1-EVAL (`J1Bench`)** (*Dynamic Evaluation of Large Language Models on Chinese Legal Multi-Role Scenarios*, [arXiv:2507.04037](https://arxiv.org/abs/2507.04037)) evaluates LLMs across **six dynamic multi-agent legal scenarios** (`KQ`, `LC`, `CD`, `DD`, `CI`, and `CR`, totaling **508 cases**) by pairing the candidate model (`trainee`, `lawyer`, or `judge`) with interactive environment agents (`general_public`, `specific_character`, `plaintiff`, `defendant`, and `procurator`) and rubric-based evaluators.

---

## 1. Overview

- **Benchmark ID**: `j1bench`
- **Upstream Repository**: [`CimoInkPool/J1-EVAL`](https://github.com/CimoInkPool/J1-EVAL) (commit `61a6ea5515e11229c13353e73b0629e39dcc6076`)
- **Dataset**: [`CimoInkPool/J1-Eval_Dataset`](https://huggingface.co/datasets/CimoInkPool/J1-Eval_Dataset) (revision `aa0ddfb6896394c48a3ddcc3e4d055b70b79d4b6`, gated HuggingFace dataset; **508 cases** across 6 scenarios)
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

LexVerse runs upstream `src/run.py` and `src/Eval/eval.py` in isolated per-Trial subprocesses and applies 9 targeted patches (`lexverse/benchmarks/j1bench/patches/001..009`) to resolve defects in the public repository (`commit 61a6ea55`):

1. **Deterministic Sequential Execution (`002-deterministic-run-order.patch`)**:
   - **Context**: Upstream `remove_processed_cases()` calls `random.shuffle(self.case_pair)` in all 6 scenarios.
   - **Adaptation**: Removes `random.shuffle` so case execution order is deterministic and reproducible.
2. **Broken Relative Imports, Missing Syntax Commas & Hardcoded `/root/J1Bench` Paths (`001`, `003`, `004`, `005`, `007`, `008`)**:
   - **Context**: Upstream `src/Eval/eval.py` imports `from bench.KQ import KQ` (a module rather than the `KQ` class), `src/agents/general_public_LC.py:169` contains `style = ['roles']['general_public']` (missing `general_public_info`), `src/Eval/bench/DD/DD.py` references `self.intermediate_path` instead of `self.intermediate_eval`, and multiple agent/evaluator files hardcode `/root/J1Bench/...`.
   - **Adaptation**: Fixes the class imports and syntax errors, routes file lookups through `J1BENCH_ROOT`, and strips `<think>...</think>` blocks in `Eval/eval.py` without crashing when `</think>` is absent.
3. **Unconditional Dialogue Persistence & Courtroom Turn Routing (`009-role-connections.patch`)**:
   - **Context**:
     1. In upstream `src/scenario/KQ.py` and `LC.py`, `self.save_dialog_info(dialog_info)` was guarded by `if '无' in unquestioned_ls:`—meaning whenever a consultation reached `max_conversation_turn` before exhausting every topic in `topic_list`, upstream silently discarded the dialogue without writing `dialog_history.jsonl`.
     2. In upstream `src/scenario/CI.py`, the courtroom loop only dispatched a turn to `plaintiff` or `defendant` if the judge's utterance contained the exact substring `'对原告说'` or `'对被告说'`. When open-weight models (such as `Qwen2.5-7B-Instruct`) addressed parties by name (e.g., `"于英雪女士，请..."`) without that prefix, neither party replied and the judge spoke in a monologue until exhausting all turns without a final judgment (`<最终宣判>`).
   - **Adaptation**: Always persists `dialog_info` to `save_path` at the end of `KQ` and `LC` consultations, infers the addressed party in `CI.py` when the judge omits the literal `'对原告说'` / `'对被告说'` prefix, and prompts the judge for `<最终宣判>` if the turn budget ends before judgment delivery in `CI` and `CR`.

---

## 4. Reference Results (`lexverse run`) & Implementation Notes

The tables below report the scores obtained by running `python -m lexverse run j1bench` on the open-source model **`Qwen/Qwen2.5-7B-Instruct`** (run locally via `vllm` on an NVIDIA A100 GPU) across all 6 scenarios, alongside the reference numbers from Tables 2 and 3 of the J1-EVAL paper ([arXiv:2507.04037](https://arxiv.org/abs/2507.04037), all scores in `%`):

### 4.1 Knowledge QA (`KQ`), Legal Consultation (`LC`), Complaint Drafting (`CD`), and Defense Drafting (`DD`)

| Model | Source | `KQ` BIN | `KQ` NBIN | **`KQ` AVE** | `LC` BIN | `LC` NBIN | **`LC` AVE** | `CD` DOC | `CD` FOR | **`CD` AVE** | `DD` DOC | `DD` FOR | **`DD` AVE** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *DeepSeek-R1* | Paper Table 2 | 88.86 | 80.52 | 84.69 | 62.63 | 69.28 | 65.96 | 79.48 | 98.92 | 89.20 | 77.96 | 89.25 | 83.60 |
| *Qwen3-32B (Think)* | Paper Table 2 | 84.56 | 78.73 | 81.65 | 55.66 | 68.18 | 61.92 | 82.97 | 100.00 | 91.48 | 74.98 | 97.85 | 86.42 |
| *QwQ-32B* | Paper Table 2 | 83.24 | 75.59 | 79.42 | 58.61 | 67.53 | 63.07 | 80.50 | 97.85 | 89.18 | 80.62 | 96.77 | 88.70 |
| **`Qwen2.5-7B-Instruct`** | **Paper Table 2** | **70.82** | **62.13** | **66.48** | **57.16** | **62.03** | **59.60** | **78.00** | **93.55** | **85.77** | **71.93** | **75.27** | **73.60** |
| | **`lexverse run j1bench`** | **25.00** | **65.00** | **45.00** | **50.00** | **46.67** | **48.33** | **83.67** | **100.00** | **91.83** | **73.89** | **50.00** | **67.92** |

### 4.2 Civil Trial (`CI`) and Criminal Trial (`CR`)

| Model | Source | `CI` JUD | `CI` LAW | `CI` REA | **`CI` PFS** | `CR` CRI | `CR` SEN | `CR` FINE | `CR` LAW | `CR` REA | **`CR` PFS** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *DeepSeek-R1* | Paper Table 3 | 67.44 | 31.14 | 58.95 | 79.29 | 98.55 | 63.36 | 34.48 | 49.04 | 62.68 | 87.30 |
| *Qwen3-32B (Think)* | Paper Table 3 | 68.61 | 31.71 | 62.09 | 67.20 | 97.10 | 53.79 | 44.83 | 47.46 | 62.58 | 70.95 |
| **`Qwen2.5-7B-Instruct`** | **Paper Table 3** | **31.79** | **12.50** | **29.75** | **15.13** | **21.74** | **18.23** | **22.97** | **10.75** | **17.75** | **9.66** |
| | **`lexverse run j1bench`** | **0.00** | **50.00** | **15.00** | **23.08** | **100.00** | **50.00** | **50.00** | **7.14** | **75.00** | **12.50** |

### Implementation & Protocol Notes

- **Procedural Gate (`PFS.UNI`) in `CI` and `CR`**: As reported in Table 3 of the J1-EVAL paper, `Qwen2.5-7B-Instruct` achieves low procedural fidelity (`PFS = 15.13%` on `CI` and `9.66%` on `CR`) because 7B models frequently merge multiple trial stages into a single turn or omit the literal `'对原告说：'` / `'对被告说：'` speaker prefixes. With Patch `009`, `CI` and `CR` recover speaker routing and final judgment generation when the turn budget is reached.
- **Turn Budget vs. Topic Count in `KQ` and `LC`**: Each `KQ` and `LC` case contains 3–6 sub-questions (`BIN` and `NBIN`). When `generation.max_conversation_turn` is set to `12` (vs. `15` in full runs), any sub-question not yet reached when the dialogue ends is scored `0` by upstream `Eval/bench/{KQ,LC}`, which primarily affects binary sub-questions (`BIN`) positioned at the end of `topic_list`.
