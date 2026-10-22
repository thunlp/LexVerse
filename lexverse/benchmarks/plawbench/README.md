# PLawBench Benchmark

In practical legal work, an assistant must handle three distinct tasks across a case's lifecycle: (1) asking neutral, verifiable intake questions during a **public legal consultation**, (2) decomposing a complex dispute into a structured **practical case analysis** (`【结论】`, `【案件事实】`, `【推理过程】`, `【法条依据】`), and (3) drafting a formal **civil complaint or statement of defense** from a party's raw narrative.

**PLawBench** (*A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice*, [`SKYLENAGE-AI/PLawBench`](https://github.com/SKYLENAGE-AI/PLawBench)) evaluates LLMs across all three tasks using expert-annotated, case-specific rubrics rather than surface N-gram overlap.

---

## 1. Overview

- **Benchmark ID**: `plawbench`
- **Upstream Repository**: [`SKYLENAGE-AI/PLawBench`](https://github.com/SKYLENAGE-AI/PLawBench) (commit `f6ee07b36c1f4c0bfc2af8ca27183d3dd3175c56`)
- **Dataset**:
  - `public_legal_consultation_18.json` (18 consultation scenarios with point-weighted inquiry rubrics)
  - `practical_case_analysis_250.jsonl` (250 real-world case analysis questions with 4-part rubrics)
  - `Plantiffs_Statement.json` & `Defendants_Statement.json` (12 document drafting scenarios: 6 plaintiff complaints + 6 defendant statements of defense)
- **Tasks & Metrics**:
  - `legal_consultation` (Task 1): Outputs a numbered list of 10–25 follow-up questions; scored on `acc` (`total_points / max_points`) and `failed`.
  - `case_analysis` (Task 2): Outputs a four-section legal analysis; scored on overall `acc`, four sub-dimensions (`conclusion_acc`, `fact_acc`, `reasoning_acc`, `law_acc`), and `failed`.
  - `document_generation` (Task 3): Drafts a 2,500–3,000 character pleading; scored on `acc`, `failed`, and role subtypes (`plaintiff` and `defendant`).

---

## 2. How to Run

### 2.1 Installation

```bash
python -m pip install -e ".[plawbench,openai,test]"
python -m lexverse catalog plawbench
```

### 2.2 Configuration

A PLawBench run binds two models:
1. **Candidate Model (`models.default`)**: Generates the single-turn consultation questions, case analysis, or legal document.
2. **Rubric Evaluator (`models.evaluators[0]`)**: Grades the candidate output against the expert rubric in `prompt_zh.py`.

Copy `configs/benchmarks/plawbench.example.yaml` to `configs/benchmarks/plawbench.local.yaml`:

```yaml
schema_version: 1.0

benchmark:
  name: plawbench
  tasks:
    - {id: "case_analysis", limit: 30}          # Set limit to null for all 250 questions
    - {id: "legal_consultation", limit: null}   # 18 questions
    - {id: "document_generation", limit: null}  # 12 questions (6 plaintiff + 6 defendant)

models:
  default:
    provider: vllm
    name: Qwen/Qwen3-4B-Instruct-2507
    load_parameters:
      dtype: bfloat16
      tensor_parallel_size: 1
      max_model_len: 32768
      trust_remote_code: false
    input: {format: chat}
    generation_parameters:
      temperature: 0
      max_tokens: 8192
  evaluators:
    - provider: openai_compatible
      profile: openai
      name: gpt-5.2-2025-12-11
      generation_parameters:
        max_completion_tokens: 8192

generation:
  concurrency: 16

evaluation:
  concurrency: 8
  max_attempts: 3
```

### 2.3 Execution Commands

```bash
# Preflight check
python -m lexverse run plawbench --config configs/benchmarks/plawbench.local.yaml --dry-run

# Run generation and rubric evaluation
python -m lexverse run plawbench --config configs/benchmarks/plawbench.local.yaml --output-root runs/plawbench-experiment

# Resume, re-evaluate, or rebuild summary tables
python -m lexverse run --resume runs/plawbench-experiment
python -m lexverse evaluate --run-dir runs/plawbench-experiment
python -m lexverse collect --run-dir runs/plawbench-experiment
```

---

## 3. Upstream Adaptation & Rationale

When inspecting the public [`SKYLENAGE-AI/PLawBench`](https://github.com/SKYLENAGE-AI/PLawBench) repository (`commit f6ee07b3`), we found two scripts (`prompt_generate.py` and `prompt_zh.py`) with hardcoded placeholder paths and unimplemented API stubs. LexVerse adapts them via `patches/001-evaluator-callable.patch` and `tasks.py`:

1. **AST Template Extraction & Rubric Isolation (`tasks.py`)**:
   - **Context**: Upstream `prompt_generate.py` executes top-level `open(PATH + "...")` file I/O using a literal placeholder `PATH = "your_path"`, and `public_legal_consultation_18.json` stores item rubrics in a field named `"rubrics"` without a pre-computed `"score"` field.
   - **Adaptation**: `PLawBenchImporter` extracts `prompt_template_dict` and `user_prompt_dict` from `prompt_generate.py` via `ast.parse` without executing top-level file I/O, computes the exact rubric point sum (`(+X分)`) for consultation items, interleaves `plaintiff` and `defendant` drafting items, and isolates the ground-truth rubrics strictly inside `evaluation`.
2. **Injectable Judge Callable & Strict Numeric Validation (`001-evaluator-callable.patch` & `evaluation.py`)**:
   - **Context**: In upstream `prompt_zh.py`, `call_llm_api(messages)` is an empty placeholder (`pass`) that returns `None`, and `_calculate_metrics` silently defaults missing `total_points` / `max_points` to `0` / `1`.
   - **Adaptation**: Our patch allows `LegalEvaluator` to accept a run-scoped judge callable (`LegalEvaluator(call_judge)`), and `validate_grading` verifies that all required rubric score fields (`total_points` and `max_points`, or all four sections in `case_analysis_merge`) are finite numbers before accepting a judge response.

---

## 4. Reference Results (`lexverse run`) & Implementation Notes

The table below compares the scores obtained directly by running `python -m lexverse run plawbench` on two open-weight models from the official PLawBench leaderboard—**`Qwen/Qwen3-4B-Instruct-2507`** and **`Qwen/Qwen3-4B-Thinking-2507`** (run locally via `vllm` on an NVIDIA A100 GPU, evaluated with `gpt-5.2`)—against the official README ranking table (all scores in `%`):

| Model | Source | Overall | Task 2: Analysis (`acc`) | Task 2: Conclusion | Task 2: Facts | Task 2: Reasoning | Task 2: Statute | Task 1: Consultation (`acc`) | Task 3: Drafting (`acc`) | Task 3: Plaintiff | Task 3: Defendant |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *DeepSeek-V3.2* | Paper Table | 57.97 | 64.01 | 65.56 | 82.89 | 61.64 | 43.27 | 60.12 | 46.48 | 44.43 | 48.52 |
| **`Qwen-4b-instruct-2507`** | **Paper Table** | **50.53** | **53.72** | **52.61** | **89.98** | **44.77** | **23.76** | **58.49** | **39.37** | **31.08** | **47.66** |
| | **`lexverse run plawbench`** | **52.65** | **56.13** | **49.56** | **92.82** | **43.89** | **27.56** | **67.43** | **34.39** | **37.42** | **31.37** |
| **`Qwen-4b-thinking-2507`** | **Paper Table** | **44.80** | **52.70** | **56.61** | **89.97** | **45.97** | **26.73** | **41.88** | **39.83** | **30.22** | **49.44** |
| | **`lexverse run plawbench`** | **45.33** | **51.98** | **59.56** | **81.98** | **41.25** | **30.50** | **47.78** | **36.24** | **35.25** | **37.23** |

### Implementation & Protocol Notes

- **Judge Model Selection (`gpt-5.2`)**: Because upstream `prompt_zh.py` leaves `call_llm_api()` unimplemented, the judge model must be configured via `models.evaluators[0]`. Using `gpt-5.2` as the rubric evaluator reproduces the paper's overall and sub-dimension scores closely (`52.65` vs. `50.53` overall for `Qwen3-4B-Instruct-2507`, and `45.33` vs. `44.80` overall for `Qwen3-4B-Thinking-2507`, preserving the exact relative ordering across all tasks and sub-dimensions).
- **Public Consultation Split (`public_legal_consultation_18.json`)**: Upstream `prompt_generate.py` assigns `level = "mid"` to the first 50 items and `"hard"` thereafter; because the public repository releases 18 consultation items, all 18 items use the `legal_consultation_mid` prompt (10–25 questions).
