# LexEval Benchmark

To evaluate legal large language models systematically across the full spectrum of legal cognition, **LexEval** (*A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models*, NeurIPS 2024 Datasets and Benchmarks Track, [arXiv:2409.20288](https://arxiv.org/abs/2409.20288)) introduces the **LexAbility Taxonomy**, organizing **23 legal tasks (14,150 questions)** into six cognitive ability levels: **Memorization (`1_1`–`1_3`)**, **Understanding (`2_1`–`2_5`)**, **Logic Inference (`3_1`–`3_6`)**, **Discrimination (`4_1`–`4_2`)**, **Generation (`5_1`–`5_4`)**, and **Ethics (`6_1`–`6_3`)**.

---

## 1. Overview

- **Benchmark ID**: `lexeval`
- **Upstream Repository**: [`CSHaitao/LexEval`](https://github.com/CSHaitao/LexEval) (commit `044c695f62894deef41ad9a30797e1da0404945c`)
- **Dataset**: [`CSHaitao/LexEval`](https://huggingface.co/datasets/CSHaitao/LexEval) (**23 tasks, 14,150 samples in total**)
- **Task Taxonomy & Metrics**:
  - **1. Memorization (`1_1`–`1_3`, Accuracy)**: `1_1` (*Concept Recitation*, 500), `1_2` (*Law Article Recitation*, 1,000), `1_3` (*Legal Knowledge QA*, 300)
  - **2. Understanding (`2_1`–`2_5`, Accuracy)**: `2_1` (*Document Proofreading*, 500), `2_2` (*Dispute Focus Identification*, 300), `2_3` (*Event Detection*, 100), `2_4` (*Trigger Extraction*, 500), `2_5` (*Reading Comprehension*, 500)
  - **3. Logic Inference (`3_1`–`3_6`, Accuracy)**: `3_1` (*Fact-based Article Prediction*, 1,000), `3_2` (*Scene-based Article Prediction*, 1,000), `3_3` (*Charge Prediction*, 1,000), `3_4` (*Prison Term Prediction*, 500), `3_5` (*Damages Calculation*, 400), `3_6` (*Case Analysis*, 500)
  - **4. Discrimination (`4_1`–`4_2`, Accuracy)**: `4_1` (*Case Element Matching*, 500), `4_2` (*Similar Case Matching*, 300)
  - **5. Generation (`5_1`–`5_4`, ROUGE-L)**: `5_1` (*Opinion Summarization*, 1,000), `5_2` (*Court View Generation*, 1,000), `5_3` (*Consultation*, 250), `5_4` (*Contract Generation*, 500)
  - **6. Ethics (`6_1`–`6_3`, Accuracy)**: `6_1` (*Legal Ethics*, 1,000), `6_2` (*Privacy Protection*, 1,000), `6_3` (*Fairness & Bias*, 500)

---

## 2. How to Run

### 2.1 Installation

```bash
python -m pip install -e ".[lexeval,openai,test]"
python -m lexverse catalog lexeval
```

### 2.2 Configuration

Copy `configs/benchmarks/lexeval.example.yaml` to `configs/benchmarks/lexeval.local.yaml`:

```yaml
schema_version: 1.0

benchmark:
  name: lexeval
  tasks:
    - {id: "1_1", limit: null}
    - {id: "1_2", limit: null}
    - {id: "1_3", limit: null}
    - {id: "2_1", limit: null}
    - {id: "2_2", limit: null}
    - {id: "2_3", limit: null}
    - {id: "2_4", limit: null}
    - {id: "2_5", limit: null}
    - {id: "3_1", limit: null}
    - {id: "3_2", limit: null}
    - {id: "3_3", limit: null}
    - {id: "3_4", limit: null}
    - {id: "3_5", limit: null}
    - {id: "3_6", limit: null}
    - {id: "4_1", limit: null}
    - {id: "4_2", limit: null}
    - {id: "5_1", limit: null}
    - {id: "5_2", limit: null}
    - {id: "5_3", limit: null}
    - {id: "5_4", limit: null}
    - {id: "6_1", limit: null}
    - {id: "6_2", limit: null}
    - {id: "6_3", limit: null}

models:
  default:
    provider: vllm
    name: Qwen/Qwen-7B-Chat
    load_parameters:
      dtype: float16
      tensor_parallel_size: 1
      max_model_len: 8192
      trust_remote_code: true
    input: {format: chat}
    generation_parameters:
      temperature: 0
      max_tokens: 512

generation:
  concurrency: 32
```

### 2.3 Execution Commands

```bash
# Preflight check
python -m lexverse run lexeval --config configs/benchmarks/lexeval.local.yaml --dry-run

# Run full 14,150-sample evaluation
python -m lexverse run lexeval --config configs/benchmarks/lexeval.local.yaml --output-root runs/lexeval-qwen7b-full

# Resume, re-evaluate, or rebuild summary tables
python -m lexverse run --resume runs/lexeval-qwen7b-full
python -m lexverse evaluate --run-dir runs/lexeval-qwen7b-full
python -m lexverse collect --run-dir runs/lexeval-qwen7b-full
```

---

## 3. Upstream Adaptation & Rationale

LexVerse executes the native `code/evaluation/evaluate.py` evaluator and `code/generation/model_gen.py` loader from `CSHaitao/LexEval` via three targeted patches (`patches/001..003`):

1. **Lazy Heavy Imports in `evaluate.py` and `model_gen.py` (`001-defer-heavy-imports.patch` & `002-defer-vllm-import.patch`)**:
   - **Context**: Upstream `code/evaluation/evaluate.py` and `code/generation/model_gen.py` eagerly import `torch`, `transformers`, `bert_score`, `pandas`, `rouge`, and `vllm` at module import time even when only running multiple-choice accuracy or API-based generation.
   - **Adaptation**: Moves optional heavy imports inside the methods that actually use them (`get_rouge_l`, `get_bert_score`, `get_bart_score`, `init_model_vllm`).
2. **Blank-Line Tolerance in JSONL Task Files (`003-skip-blank-lines.patch`)**:
   - **Context**: Upstream `code/generation/model_gen.py` calls `json.loads(line)` on every line in `data/*.json`, which raises `JSONDecodeError` on trailing blank lines.
   - **Adaptation**: Skips empty lines (`if not line.strip(): continue`) when loading task records.

---

## 4. Reference Results (`lexverse run`) & Implementation Notes

The tables below report the full **14,150-sample (23 tasks)** zero-shot evaluation of the open-source model **`Qwen/Qwen-7B-Chat`** (run locally via `vllm` on an NVIDIA A100 GPU), alongside the reference numbers from Table 3 of the LexEval paper ([arXiv:2409.20288](https://arxiv.org/abs/2409.20288), all scores in `%`):

### 4.1 Category Summary (`Qwen/Qwen-7B-Chat`, Zero-Shot, 14,150 Samples)

| Model | Source | Memorization (`1_1–1_3`) | Understanding (`2_1–2_5`) | Logic Inference (`3_1–3_6`) | Discrimination (`4_1–4_2`) | Generation (`5_1–5_4`) | Ethics (`6_1–6_3`) | **Overall Average (23 Tasks)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *GPT-4* | Paper Table 3 | 31.13 | 75.56 | 56.55 | 25.45 | 28.08 | 49.40 | 48.77 |
| *Qwen-14B-Chat* | Paper Table 3 | 29.73 | 81.20 | 61.97 | 42.10 | 26.83 | 49.60 | 52.49 |
| **`Qwen-7B-Chat`** | **Paper Table 3** | **23.37** | **73.86** | **53.63** | **24.80** | **25.70** | **39.27** | **45.17** *(44.80)* |
| | **`lexverse run lexeval` (Full 14,150)** | **23.18** | **74.19** | **53.74** | **24.50** | **27.74** | **39.37** | **45.26** |

### 4.2 Per-Task Breakdown (`Qwen/Qwen-7B-Chat`, All 23 Tasks / 14,150 Samples)

| Task ID | Task Name | Samples | Metric | Upstream Reference (`Qwen_7B_chat`) | **`lexverse run lexeval` (Full 14,150)** | Diff |
| :---: | :--- | ---: | :---: | ---: | ---: | ---: |
| `1_1` | Concept Recitation | 500 | Accuracy | 21.80 | **22.40** | +0.60 |
| `1_2` | Law Article Recitation | 1,000 | Accuracy | 40.30 | **38.80** | -1.50 |
| `1_3` | Legal Knowledge QA | 300 | Accuracy | 8.00 | **8.33** | +0.33 |
| `2_1` | Document Proofreading | 500 | Accuracy | 80.20 | **79.80** | -0.40 |
| `2_2` | Dispute Focus Identification | 300 | Accuracy | 42.33 | **43.33** | +1.00 |
| `2_3` | Event Detection | 100 | Accuracy | 87.00 | **88.00** | +1.00 |
| `2_4` | Trigger Extraction | 500 | Accuracy | 67.60 | **67.20** | -0.40 |
| `2_5` | Reading Comprehension | 500 | Accuracy | 92.20 | **92.60** | +0.40 |
| `3_1` | Fact-based Article Prediction | 1,000 | Accuracy | 78.60 | **79.10** | +0.50 |
| `3_2` | Scene-based Article Prediction | 1,000 | Accuracy | 84.30 | **84.10** | -0.20 |
| `3_3` | Charge Prediction | 1,000 | Accuracy | 53.60 | **53.20** | -0.40 |
| `3_4` | Prison Term Prediction | 500 | Accuracy | 23.80 | **24.60** | +0.80 |
| `3_5` | Damages Calculation | 400 | Accuracy | 36.25 | **36.25** | 0.00 |
| `3_6` | Case Analysis | 500 | Accuracy | 45.20 | **45.20** | 0.00 |
| `4_1` | Case Element Matching | 500 | Accuracy | 19.60 | **20.00** | +0.40 |
| `4_2` | Similar Case Matching | 300 | Accuracy | 30.00 | **29.00** | -1.00 |
| `5_1` | Opinion Summarization | 1,000 | ROUGE-L | 32.02 | **34.09** | +2.07 |
| `5_2` | Court View Generation | 1,000 | ROUGE-L | 17.64 | **18.66** | +1.02 |
| `5_3` | Consultation | 250 | ROUGE-L | 31.69 | **35.48** | +3.79 |
| `5_4` | Contract Generation | 500 | ROUGE-L | 21.46 | **22.74** | +1.28 |
| `6_1` | Legal Ethics | 1,000 | Accuracy | 22.10 | **22.10** | 0.00 |
| `6_2` | Privacy Protection | 1,000 | Accuracy | 39.70 | **39.20** | -0.50 |
| `6_3` | Fairness & Bias | 500 | Accuracy | 56.00 | **56.80** | +0.80 |
| **All** | **23-Task Macro Average (14,150 Samples)** | **14,150** | **Mean** | **45.17** | **45.26** | **+0.09** |

### Implementation & Protocol Notes

- **Full-Dataset Reproduction (`45.26` vs. `45.17`, `+0.09` Overall Difference)**: Across all 14,150 questions in the 23 LexEval tasks, `python -m lexverse run lexeval` with `Qwen/Qwen-7B-Chat` matches the upstream `Qwen_7B_chat` reference outputs within `0.09–0.33` points on all five multiple-choice categories (`Memorization: 23.18 vs. 23.37`, `Understanding: 74.19 vs. 73.86`, `Logic Inference: 53.74 vs. 53.63`, `Discrimination: 24.50 vs. 24.80`, `Ethics: 39.37 vs. 39.27`) and achieves `+0.09` overall macro-average parity (`45.26` vs. `45.17`).
- **Context Window Handling (`max_model_len: 8192`)**: In upstream `code/generation/Qwen_gen.py`, prompts exceeding `max_Length = 3500` characters are middle-truncated (`half = int(max_Length / 2)`). In LexVerse, `LocalBackend` applies token-level head+tail middle truncation only when a prompt plus `max_tokens` exceeds the 8,192-token model context window (`11` ultra-long case documents in `3_2`, `4_1`, and `5_4`), preserving full case context on `99.92%` of the benchmark.
