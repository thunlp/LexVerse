# LeCaRDv2 Benchmark

Finding prior cases that are legally analogous to a pending criminal case (*Legal Case Retrieval*, LCR) is essential to consistent sentencing in judicial practice. In Chinese criminal law, two cases are legally relevant not merely when they share surface keywords, but when they align across **characterization** (constitutive elements of the crime and key factual circumstances), **penalty**, and **procedure**.

**LeCaRDv2** (*A Large-Scale Chinese Legal Case Retrieval Dataset*, [SIGIR 2024](https://arxiv.org/abs/2310.17609)) benchmarks this task with **800 query cases** (640 train / 160 test, covering 50 criminal charges) and a corpus of **55,192 candidate criminal judgments**.

---

## 1. Overview

- **Benchmark ID**: `lecardv2`
- **Upstream Repository**: [`THUIR/LeCaRDv2`](https://github.com/THUIR/LeCaRDv2) (commit `3326485a6c41f8aeda15502fea24e658f8fab746`)
- **Dataset**:
  - **Queries**: 800 query cases (`query/query.json`; 640 in `train_query.json`, 160 in `test_query.json`)
  - **Candidate Corpus**: 55,192 criminal judgments (`candidate.tar`, SHA256: `6b80d09fc09b4af688ea36697252cf1bb1af0b3281422f5ae78a0899dfb010b6`)
  - **Relevance Labels**: `label/relevence.trec` (multi-level grades `0..3` for ~30 pooled docs per query) and `label/relevence_gold.trec` (binary positives with `grade >= 2`)
- **Task**: `case_retrieval` (outputs a ranked list of candidate PIDs in standard TREC format)
- **Metrics**:
  - **Full-Corpus Recall**: `recall_at_100`, `recall_at_200`, `recall_at_500`, `recall_at_1000`
  - **Candidate-Pool Ranking**: `map`, `mrr`, `p_at_1`, `p_at_3`, `ndcg_at_3`, `ndcg_at_5`, `ndcg_at_10`, `ndcg_at_20`, `ndcg_at_30`

---

## 2. Official benchmark and optional methods

### Official LeCaRDv2 setting

The [official repository](https://github.com/THUIR/LeCaRDv2#experiment) releases the dataset and relevance labels. Its original experiments retrieve directly from 55,192 candidate cases using BM25, QLD or neural retrievers, without requiring LLM query rewriting. Zero-shot evaluation uses all 800 queries; the fine-tuning setting uses 640 training and 160 test queries. The reported metrics are Recall@100/200/500/1000. The authors identify Pyserini as their traditional retrieval toolkit and `luyug/Dense` as the neural retrieval reference.

The shipped [Job YAML](../../../configs/jobs/lecardv2.yaml) selects `bm25_retrieval`, `full_corpus` and `split: all`. It runs **two queries as a smoke test**; set the task's `limit: null` for all queries. BM25 uses the original case facts without an LLM or prompt template. The required `models.default` entry is unused in this mode; no model API credentials are needed.

**LexVerse supplies its own BM25 implementation and scoring adapter; this is not the official Pyserini runner or a claim of exact score reproduction.** See implementation details below.

### Optional LexVerse modes

| Mode | Behavior |
| --- | --- |
| `bm25_retrieval` | Direct lexical retrieval (shipped default) |
| `encoder_retrieval` | Direct retrieval using a configured encoder |
| `ranked_submission` | Score an external TREC run specified by `submission_path` |
| `llm_query_rewrite` | Append LLM-generated text to the original facts, then retrieve with BM25 |
| `keller_subfact_retrieval` | KELLER-inspired sub-fact retrieval using a local encoder |

The following are **LexVerse template keys referencing separate papers**, not LeCaRDv2 requirements or official template names:

| `prompt_template` | Referenced source | Local template behavior |
| --- | --- | --- |
| `when_rules_learn_human1` | [When Rules Learn](https://arxiv.org/abs/2606.17220), Human-1 baseline | Expand synonyms and legal terminology |
| `jurex_4e_fet` | [JUREX-4E](https://arxiv.org/abs/2502.17166) | Extract crime elements, charges and statutes |
| `keller_extract` | [KELLER](https://aclanthology.org/2024.emnlp-main.73/) | Extract charges and summarize sub-facts |
| `promptcase_summary` | [PromptCase](https://arxiv.org/abs/2309.02962) | Summarize legal facts and issues |
| `custom` | User-defined | Supply `custom_prompt` containing `{fact}` |

These options do not establish full reproduction of those papers. In particular, the Human-1 template does not implement the self-evolving agent in *When Rules Learn*. Exact prompt fidelity and experimental equivalence have not been verified here. Select a template explicitly when enabling `llm_query_rewrite`; if omitted in that mode, the existing fallback remains `when_rules_learn_human1`. BM25 mode rejects prompt settings.

### Retrieval scope

`full_corpus` searches the entire corpus and reports recall. `labeled_pool` and `keller_pool` are optional restricted-pool ranking settings, with MAP, MRR, precision and NDCG metrics. They are not interchangeable with the original full-corpus experiment. LexVerse's `keller_pool` reads a prebuilt candidate-pool file, which can be specified with `candidate_pool_path`; record the scope and split when comparing results.

---

## 3. How to Run & Configure

### 3.1 Installation

```bash
python -m pip install -e ".[lecardv2]"
lexverse task catalog lecardv2
```

### Default BM25 run

```bash
lexverse run -c configs/jobs/lecardv2.yaml --dry-run
lexverse run -c configs/jobs/lecardv2.yaml
```

Using `-c` explicitly avoids an existing personal YAML overriding the shipped BM25 example. The full candidate corpus is required even for a two-query smoke test.

### 3.2 Optional Example A: Evaluating a Retrieval Encoder (`mode: encoder_retrieval`)

```yaml
schema_version: '1.0'
benchmark:
  name: lecardv2
  mode: encoder_retrieval
  split: test
  candidate_scope: keller_pool
  document_field: fact
  pooling: cls
  normalize: false
  max_length: 512
  tasks:
  - id: case_retrieval
    limit: null
models:
  default:
    provider: transformers
    name: CSHaitao/SAILER_zh
    load_parameters:
      local_files_only: true
      device_map: cuda
generation:
  offline: true
agents:
- name: native
verification:
  concurrency: 1
concurrency: 1
```

### 3.3 Optional Example B: Evaluating a Generative LLM (`mode: llm_query_rewrite`)

```yaml
schema_version: '1.0'
benchmark:
  name: lecardv2
  mode: llm_query_rewrite
  prompt_template: when_rules_learn_human1
  split: all
  candidate_scope: full_corpus
  document_field: qw
  bm25_k1: 1.2
  bm25_b: 0.75
  top_k: 1000
  cutoffs:
  - 100
  - 200
  - 500
  - 1000
  tasks:
  - id: case_retrieval
    limit: 10
models:
  default:
    provider: transformers
    name: Qwen/Qwen2.5-7B-Instruct
    load_parameters:
      local_files_only: true
      dtype: bfloat16
      device_map: cuda
    input:
      format: chat
    generation_parameters:
      temperature: 0
      max_tokens: 512
generation:
  offline: true
agents:
- name: native
verification:
  concurrency: 1
concurrency: 1
```

### 3.4 Execution Commands

```bash
lexverse run -b lecardv2 -c configs/jobs/lecardv2.local.yaml --dry-run
lexverse run -b lecardv2 -c configs/jobs/lecardv2.local.yaml --output-dir runs/lecardv2-experiment
lexverse job resume runs/lecardv2-experiment
lexverse job verify runs/lecardv2-experiment
lexverse job report runs/lecardv2-experiment
```

---

## 4. LexVerse implementation

The pinned upstream dataset (`3326485a6c41f8aeda15502fea24e658f8fab746`) does not include a runnable retrieval/scoring pipeline. LexVerse implements:

- `BM25CorpusIndex`: a sparse BM25 index using Jieba tokenization and upstream stopwords, rather than the original Pyserini/Lucene runner.
- Encoder and sub-fact retrieval adapters with configurable pooling, normalization, text fields and truncation.
- A verifier using `pytrec_eval` for per-query metrics and LexVerse aggregation (`macro_nonempty`, `macro_all_zero`, `micro` and pool metrics).

Using the original labels and standard metric library does not establish identical preprocessing, ranking or aggregation to a paper. Full reproduction requires those settings and run artifacts to be checked separately.

---

## 5. Reference Results (`lexverse run`) & Implementation Notes

The tables below are historical results carried over from the benchmark integration branch, which reported runs on an NVIDIA A100. They have not been revalidated during this integration and are not results of the shipped two-query BM25 example. Paper figures and reported LexVerse figures are labeled separately; matching numbers alone do not prove exact reproduction.

### 5.1 Candidate-Pool Reranking (160 Test Queries)

**Comparison with *KELLER* (EMNLP 2024, Table 2) under `candidate_scope: keller_pool`**:

| Mode | Model / Method | Source | `MAP` (%) | `P@3` (%) | `NDCG@3` (%) | `NDCG@5` (%) | `NDCG@10` (%) |
| :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Mode A** | **`BM25`** (`mode: bm25_retrieval`) | Paper Table 2 | 55.20 | 48.75 | 72.11 | 72.51 | 74.47 |
| | | **`lexverse run lecardv2`** | **55.20** | **48.75** | **72.11** | **72.51** | **74.47** |
| **Mode A** | **`CSHaitao/SAILER_zh`** (`mode: encoder_retrieval`) | Paper Table 2 | 62.80 | 55.00 | 79.38 | 81.17 | 83.83 |
| | | **`lexverse run lecardv2`** | **62.80** | **55.00** | **79.38** | **81.15** | **83.82** |
| **Mode A** | **`hfl/chinese-roberta-wwm-ext`** (`mode: encoder_retrieval`) | Paper Table 2 | 57.89 | 52.08 | 75.48 | 76.33 | 78.38 |
| | | **`lexverse run lecardv2`** | **57.90** | **52.08** | **75.50** | **76.34** | **78.36** |
| **Mode A** | **`BAAI/bge-base-zh-v1.5`** (`mode: encoder_retrieval`) | Paper Table 2 | 57.21 | 50.42 | 73.59 | 75.36 | 77.80 |
| | | **`lexverse run lecardv2`** | **57.21** | **50.42** | **73.59** | **75.36** | **77.81** |
| **Mode A** | **`google-bert/bert-base-chinese`** (`mode: encoder_retrieval`) | Paper Table 2 | 56.46 | 52.08 | 75.82 | 77.05 | 79.39 |
| | | **`lexverse run lecardv2`** | **56.46** | **52.08** | **75.82** | **77.04** | **79.31** |
| **Mode A** | **`thunlp/Lawformer`** (`max_length: 3072`) | Paper Table 2 | 55.05 | 49.58 | 74.42 | 74.31 | 76.96 |
| | | **`lexverse run lecardv2`** | **55.05** | **49.58** | **73.42** | **74.31** | **76.96** |
| **Mode B** | **`KELLER Zero-shot` (LLM Sub-Facts + `SAILER_zh`)** | Paper Table 2 | 65.87 | 61.67 | 83.33 | 83.75 | 86.06 |
| | | **`lexverse run lecardv2`** | **65.92** | **61.88** | **83.35** | **83.89** | **86.08** |

**Comparison with *CCE Cross-Benchmark Audit* (arXiv:2606.12993, Table 8) under `candidate_scope: labeled_pool` (`pooling: mean, normalize: true`)**:

| Mode | Model | Metric | Paper Table 8 | Authors' Released JSON | **`lexverse run lecardv2`** |
| :---: | :--- | :--- | ---: | ---: | ---: |
| **Mode A** | **`BM25`** (`mode: bm25_retrieval`) | Standard `ndcg_at_10` | `0.7453` | — | **`0.7479`** |
| | | Charge-Stratified `ndcg_at_10` | `0.7440` | — | **`0.7456`** |
| **Mode A** | **`CSHaitao/SAILER_zh`** | Standard `ndcg_at_10` | `0.7924` | `0.7924109977064477` | **`0.7924109977064477`** |
| | | Charge-Stratified `ndcg_at_10` | `0.8044` | `0.8044112285048912` | **`0.8044112285048913`** |
| **Mode A** | **`hfl/chinese-roberta-wwm-ext`** | Standard `ndcg_at_10` | `0.7909` | `0.7909082647140177` | **`0.7909082647140177`** |
| | | Charge-Stratified `ndcg_at_10` | `0.8011` | `0.8011275600688926` | **`0.8011275600688925`** |

### 5.2 Full-Corpus Retrieval (800 Queries × 55,192 Docs)

**Comparison on `candidate_scope: full_corpus`**:

| Mode | Model / Method | Source | `R@100` | `R@200` | `R@500` | `R@1000` |
| :---: | :--- | :--- | ---: | ---: | ---: | ---: |
| **Mode A** | **`BM25`** (`mode: bm25_retrieval`) | LeCaRDv2 Paper (SIGIR 2024 Table 2) | 0.6262 | 0.6629 | 0.6949 | 0.7207 |
| | | **`lexverse run lecardv2`** | **0.4651** | **0.5656** | **0.6779** | **0.7549** |
| **Mode A** | **`CSHaitao/SAILER_zh`** (`mode: encoder_retrieval`) | *JUREX-4E* (EMNLP 2025 Appendix H Table 15) | 0.2834 | 0.4033 | 0.6104 | 0.7568 |
| | | **`lexverse run lecardv2`** | **0.2822** | **0.4109** | **0.6312** | **0.7848** |
| **Mode A** | **`hfl/chinese-bert-wwm`** (`mode: encoder_retrieval`) | LeCaRDv2 Paper (SIGIR 2024 Table 2) | 0.1165 | 0.1526 | 0.2184 | 0.2805 |
| | | **`lexverse run lecardv2`** | **0.1218** | **0.1594** | **0.2296** | **0.2977** |

### Implementation & Protocol Notes

- **Default `max_length: 512` and `document_field: fact` for Encoders**: Full Chinese criminal judgments (`qw`) begin with ~450 tokens of court headers, docket numbers, and defendant biographical records before the case facts (`fact`) appear (median start offset: token 467). Using `document_field: fact` with `max_length: 512` ensures 512-token encoders encode the actual case facts rather than truncating inside the header preamble.
- **Zero-Gold Query Denominators**: In `label/relevence_gold.trec` (`grade >= 2`), two queries (`qid 77` and `qid 420`) have zero positive documents. `LeCaRDv2BenchmarkVerifier` reports `macro_nonempty` (798 queries) as the primary metric and simultaneously records `macro_all_zero` (800 queries) and `micro` in `summary.json`.
