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

## 2. Evaluation Modes & Scopes

The LeCaRDv2 paper and subsequent LLM literature evaluate two families of methods across two retrieval scopes. LexVerse supports both so you can benchmark **traditional lexical retrieval (`BM25`)**, **dedicated neural retrieval encoders**, or **generative LLMs**:

### 2.1 Choosing the Evaluation Mode (`benchmark.mode`)

1. **Mode A — Direct Retrieval Baselines (`bm25_retrieval` & `encoder_retrieval`)**:
   - **Origin**: The original LeCaRDv2 paper ([SIGIR 2024, Table 2](https://arxiv.org/abs/2310.17609)), *KELLER* ([EMNLP 2024, Table 2](https://aclanthology.org/2024.emnlp-main.73/)), and *CCE Audit* ([arXiv:2606.12993, Table 8](https://arxiv.org/abs/2606.12993)).
   - **How it works**: Evaluates non-generative retrieval baselines directly on raw query and candidate texts without any generative rewriting prompt:
     - `mode: bm25_retrieval`: Runs traditional lexical **BM25** retrieval over the candidate pool or full 55,192-doc corpus.
     - `mode: encoder_retrieval`: Runs a pretrained bi-encoder (`CSHaitao/SAILER_zh`, `hfl/chinese-roberta-wwm-ext`, `BAAI/bge-base-zh-v1.5`, `google-bert/bert-base-chinese`, `thunlp/Lawformer`) on local GPU.
     - *(Optional utility)* `mode: ranked_submission`: Directly scores an already-generated external TREC run file (`benchmark.submission_path`) without running retrieval.

2. **Mode B — Generative LLMs via Case Reformulation (`llm_query_rewrite` & `keller_subfact_retrieval`)**:
   - **Origin**: Because 55,192 candidate judgments cannot fit into an LLM's context window at once, published LLM papers on LeCaRDv2 prompt a generative LLM (`Qwen3-4B`, `Qwen2.5-7B-Instruct`, `GPT-4o`, `DeepSeek`) to **extract legal elements, summarize sub-facts, or expand legal terminology** before running downstream retrieval (via **BM25** in `llm_query_rewrite` or **SAILER MaxSim** in `keller_subfact_retrieval`).
   - In `mode: llm_query_rewrite`, you can select any of four published prompts via `benchmark.prompt_template`:
     - `when_rules_learn_human1` *(default)* — *When Rules Learn* ([ACL Findings 2026, Appendix B](https://arxiv.org/abs/2606.17220)): Expands key legal concepts with synonyms and statutory variants, concatenates them with the original query `fact`, and retrieves over the 55,192-doc BM25 index.
     - `jurex_4e_fet` — *JUREX-4E* ([EMNLP 2025, Table 14](https://arxiv.org/abs/2502.17166)): Extracts Chinese criminal law's Four Constitutive Elements (`犯罪四要件`: Object, Objective Aspect, Subject, Subjective Aspect), charge, and article in JSON.
     - `keller_extract` — *KELLER* ([EMNLP 2024, Appendix B](https://aclanthology.org/2024.emnlp-main.73/)): Extracts charges and summarizes each crime's cause, procedure, and outcome (`起因、经过、结果`).
     - `promptcase_summary` — *PromptCase* ([ADC 2023](https://arxiv.org/abs/2309.02962)): Distills lengthy case narratives into concise `法律事实` (Legal Facts) and `法律纠纷` (Legal Issues).
     - `custom` — Pair with `benchmark.custom_prompt: "...{fact}..."` to test custom instructions.
   - In `mode: keller_subfact_retrieval`, LexVerse runs KELLER's crime-decomposed sub-fact representation with local `SAILER_zh` MaxSim scoring.

### 2.2 Choosing the Retrieval Scope (`benchmark.candidate_scope`)

In LeCaRDv2, human experts annotated **~30 candidate documents per query** in `relevence.trec`, while the remaining ~55,162 documents per query are unlabeled. Published papers therefore report results under two scopes:
- **`full_corpus` (55,192 docs, used by the LeCaRDv2 paper, *When Rules Learn*, and *JUREX-4E* Appendix H)**: Retrieves the Top-1000 documents from the entire 55,192-doc corpus and evaluates `Recall@100 / 200 / 500 / 1000`.
- **`keller_pool` / `labeled_pool` (~30–40 labeled docs per query on the 160 test queries, used by *KELLER* and *CCE Audit*)**: Ranks only the annotated candidate pool to evaluate top-rank precision metrics (`MAP`, `P@3`, `NDCG@3/5/10`) without unlabeled false negatives. `labeled_pool` uses the exact 30-doc pool from `test_relevence.trec` (*CCE Audit* Table 8); `keller_pool` adds 10 BM25 negative distractors (ranks 100–110) to the 21 test queries whose 30 labeled docs are all positives (*KELLER* Table 2).

---

## 3. How to Run & Configure

### 3.1 Installation

```bash
python -m pip install -e ".[lecardv2,openai,test]"
python -m lexverse catalog lecardv2
```

### 3.2 Config Example A: Evaluating a Retrieval Encoder (`mode: encoder_retrieval`)

```yaml
schema_version: 1.0

benchmark:
  name: lecardv2
  mode: encoder_retrieval
  split: test                      # "test" (160 queries) or "all" (800 queries)
  candidate_scope: keller_pool     # "keller_pool" | "labeled_pool" | "full_corpus"
  document_field: fact             # "fact" or "qw"
  pooling: cls                     # "cls" (KELLER Table 2) or "mean" (CCE Table 8)
  normalize: false                 # false (inner product) or true (cosine similarity)
  max_length: 512                  # 512 for BERT/RoBERTa/SAILER/BGE; 3072 for Lawformer
  tasks:
    - {id: "case_retrieval", limit: null}

models:
  default:
    provider: transformers
    name: CSHaitao/SAILER_zh
    load_parameters:
      local_files_only: true
      device_map: cuda

generation:
  offline: true
  concurrency: 1

evaluation:
  concurrency: 1
```

### 3.3 Config Example B: Evaluating a Generative LLM (`mode: llm_query_rewrite`)

```yaml
schema_version: 1.0

benchmark:
  name: lecardv2
  mode: llm_query_rewrite
  prompt_template: when_rules_learn_human1   # Or: jurex_4e_fet | keller_extract | promptcase_summary
  split: all
  candidate_scope: full_corpus
  document_field: qw
  bm25_k1: 1.2
  bm25_b: 0.75
  top_k: 1000
  cutoffs: [100, 200, 500, 1000]
  tasks:
    - {id: "case_retrieval", limit: 10}      # Set limit to null for all 800 queries

models:
  default:
    provider: transformers
    name: Qwen/Qwen2.5-7B-Instruct           # Or an openai_compatible API model
    load_parameters:
      local_files_only: true
      dtype: bfloat16
      device_map: cuda
    input: {format: chat}
    generation_parameters:
      temperature: 0
      max_tokens: 512

generation:
  offline: true
  concurrency: 1

evaluation:
  concurrency: 1
```

### 3.4 Execution Commands

```bash
python -m lexverse run lecardv2 --config configs/benchmarks/lecardv2.local.yaml --dry-run
python -m lexverse run lecardv2 --config configs/benchmarks/lecardv2.local.yaml --output-root runs/lecardv2-experiment
python -m lexverse run --resume runs/lecardv2-experiment
python -m lexverse evaluate --run-dir runs/lecardv2-experiment
python -m lexverse collect --run-dir runs/lecardv2-experiment
```

---

## 4. Upstream Adaptation & Rationale

When inspecting the public [`THUIR/LeCaRDv2`](https://github.com/THUIR/LeCaRDv2) repository (`commit 3326485a`), we found that it is a **data-only repository** (`query/`, `label/`, `others/`, `candidate/README.md`) with zero `.py` source files. Instead, upstream `README.md` specifies the external toolkits used for the paper's experiments:
- *"The traditional methods are implemented with [pyserini](https://github.com/castorini/pyserini) toolkit and the code of the pre-trained model is referenced from [dense](https://github.com/luyug/Dense)."*
- In the authors' companion repositories ([`THUIR/SAILER`](https://github.com/THUIR/SAILER) and [`CSHaitao/DELTA`](https://github.com/CSHaitao/DELTA)), dense encoding is implemented on top of [`luyug/Dense`](https://github.com/luyug/Dense) (`src/dense/driver/encode.py`, `src/dense/modeling.py`, `src/dense/faiss_retriever/`) and evaluation is performed via [`trec_eval`](https://trec.nist.gov/trec_eval/) (`src/eval.sh`).

LexVerse integrates these exact upstream reference components in `lexverse/benchmarks/lecardv2/`:
1. **Dense Dual-Encoder Retrieval Aligned with `luyug/Dense` & `THUIR/SAILER` (`EncoderRetriever`, `KellerSubfactRetriever`)**: Follows `DenseModelForInference` (`src/dense/modeling.py`) and `EncodeDataset` (`src/dense/data.py`) to encode queries and candidate passages (`CLS` or `mean` pooling, inner-product or cosine similarity, configurable `max_length` and `document_field`) and supports KELLER sub-fact MaxSim scoring via LexVerse's standard `resolve_source(model_cfg)` loader.
2. **Vectorized BM25 Index (`BM25CorpusIndex`)**: Tokenizes the 55,192 candidate documents with `jieba==0.42.1` and the repository's `others/stopword.txt` (matching `myx666/LeCaRD` and `THUIR/LeCaRDv2`), and caches a compressed sparse row (CSR) inverted index so full-corpus retrieval runs in milliseconds per query.
3. **Official `trec_eval` Scoring via `pytrec_eval` (`LeCaRDv2TaskVerifier`, `LeCaRDv2Aggregator`)**: Delegates per-query metric calculation (`recall_100/200/500/1000`, `map`, `recip_rank`, `P_1/3`, `ndcg_cut_3/5/10/20/30`) directly to `pytrec_eval.RelevanceEvaluator` (the official `trec_eval` C engine wrapped by Pyserini and `THUIR/SAILER`), while computing `macro_nonempty`, `macro_all_zero`, `micro`, `pool_macro`, and `charge_stratified` macro averages.

---

## 5. Reference Results (`lexverse run`) & Implementation Notes

All LexVerse results below are obtained directly by running `python -m lexverse run lecardv2` on a local NVIDIA A100 GPU.

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

- **Missing Upstream Retrieval & Evaluation Scripts**: Because [`THUIR/LeCaRDv2`](https://github.com/THUIR/LeCaRDv2) contains only the dataset files (`query/`, `label/`, `others/`) and no `.py` source files (only noting in `README.md` that traditional baselines used `pyserini` and dense baselines referenced `luyug/Dense`, with `<!-- ## Evaluation -->` commented out), full-corpus numbers (`R@100..1000`) exhibit slight differences (`0.1218` vs. `0.1165` on `R@100` for `chinese-bert-wwm`) due to unreleased upstream preprocessing/indexing details. By contrast, on the follow-up benchmarks that did release their evaluation protocols (*KELLER* Table 2 and *CCE Audit* Table 8), `lexverse run lecardv2` reproduces the published scores 1:1.
- **Default `max_length: 512` and `document_field: fact` for Encoders**: Full Chinese criminal judgments (`qw`) begin with ~450 tokens of court headers, docket numbers, and defendant biographical records before the case facts (`fact`) appear (median start offset: token 467). Using `document_field: fact` with `max_length: 512` ensures 512-token encoders encode the actual case facts rather than truncating inside the header preamble.
- **Zero-Gold Query Denominators**: In `label/relevence_gold.trec` (`grade >= 2`), two queries (`qid 77` and `qid 420`) have zero positive documents. `LeCaRDv2TaskVerifier` reports `macro_nonempty` (798 queries) as the primary metric and simultaneously records `macro_all_zero` (800 queries) and `micro` in `summary.json`.
