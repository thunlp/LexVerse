# Legal-World Benchmark

In real-world civil litigation, a lawyer's decisions at one stage constrain and shape every subsequent stage. An incomplete intake interview (`LC`) leads to missing claims or evidence in the initial pleading (`DRAFT`), which in turn weakens courtroom presentation and cross-examination in the first-instance hearing (`CI`); after the court delivers its judgment (`SD`), counsel must decide whether and how to draft an appeal (`APPEAL_DRAFT`) and argue before the appellate panel (`CIA`).

**Legal-World** (*Evaluating Legal Agents in Dynamic, Multi-Stage Civil Litigation Environments*, [arXiv:2606.18728](https://arxiv.org/abs/2606.18728)) models this end-to-end civil litigation lifecycle across **100 real Chinese civil cases** (`200` complete-flow tasks across plaintiff and defendant counsel roles) with shared cross-stage memory (`CaseFile`) and stage-specific rubric evaluation.

---

## 1. Overview

- **Benchmark ID**: `legalworld`
- **Upstream Repository**: [`riotulab/Legal-World`](https://github.com/riotulab/Legal-World) (commit `1cc47ff948133d6d51b84e629dded9440eb9e55f`)
- **Dataset**: `data/benchmark_cases.jsonl` (100 civil cases × 2 party roles = **200 full-pipeline tasks**)
- **Stages & Metrics**:
  - `LC` (*Legal Consultation*): Multi-turn client intake interview; scored on `LC.stage_score` (fact discovery and legal consultation rubrics).
  - `DRAFT` (*First-Instance Pleading Drafting*): Drafts a civil complaint (`plaintiff`) or statement of defense (`defendant`); scored on `DRAFT.stage_score`.
  - `CI` (*First-Instance Court Hearing*): Multi-turn trial simulation (statement, evidence cross-examination, debate, mediation); scored on `CI.stage_score`.
  - `SD` (*Sentence Delivery*): First-instance judicial decision rendered by the simulated court (`not_implemented` as an evaluated lawyer score).
  - `APPEAL_DRAFT` (*Appeal / 答辩状 Drafting*): Drafts the appellate brief or response; scored on `APPEAL_DRAFT.stage_score`.
  - `CIA` (*Second-Instance Appellate Hearing*): Multi-turn appellate court simulation; scored on `CIA.stage_score`.
  - **Aggregate Metrics**: `overall_score` (`0–100`), `overall_possible_score` (`100.0`), and `overall_score_normalized` (`0–1`).

---

## 2. How to Run

### 2.1 Installation

```bash
python -m pip install -e ".[legalworld,openai,test]"
python -m lexverse catalog legalworld
```

### 2.2 Configuration

A Legal-World run binds three model roles:
1. **Candidate Lawyer (`models.default`)**: Acts as either the plaintiff's lawyer (`party_role: plaintiff`) or the defendant's lawyer (`party_role: defendant`) across the configured stage range (`start_stage` to `end_stage`, default `LC` through `CIA`).
2. **Environment Simulator (`models.roles.simulation`)**: Role-plays the client, opposing counsel, and presiding judge(s).
3. **Evaluator (`models.evaluators[0]`)**: Grades each completed stage against the case's expert rubrics (`data/eval_rubrics.jsonl`).

Copy `configs/benchmarks/legalworld.example.yaml` to `configs/benchmarks/legalworld.local.yaml`:

```yaml
schema_version: 1.0

benchmark:
  name: legalworld
  retrieval:
    laws: {enabled: false}
    cases: {enabled: false}
  tasks:
    - {id: "plaintiff_full", limit: 2, party_role: plaintiff, start_stage: LC, end_stage: CIA}
    - {id: "defendant_full", limit: 2, party_role: defendant, start_stage: LC, end_stage: CIA}

models:
  default:
    provider: openai_compatible
    profile: openai
    name: DeepSeek-V4-Flash
    generation_parameters:
      max_tokens: 8192
  roles:
    simulation:
      provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters:
        max_tokens: 8192
  evaluators:
    - provider: openai_compatible
      profile: openai
      name: gpt-4o-2024-08-06
      generation_parameters:
        max_tokens: 8192

generation:
  concurrency: 2
  timeout_sec: 2400

evaluation:
  concurrency: 2
  timeout_sec: 1800
```

### 2.3 Execution Commands

```bash
# Preflight check
python -m lexverse run legalworld --config configs/benchmarks/legalworld.local.yaml --dry-run

# Run multi-stage litigation simulation and rubric evaluation
python -m lexverse run legalworld --config configs/benchmarks/legalworld.local.yaml --output-root runs/legalworld-experiment

# Resume, re-evaluate, or rebuild summary tables
python -m lexverse run --resume runs/legalworld-experiment
python -m lexverse evaluate --run-dir runs/legalworld-experiment
python -m lexverse collect --run-dir runs/legalworld-experiment
```

---

## 3. Upstream Adaptation & Rationale

LexVerse executes the native `legal_world` headless pipeline (`legal_world/pipeline/runner.py` and `legal_world/evaluation/evaluator.py`) in an isolated per-Trial subprocess:

1. **Configurable Continuous Stage Ranges & Role Separation (`tasks.py` & `environment.py`)**:
   - **Context**: Upstream `benchmark_cases.jsonl` defines 100 cases, each supporting two counsel assignments (`plaintiff` and `defendant`) and any continuous slice of the six-stage pipeline (`LC -> DRAFT -> CI -> SD -> APPEAL_DRAFT -> CIA`).
   - **Adaptation**: `LegalWorldImporter` validates task definitions (`party_role`, `start_stage`, `end_stage`) and exposes only non-leaking case metadata in `TaskInput`, while `LegalWorldEnvironment` routes the evaluated lawyer role to `models.default` and all counterpart/judicial agents to `models.roles.simulation`.
2. **Optional Law & Case Retrieval Isolation (`environment.py`)**:
   - **Context**: Upstream `Legal-World` disables external law retrieval by default (`law_retrieval.enabled: false`) and optionally supports BM25 case retrieval (`search_cases`).
   - **Adaptation**: LexVerse preserves upstream's default zero-retrieval mode out of the box and, when `benchmark.retrieval.cases.enabled: true` is configured, automatically filters out benchmark docket numbers and near-duplicate case texts per Trial to prevent ground-truth leakage.

---

## 4. Reference Results (`lexverse run`) & Implementation Notes

The tables below report the results obtained by running `python -m lexverse run legalworld` across 3 representative models—the open-weight **`DeepSeek-V4-Flash`** and **`Kimi-K2.5`** alongside **`GPT-5.2`** (replacing the older `GPT-4o` baseline)—compared against Table 2 of the Legal-World paper ([arXiv:2606.18728](https://arxiv.org/abs/2606.18728), all scores in `%`):

### 4.1 Overall & Role-Specific Scores (`Plaintiff` vs. `Defendant` Full Pipeline `LC -> CIA`)

| Model | Source | **Overall** | **Plaintiff (`Overall`)** | Plaintiff `LC` | Plaintiff `DRAFT` | Plaintiff `CI` | Plaintiff `APPEAL_DRAFT` | Plaintiff `CIA` | **Defendant (`Overall`)** | Defendant `LC` | Defendant `DRAFT` | Defendant `CI` | Defendant `APPEAL_DRAFT` | Defendant `CIA` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *DeepSeek-V3.2* | Paper Table 2 | 76.62 | 78.64 | 62.66 | 93.74 | 75.87 | 92.54 | 68.40 | 74.60 | 55.83 | 86.26 | 71.75 | 90.48 | 68.67 |
| **`DeepSeek-V4-Flash`** | **`lexverse run legalworld`** | **75.85** | **77.13** | **90.00** | **86.00** | **70.00** | **91.50** | **68.33** | **74.58** | **76.67** | **74.44** | **51.67** | **95.00** | **86.67** |
| *Kimi-K2-Thinking* | Paper Table 2 | 76.96 | 77.77 | 65.58 | 92.60 | 71.09 | 89.03 | 70.53 | 76.15 | 62.58 | 88.24 | 68.18 | 91.61 | 70.13 |
| **`Kimi-K2.5`** | **`lexverse run legalworld`** | **71.91** | **78.50** | **53.33** | **87.50** | **71.67** | **90.25** | **83.33** | **65.33** | **81.67** | **76.11** | **28.33** | **95.00** | **76.67** |
| *GPT-4o (Baseline)* | Paper Table 2 | 68.49 | 69.67 | 49.08 | 88.68 | 65.91 | 83.27 | 61.40 | 67.31 | 44.33 | 82.99 | 63.36 | 85.40 | 60.47 |
| **`GPT-5.2`** | **`lexverse run legalworld`** | **74.21** | **67.72** | **73.33** | **61.50** | **75.00** | **81.08** | **55.00** | **80.71** | **91.67** | **83.61** | **66.67** | **86.67** | **86.67** |

### Implementation & Protocol Notes

- **Cross-Stage Error Propagation (`CaseFile`)**: Because Legal-World passes a single persistent `CaseFile` across stages (`LC -> DRAFT -> CI -> SD -> APPEAL_DRAFT -> CIA`), facts elicited during `LC` directly populate the drafting context for `DRAFT`, and the first-instance judgment rendered in `SD` conditions `APPEAL_DRAFT` and `CIA`.
- **Stage Weighting (`overall_score`)**: Upstream `legal_world/evaluation/evaluator.py` computes `overall_score` as a point-weighted sum across the 5 lawyer-evaluated stages (`LC`, `DRAFT`, `CI`, `APPEAL_DRAFT`, and `CIA`, totaling `100.0` possible points), while `SD` records the court's first-instance decision (`coverage.SD = "not_implemented"`).
