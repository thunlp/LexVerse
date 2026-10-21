"""Verify LeCaRDv2 TREC rankings via pytrec_eval and aggregate recall and pool metrics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from lexverse.runtime.results import execution_artifacts_valid
from lexverse.verifiers.base import (
    BenchmarkAggregator, RunEvaluationResult, TaskVerifier, VerifierProvenance, VerifierResult, mean_metrics,
)
from . import UPSTREAM
from .environment import parse_trec_submission
from .tasks import (
    LIMITATIONS, PROTOCOL, TASK_TYPE, identifier, load_candidate_pids, load_qrels, resolve_data_layout,
)


def compute_pool_ranking_metrics(ranked_pids, graded_qrels):
    import pytrec_eval

    rels = {pid: graded_qrels.get(pid, 0) for pid in ranked_pids}
    for pid, rel in graded_qrels.items():
        rels.setdefault(pid, rel)
    golden_labels = {2, 3} if all(rel != 3 for rel in rels.values()) else {3}
    qrel_bin = {"q": {str(pid): (1 if rel in golden_labels else 0) for pid, rel in rels.items()}}
    qrel_ndcg = {"q": {str(pid): ((2 ** (rel - 1)) if rel >= 1 else 0) for pid, rel in rels.items()}}
    run_dict = {"q": {str(pid): float(len(ranked_pids) - idx) for idx, pid in enumerate(ranked_pids)}}

    res_bin = pytrec_eval.RelevanceEvaluator(qrel_bin, {"map", "recip_rank", "P_1", "P_3"}).evaluate(run_dict)["q"]
    res_ndcg = pytrec_eval.RelevanceEvaluator(
        qrel_ndcg, {"ndcg_cut_3", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "ndcg_cut_30"},
    ).evaluate(run_dict)["q"]
    return {
        "map": float(res_bin["map"]),
        "mrr": float(res_bin["recip_rank"]),
        "p_at_1": float(res_bin["P_1"]),
        "p_at_3": float(res_bin["P_3"]),
        "ndcg_at_3": float(res_ndcg["ndcg_cut_3"]),
        "ndcg_at_5": float(res_ndcg["ndcg_cut_5"]),
        "ndcg_at_10": float(res_ndcg["ndcg_cut_10"]),
        "ndcg_at_20": float(res_ndcg["ndcg_cut_20"]),
        "ndcg_at_30": float(res_ndcg["ndcg_cut_30"]),
    }


def compute_trec_recalls(ranked_by_qid, gold_by_qid, cutoffs):
    import pytrec_eval

    measures = {f"recall_{k}" for k in cutoffs}
    qrels_trec = {
        str(qid): {str(pid): int(rel) for pid, rel in gold_by_qid[qid].items()}
        for qid in ranked_by_qid if gold_by_qid.get(qid)
    }
    run_trec = {
        str(qid): {str(pid): float(len(pids) - idx) for idx, pid in enumerate(pids)}
        for qid, pids in ranked_by_qid.items() if qrels_trec.get(str(qid))
    }
    eval_res = pytrec_eval.RelevanceEvaluator(qrels_trec, measures).evaluate(run_trec) if qrels_trec and measures else {}
    hits_by_qid, recalls_by_qid = {}, {}
    for qid, ranked in ranked_by_qid.items():
        gold_set = set(gold_by_qid.get(qid, {}))
        hits_by_qid[qid] = {f"recall_at_{k}": len(set(ranked[:k]) & gold_set) for k in cutoffs}
        recalls_by_qid[qid] = {
            f"recall_at_{k}": float(eval_res[str(qid)][f"recall_{k}"]) if str(qid) in eval_res else 0.0
            for k in cutoffs
        }
    return hits_by_qid, recalls_by_qid


def load_trial_ranking(task, record, run_dir, allowed_pids, *, expected_top_k=1000):
    declared = record.get("execution_artifacts", record.get("artifacts", {}))
    if (
        not isinstance(declared, dict) or not {"ranking_trec", "ranking_meta"}.issubset(declared)
        or not execution_artifacts_valid({**record, "execution_artifacts": declared}, Path(run_dir))
    ):
        raise ValueError("missing or hash-mismatched LeCaRDv2 ranking artifacts")
    trec_path = (Path(run_dir) / declared["ranking_trec"]).resolve()
    meta_path = (Path(run_dir) / declared["ranking_meta"]).resolve()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    qid = identifier(task.raw_payload["qid"])
    if meta.get("qid") != qid or hashlib.sha256(trec_path.read_bytes()).hexdigest() != meta.get("ranking_sha256"):
        raise ValueError("ranking metadata/sha256 mismatch")
    parsed = parse_trec_submission(trec_path, allowed_pids, top_k=expected_top_k)
    if set(parsed) != {qid}:
        raise ValueError(f"trial TREC must contain exactly query {qid}")
    ranked = [pid for _, pid, _, _ in parsed[qid]]
    pool = task.raw_payload.get("candidate_pids")
    if pool is not None and not set(ranked).issubset(pool):
        raise ValueError("ranking contains documents outside the candidate pool")
    if meta.get("top_10_pids") != ranked[:10]:
        raise ValueError("ranking.json top_10_pids mismatch against TREC")
    # Match trec_eval: score descending, then document ID descending as a string.
    rows = sorted(parsed[qid], key=lambda row: (row[2], str(row[1])), reverse=True)
    return [pid for _, pid, _, _ in rows]


def _build_aggregations(per_query_metrics, per_query_hits, gold_counts, primary_charge, cutoffs):
    scored_qids = sorted(per_query_metrics)
    nonempty_qids = [q for q in scored_qids if gold_counts[q] > 0]
    total_gold = sum(gold_counts[q] for q in nonempty_qids)
    agg = {"macro_nonempty": {}, "macro_all_zero": {}, "micro": {}, "pool_macro": {}, "charge_stratified": {}}
    for k in cutoffs:
        m = f"recall_at_{k}"
        s_rec = sum(per_query_metrics[q][m] for q in nonempty_qids)
        agg["macro_nonempty"][m] = s_rec / len(nonempty_qids) if nonempty_qids else 0.0
        agg["macro_all_zero"][m] = s_rec / len(scored_qids) if scored_qids else 0.0
        agg["micro"][m] = sum(per_query_hits[q][m] for q in nonempty_qids) / total_gold if total_gold > 0 else 0.0
    pool_keys = ("map", "mrr", "p_at_1", "p_at_3", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "ndcg_at_30")
    for k in pool_keys:
        if scored_qids and k in per_query_metrics[scored_qids[0]]:
            agg["pool_macro"][k] = sum(per_query_metrics[q][k] for q in scored_qids) / len(scored_qids)
    strata = {}
    for q in scored_qids:
        if primary_charge.get(q):
            strata.setdefault(primary_charge[q], []).append(q)
    if strata:
        for k in ("ndcg_at_5", "ndcg_at_10", "ndcg_at_20", "map"):
            if k in agg["pool_macro"]:
                agg["charge_stratified"][k] = sum(
                    sum(per_query_metrics[q][k] for q in qs) / len(qs) for qs in strata.values()
                ) / len(strata)
    return agg


class LeCaRDv2TaskVerifier(TaskVerifier):
    name = "lecardv2"
    grouped = True

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.paths = resolve_data_layout(self.config, offline=bool(self.config.get("offline", False)))
        self.allowed_pids = set(load_candidate_pids(self.paths))

    async def evaluate_unit(self, tasks, records, *, run_dir, model_name, context):
        del model_name, context
        result = RunEvaluationResult(expected_task_ids=[task.id for task in tasks])
        provenance = VerifierProvenance(name=self.name, version="2.0", upstream_commit=UPSTREAM.commit)
        if not tasks:
            return result

        scope = tasks[0].metadata.get("candidate_scope", "full_corpus")
        top_k = int(tasks[0].metadata.get("top_k", 1000)) if scope == "full_corpus" else None
        cutoffs = list(tasks[0].metadata.get("cutoffs", [100, 200, 500, 1000] if scope == "full_corpus" else []))
        qids = {identifier(task.raw_payload["qid"]) for task in tasks}
        gold = load_qrels(self.paths["gold_qrels"], qids, self.allowed_pids, gold=True)
        graded = load_qrels(self.paths["graded_qrels"], qids, self.allowed_pids, gold=False)

        primary_charge = {}
        if "query_allcontext" in self.paths:
            for line in self.paths["query_allcontext"].read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("law"):
                        primary_charge[identifier(row["id"])] = str(row["law"][0]).strip()

        valid_cases = []
        for task in tasks:
            record = records.get(task.id, {})
            if record.get("status") != "completed":
                continue
            try:
                ranked = load_trial_ranking(task, record, Path(run_dir), self.allowed_pids, expected_top_k=top_k)
                valid_cases.append((task, identifier(task.raw_payload["qid"]), ranked))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result.trials[task.id] = VerifierResult(status="invalid", raw_result={"reason": str(exc)}, provenance=provenance)
        if not valid_cases:
            return result

        work = Path(run_dir) / "verifier" / TASK_TYPE
        work.mkdir(parents=True, exist_ok=True)
        merged_trec = work / "merged.top1000.trec"
        merged_lines, ranked_by_qid, gold_counts = [], {}, {}
        for task, qid, ranked in sorted(valid_cases, key=lambda item: item[1]):
            declared = records[task.id].get("execution_artifacts", records[task.id].get("artifacts", {}))
            merged_lines.append((Path(run_dir) / declared["ranking_trec"]).read_text(encoding="utf-8"))
            gold_counts[qid] = len(gold[qid])
            ranked_by_qid[qid] = ranked
        merged_trec.write_text("".join(merged_lines), encoding="utf-8")

        per_query_hits, per_query_recalls = compute_trec_recalls(ranked_by_qid, gold, cutoffs)
        per_query_metrics = {
            qid: {**per_query_recalls[qid], **compute_pool_ranking_metrics(ranked, graded[qid])}
            for qid, ranked in ranked_by_qid.items()
        }
        aggregations = _build_aggregations(per_query_metrics, per_query_hits, gold_counts, primary_charge, cutoffs)
        primary_metrics = {**aggregations["macro_nonempty"], **aggregations["pool_macro"]}
        zero_gold_qids = sorted(q for q, c in gold_counts.items() if c == 0)
        nonempty_count = sum(1 for c in gold_counts.values() if c > 0)

        grading_path = work / "grading.json"
        grading_path.write_text(
            json.dumps({
                "protocol": PROTOCOL,
                "primary_aggregation": "macro_nonempty" if scope == "full_corpus" else "pool_macro",
                "reproduction_mode": tasks[0].metadata.get("reproduction_mode", "ranked_submission"),
                "candidate_scope": scope,
                "prompt_template": tasks[0].metadata.get("prompt_template"),
                "metrics": primary_metrics,
                "aggregations": aggregations,
                "zero_gold_query_ids": zero_gold_qids,
                "nonempty_gold_queries": nonempty_count,
                "merged_trec_sha256": hashlib.sha256(merged_trec.read_bytes()).hexdigest(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts = {
            "grading": str(grading_path.relative_to(Path(run_dir).resolve())),
            "merged_trec": str(merged_trec.relative_to(Path(run_dir).resolve())),
        }
        for task, qid, _ in valid_cases:
            result.trials[task.id] = VerifierResult(
                status="completed",
                metrics=per_query_metrics[qid],
                raw_result={
                    "qid": qid, "gold_count": gold_counts[qid], "zero_gold": gold_counts[qid] == 0,
                    "hits": per_query_hits[qid], "primary_charge": primary_charge.get(qid),
                    "reproduction_mode": task.metadata.get("reproduction_mode"),
                    "candidate_scope": scope, "prompt_template": task.metadata.get("prompt_template"),
                },
                artifacts=artifacts,
                provenance=provenance,
            )
        result.groups = [{
            "task_type": TASK_TYPE, "granularity": "task_type",
            "task_ids": [task.id for task, _, _ in valid_cases],
            "expected_task_ids": result.expected_task_ids,
            "metrics": primary_metrics, "aggregations": aggregations,
            "zero_gold_query_ids": zero_gold_qids, "nonempty_gold_queries": nonempty_count,
            "artifacts": artifacts, "provenance": provenance.model_dump(mode="json"),
        }]
        return result

    def write_results(self, result, tasks, records, *, run_dir, model_name):
        scored_records = []
        for task in tasks:
            record = dict(records.get(task.id) or {"id": task.id, "status": "missing"}, task=task.source.task_type)
            if task.id in result.trials:
                verified = result.trials[task.id]
                record["evaluation"] = verified.model_dump(mode="json")
                if verified.status != "completed":
                    record["status"] = "failed"
            scored_records.append(record)
        aggregator = LeCaRDv2Aggregator()
        summary = aggregator.aggregate(scored_records, model_name=model_name)
        for group in result.groups:
            summary["tasks"][group["task_type"]]["metrics"] = group["metrics"]
        aggregator.write_csv(summary, Path(run_dir) / "evaluation_result.csv", model_name=model_name)


class LeCaRDv2Aggregator(BenchmarkAggregator):
    def aggregate(self, records, *, model_name):
        tasks = mean_metrics(records)
        completed = [
            r for r in records
            if r.get("status") == "completed" and (r.get("evaluation") or {}).get("status") == "completed"
        ]
        aggregations = {}
        zero_gold_ids = []
        nonempty_count = 0
        if TASK_TYPE in tasks and completed:
            per_q_metrics, per_q_hits, gold_counts, primary_charge = {}, {}, {}, {}
            for r in completed:
                raw = (r.get("evaluation") or {}).get("raw_result") or {}
                qid = int(raw["qid"])
                per_q_metrics[qid] = {k: float(v) for k, v in r["evaluation"]["metrics"].items()}
                per_q_hits[qid] = {k: int(v) for k, v in (raw.get("hits") or {}).items()}
                gold_counts[qid] = int(raw.get("gold_count", 0))
                if raw.get("primary_charge"):
                    primary_charge[qid] = str(raw["primary_charge"])
            cutoffs = sorted(int(k.removeprefix("recall_at_")) for k in per_q_metrics[next(iter(per_q_metrics))] if k.startswith("recall_at_"))
            aggregations = _build_aggregations(per_q_metrics, per_q_hits, gold_counts, primary_charge, cutoffs)
            tasks[TASK_TYPE]["metrics"] = {**aggregations["macro_nonempty"], **aggregations["pool_macro"]}
            zero_gold_ids = sorted(q for q, c in gold_counts.items() if c == 0)
            nonempty_count = sum(1 for c in gold_counts.values() if c > 0)

        raw0 = ((completed[0].get("evaluation") or {}).get("raw_result") or {}) if completed else {}
        scope = raw0.get("candidate_scope", "full_corpus")
        return {
            "benchmark": "lecardv2",
            "model": model_name,
            "tasks": tasks,
            "protocol": PROTOCOL,
            "primary_aggregation": "macro_nonempty" if scope == "full_corpus" else "pool_macro",
            "reproduction_mode": raw0.get("reproduction_mode", "ranked_submission"),
            "candidate_scope": scope,
            "prompt_template": raw0.get("prompt_template"),
            "aggregations": aggregations,
            "zero_gold_query_ids": zero_gold_ids,
            "nonempty_gold_queries": nonempty_count,
            "limitations": list(LIMITATIONS),
        }

    def write_csv(self, summary, path, *, model_name):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["task", "model", "metrics", "score"])
            for task_type, group in summary["tasks"].items():
                for metric, value in group["metrics"].items():
                    writer.writerow([task_type, model_name, metric, value])
        return path
