"""Import LeCaRDv2 queries and query-rewriting templates while isolating qrels to the verifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from lexverse.benchmarks.catalog import load_catalog
from lexverse.benchmarks.plugin import TaskImporter
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from . import CATALOG_PATH, UPSTREAM

TASK_TYPE = "case_retrieval"
PROTOCOL = "lecardv2_public_v1"
MODES = {"ranked_submission", "bm25_retrieval", "encoder_retrieval", "keller_subfact_retrieval", "llm_query_rewrite"}
SPLITS = {"all", "test", "train"}
CANDIDATE_SCOPES = {"full_corpus", "labeled_pool", "keller_pool"}
LIMITATIONS = [
    "Upstream THUIR/LeCaRDv2 releases dataset files only; LexVerse evaluates TREC runs via pytrec_eval (macro_nonempty / macro_all_zero / micro recall and candidate-pool MAP/MRR/P@K/NDCG@K).",
    "Mode A (bm25_retrieval / encoder_retrieval / ranked_submission) evaluates retrieval directly on raw case facts; Mode B (llm_query_rewrite / keller_subfact_retrieval) evaluates LLM-reformulated queries.",
    "Candidate-pool evaluation supports labeled_pool (test_relevence.trec, CCE Table 8) and keller_pool (KELLER Table 2).",
]

PROMPT_TEMPLATES = {
    "when_rules_learn_human1": {
        "paper": "When Rules Learn: A Self-Evolving Agent for Legal Case Retrieval (ACL Findings 2026, Appendix B & Fig. 1)",
        "template": (
            "你是一名法律专家，你的最终目标是利用检索器从历史案件数据库中搜集与**待决案件**相似的案件，即待决案件的“类案”。\n"
            "现在，请你结合下列**修改规则**，基于**待决案件基本事实**进行修改。修改后的文本将作为原有检索查询语的增强，以确保待决案件的类案出现在检索结果中。\n\n"
            "# 修改规则：\n1. 同义词和术语变体扩展：对于每个关键法律概念，添加其同义词、近义词或常见变体。格式：[术语1]、[术语2]、[术语3]、……\n\n"
            "# 待决案件基本事实：\n{fact}\n\n"
            "# 要求：\n你需要严格遵循**修改规则**，以**待决案件基本事实**为基础，将其修改为更适合作为检索查询语的文本。"
            "已知，数据库中存储的是历史案件的全文，包括基本事实、裁判理由、判决结果等内容。检索器采用的是BM25方法。\n"
            "请注意，你无需重新输出一遍待决案件基本事实的原文。\n"
            "请你在思考后，针对每一条修改规则，按照该规则的改写策略，并严格遵循规则后附的格式要求，直接输出相应的增强查询语，不要做额外解释。\n"
            "在输出时，请你以“修改后的检索查询语：”作为总体的开头。之后，每条规则对应的增强查询语依次以数字标号1、2、3……作为起始。请不要提供“修改说明”之类的冗余解释。"
        ),
    },
    "jurex_4e_fet": {
        "paper": "JUREX-4E: Juridical Expert-Annotated Four-Element Knowledge Base for Legal Reasoning (EMNLP 2025, Table 14)",
        "template": (
            "你是一个刑法领域的专家。请对以下案情事实，根据中国刑法知识，按顺序输出："
            "1、犯罪的四要件，包括：犯罪客体（指某种抽象的社会利益的具像化，比如侵犯人身利益的对象是生命权，侵犯财产利益案件的对象是手机、钱包等）、"
            "犯罪客观方面（指犯罪活动的客观事实，包括触发犯罪的关键行为，如偷、抢，和行为所导致的结果，如重伤、死亡、财产损失等）、"
            "犯罪主体（通常为一般主体，但是有特殊主体，比如国家工作人员）、犯罪主观方面（故意或者过失）"
            "2.罪名：只输出具体的罪名"
            "3.相关法条：只输出法条编号。"
            "输出格式为json。对于案件中涉及的每个罪名，分别输出为一个字典。参考样例如下：\n"
            '{"罪名1":{"犯罪的四要件":{"犯罪客体":"人身利益：被害人王某某的生命权、财产利益：车辆",'
            '"犯罪客观方面":"被告人吴某某酒后驾驶小型轿车与被害人王某某相撞，造成被害人王某某当场死亡及车辆损坏",'
            '"犯罪主体":"司机被告人吴某某","犯罪主观方面":"过失"},"罪名":"交通肇事罪","相关法条":"第133条"}}\n'
            "请根据以下案情事实，输出分析结果。请注意，不要直接回答样例的内容。案情事实：{fact}\n你只需要输出json，不要输出多余的信息。"
        ),
    },
    "keller_extract": {
        "paper": "KELLER: Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation (EMNLP 2024, Appendix B)",
        "template": (
            "你现在是一位中国法律专家，擅长从冗长的法律文本中总结关键事实。"
            "请你先指出以下案件中涉及的罪名，然后以“与该罪名相关的犯罪情节的起因、经过、结果分别为：”作为开头，"
            "用自己的语言分别总结案件文本中与该罪名相关的起因、经过、结果，每一部分都保证在50字内。\n\n案件基本事实：\n{fact}"
        ),
    },
    "promptcase_summary": {
        "paper": "PromptCase: A Prompt-Based Case Representation Method for Legal Case Retrieval (ADC 2023)",
        "template": (
            "请根据以下刑事案件的基本事实，分别用精炼的法律语言总结出两段内容：\n"
            "1. 法律事实（概括犯罪时间、主体、核心犯罪行为、数额及危害后果，150字以内）；\n"
            "2. 法律纠纷与焦点（概括本案涉嫌的罪名、定罪量刑核心争议点及法定情节，100字以内）。\n\n"
            "请严格按以下格式输出：\n法律事实：...\n法律纠纷：...\n\n案件基本事实：\n{fact}"
        ),
    },
}


def identifier(value):
    if isinstance(value, bool) or not re.fullmatch(r"0|[1-9][0-9]*", str(value)):
        raise ValueError(f"invalid LeCaRDv2 numeric ID: {value!r}")
    return int(value)


def resolve_data_layout(settings=None, *, offline=False):
    settings = settings or {}
    raw_dir = settings.get("data_dir")
    root = Path(raw_dir).expanduser().resolve() if raw_dir else (Path(".lexverse/datasets/lecardv2") / UPSTREAM.commit).resolve()
    repo = root / "repository" if (root / "repository/query/query.json").is_file() else root
    if not (repo / "query/query.json").is_file() and not raw_dir:
        repo = UPSTREAM.ensure(offline=offline)

    query_file = {"all": "query/query.json", "test": "query/test_query.json", "train": "query/train_query.json"}[
        settings.get("split", "all")
    ]
    paths = {
        "root": root,
        "repo": repo,
        "query": repo / query_file,
        "all_query": repo / "query/query.json",
        "graded_qrels": repo / "label/relevence.trec",
        "gold_qrels": repo / "label/relevence_gold.trec",
        "stopwords": repo / "others/stopword.txt",
    }
    if (repo / "query/query_allcontext.json").is_file():
        paths["query_allcontext"] = repo / "query/query_allcontext.json"
    for name in ("query", "all_query", "graded_qrels", "gold_qrels", "stopwords"):
        if not paths[name].is_file():
            raise ValueError(f"LeCaRDv2 dataset file missing: {paths[name]}")

    if (root / "candidate.files.json").is_file():
        paths["candidate_manifest"] = root / "candidate.files.json"
    for cand_dir in (root / "extracted/candidate_55192", root / "candidate_55192"):
        if cand_dir.is_dir():
            paths["candidate_dir"] = cand_dir
            break
    for field in ("qw", "fact"):
        tok_path = root / f"baseline/corpus/{field}/documents.jsonl"
        if tok_path.is_file():
            paths[f"tokenized_corpus_{field}"] = tok_path

    keller_dir = root / "keller_cache" if (root / "keller_cache").is_dir() else root.parent / "keller_cache"
    if keller_dir.is_dir():
        for key, fname in (
            ("keller_pool", "keller_test_pool_160.json"),
            ("keller_query_fact", "query_fact.json"),
            ("keller_doc_fact", "doc_fact.json"),
            ("keller_bm25", "rank_bm25.json"),
        ):
            if (keller_dir / fname).is_file():
                paths[key] = keller_dir / fname
    for cfg_key, path_key in (
        ("candidate_pool_path", "custom_pool"),
        ("decomposed_query_path", "keller_query_fact"),
        ("decomposed_doc_path", "keller_doc_fact"),
    ):
        if settings.get(cfg_key):
            paths[path_key] = Path(settings[cfg_key]).expanduser().resolve()
    return paths


def validate_benchmark(bench):
    mode = bench.get("mode", "ranked_submission")
    if mode not in MODES:
        raise ValueError(f"LeCaRDv2 benchmark.mode must be one of {sorted(MODES)}, got {mode!r}")
    if bench.get("split", "all") not in SPLITS:
        raise ValueError(f"LeCaRDv2 benchmark.split must be one of {sorted(SPLITS)}")
    scope = bench.get("candidate_scope", "full_corpus")
    if scope not in CANDIDATE_SCOPES:
        raise ValueError(f"LeCaRDv2 benchmark.candidate_scope must be one of {sorted(CANDIDATE_SCOPES)}")

    top_k = bench.get("top_k", 1000 if scope == "full_corpus" else 40)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("LeCaRDv2 benchmark.top_k must be a positive integer")
    cutoffs = bench.get("cutoffs", [100, 200, 500, 1000] if scope == "full_corpus" else [])
    if cutoffs and (
        not isinstance(cutoffs, list)
        or any(type(k) is not int or k < 1 or k > top_k for k in cutoffs)
        or cutoffs != sorted(set(cutoffs))
    ):
        raise ValueError("LeCaRDv2 benchmark.cutoffs must be a sorted list of unique positive integers <= top_k")

    if mode == "ranked_submission":
        if "prompt_template" in bench or "custom_prompt" in bench:
            raise ValueError("LeCaRDv2 ranked_submission mode does not accept prompt_template or custom_prompt")
        sub = bench.get("submission_path")
        if not isinstance(sub, str) or not sub.strip() or not Path(sub).expanduser().is_file():
            raise ValueError("LeCaRDv2 ranked_submission mode requires an existing benchmark.submission_path")
    elif mode in {"bm25_retrieval", "encoder_retrieval"}:
        if "submission_path" in bench or "prompt_template" in bench or "custom_prompt" in bench:
            raise ValueError(f"LeCaRDv2 {mode} mode does not accept submission_path or prompt_template")
        if bench.get("document_field", "fact" if mode == "encoder_retrieval" else "qw") not in {"fact", "qw"}:
            raise ValueError("LeCaRDv2 benchmark.document_field must be 'fact' or 'qw'")
        if mode == "encoder_retrieval":
            if bench.get("pooling", "cls") not in {"cls", "mean"}:
                raise ValueError("LeCaRDv2 benchmark.pooling must be 'cls' or 'mean'")
            if type(bench.get("max_length", 512)) is not int or bench.get("max_length", 512) < 8:
                raise ValueError("LeCaRDv2 benchmark.max_length must be an integer >= 8")
    elif mode == "keller_subfact_retrieval":
        if "submission_path" in bench:
            raise ValueError("LeCaRDv2 keller_subfact_retrieval mode does not accept submission_path")
        if type(bench.get("max_crime_num", 4)) is not int or bench.get("max_crime_num", 4) < 1:
            raise ValueError("LeCaRDv2 benchmark.max_crime_num must be a positive integer")
    else:
        if "submission_path" in bench:
            raise ValueError("LeCaRDv2 llm_query_rewrite mode cannot combine with benchmark.submission_path")
        template = bench.get("prompt_template", "when_rules_learn_human1")
        if template == "custom":
            if not isinstance(bench.get("custom_prompt"), str) or "{fact}" not in bench["custom_prompt"]:
                raise ValueError("LeCaRDv2 prompt_template='custom' requires benchmark.custom_prompt containing '{fact}'")
        elif template not in PROMPT_TEMPLATES:
            raise ValueError(f"unknown LeCaRDv2 prompt_template {template!r}")
        if bench.get("document_field", "qw") not in {"qw", "fact"}:
            raise ValueError("LeCaRDv2 benchmark.document_field must be 'qw' or 'fact'")


def load_candidate_pids(paths):
    if "candidate_manifest" in paths:
        data = json.loads(paths["candidate_manifest"].read_text(encoding="utf-8"))
        return sorted({identifier(entry["pid"]) for entry in data.get("files", [])})
    if "candidate_dir" in paths:
        return sorted(identifier(json.loads(p.read_text(encoding="utf-8"))["pid"]) for p in paths["candidate_dir"].glob("*.json"))
    raise ValueError("LeCaRDv2 requires candidate.files.json or candidate_55192/*.json in data_dir")


def load_qrels(path, allowed_qids, allowed_pids, *, gold):
    labels = {qid: {} for qid in allowed_qids}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4 or fields[1] != "0":
            raise ValueError(f"invalid qrels format at {path}:{line_no}")
        qid, pid, grade = identifier(fields[0]), identifier(fields[2]), int(fields[3])
        if pid not in allowed_pids or grade not in ((2, 3) if gold else (0, 1, 2, 3)):
            raise ValueError(f"invalid qrels entry at {path}:{line_no}")
        if qid in labels:
            labels[qid][pid] = grade
    return labels


def load_candidate_pools(settings, paths, qids, allowed_pids):
    scope = settings.get("candidate_scope", "full_corpus")
    if scope == "full_corpus":
        return None
    if scope == "labeled_pool":
        graded = load_qrels(paths["graded_qrels"], qids, allowed_pids, gold=False)
        return {qid: sorted(graded[qid]) for qid in qids}
    pool_file = paths.get("custom_pool") or paths.get("keller_pool")
    if not pool_file or not pool_file.is_file():
        raise ValueError("LeCaRDv2 candidate_scope='keller_pool' requires keller_test_pool_160.json")
    raw = json.loads(pool_file.read_text(encoding="utf-8"))
    return {
        qid: sorted({identifier(d) for d in (raw[str(qid)].keys() if isinstance(raw[str(qid)], dict) else raw[str(qid)])})
        for qid in qids
    }


def source_inputs(settings=None, *, offline=False):
    settings = settings or {}
    paths = resolve_data_layout(settings, offline=offline)
    inputs = {"query": paths["query"], "stopwords": paths["stopwords"]}
    if "candidate_manifest" in paths:
        inputs["candidate_manifest"] = paths["candidate_manifest"]
    elif "candidate_dir" in paths:
        inputs.update({f"candidate/{p.name}": p for p in sorted(paths["candidate_dir"].glob("*.json"))})
    mode = settings.get("mode", "ranked_submission")
    if mode == "ranked_submission" and settings.get("submission_path"):
        inputs["submission"] = Path(settings["submission_path"]).expanduser().resolve()
    if settings.get("candidate_scope") == "keller_pool":
        pool_file = paths.get("custom_pool") or paths.get("keller_pool")
        if pool_file and pool_file.is_file():
            inputs["candidate_pool"] = pool_file
    if mode == "bm25_retrieval" and settings.get("candidate_scope") == "keller_pool" and "keller_bm25" in paths:
        inputs["keller_bm25"] = paths["keller_bm25"]
    if mode == "keller_subfact_retrieval":
        for key in ("keller_query_fact", "keller_doc_fact"):
            if key in paths and paths[key].is_file():
                inputs[key] = paths[key]
    return inputs


def evaluation_source_inputs(settings=None, *, offline=False):
    paths = resolve_data_layout(settings, offline=offline)
    inputs = {"graded_qrels": paths["graded_qrels"], "gold_qrels": paths["gold_qrels"]}
    if "query_allcontext" in paths:
        inputs["query_allcontext"] = paths["query_allcontext"]
    return inputs


class LeCaRDv2Importer(TaskImporter):
    def __init__(self, settings=None, *, offline=False):
        self.settings = dict(settings or {})
        validate_benchmark(self.settings)
        self.catalog = load_catalog(CATALOG_PATH)
        self.paths = resolve_data_layout(self.settings, offline=offline)
        self.candidate_pids = load_candidate_pids(self.paths)
        self.allowed_pids = set(self.candidate_pids)
        self.queries = sorted(
            (
                {"id": identifier(row["id"]), "fact": row["fact"].strip()}
                for line in self.paths["query"].read_text(encoding="utf-8").splitlines()
                if line.strip()
                for row in [json.loads(line)]
            ),
            key=lambda r: r["id"],
        )
        qids = {r["id"] for r in self.queries}
        self.gold = load_qrels(self.paths["gold_qrels"], qids, self.allowed_pids, gold=True)
        self.graded = load_qrels(self.paths["graded_qrels"], qids, self.allowed_pids, gold=False)
        self.pools = load_candidate_pools(self.settings, self.paths, qids, self.allowed_pids)

    def import_tasks(self, selection):
        mode = self.settings.get("mode", "ranked_submission")
        split = self.settings.get("split", "all")
        scope = self.settings.get("candidate_scope", "full_corpus")
        is_mode_b = mode in {"llm_query_rewrite", "keller_subfact_retrieval"}
        template_key = (
            self.settings.get("prompt_template", "keller_extract" if mode == "keller_subfact_retrieval" else "when_rules_learn_human1")
            if is_mode_b else None
        )
        query_sha256 = hashlib.sha256(self.paths["query"].read_bytes()).hexdigest()
        tasks = []
        for kind in selection.select_types([TASK_TYPE]):
            for row in selection.apply(kind, self.queries, sample_id=lambda r: str(r["id"])):
                qid, fact = row["id"], row["fact"]
                sid = str(qid)
                if is_mode_b:
                    raw_tpl = self.settings["custom_prompt"] if template_key == "custom" else PROMPT_TEMPLATES[template_key]["template"]
                    instruction, content, messages = raw_tpl.replace("{fact}", fact), "", [{"role": "user", "content": raw_tpl.replace("{fact}", fact)}]
                else:
                    instruction, content, messages = "", fact, []
                payload = {"qid": qid, "fact": fact}
                if self.pools is not None:
                    payload["candidate_pids"] = self.pools[qid]
                role = "rewriter" if is_mode_b else "retriever"
                tasks.append(LexVerseTask(
                    id=f"lecardv2:{kind}:{sid}",
                    source=SourceRef(
                        name="lecardv2", version=UPSTREAM.commit, task_type=kind, sample_id=sid,
                        provenance={
                            "protocol": PROTOCOL, "reproduction_mode": mode, "split": split, "candidate_scope": scope,
                            "query_sha256": query_sha256,
                            "candidate_count": len(self.pools[qid]) if self.pools is not None else len(self.candidate_pids),
                            "prompt_template": template_key,
                            "prompt_paper": PROMPT_TEMPLATES.get(template_key, {}).get("paper"),
                        },
                    ),
                    input=TaskInput(instruction=instruction, content=content, messages=messages),
                    participants=[ParticipantSpec(id=role, role=role, kind="model" if mode == "llm_query_rewrite" else "scripted")],
                    interaction=InteractionSpec(policy=f"lecardv2_{mode}", max_steps=1),
                    evaluation=EvaluationSpec(
                        verifier="lecardv2", metrics=list(self.catalog["tasks"][kind]["metrics"]), reference=None,
                    ),
                    metadata={
                        "protocol": PROTOCOL, "reproduction_mode": mode, "split": split, "candidate_scope": scope,
                        "prompt_template": template_key,
                        "top_k": self.settings.get("top_k", 1000 if scope == "full_corpus" else 40),
                        "cutoffs": self.settings.get("cutoffs", [100, 200, 500, 1000] if scope == "full_corpus" else []),
                        "tie_break": self.settings.get(
                            "tie_break",
                            "str_desc" if scope == "keller_pool" else ("did_desc" if scope == "labeled_pool" else "pid_asc"),
                        ),
                    },
                    raw_payload=payload,
                ))
        return tasks
