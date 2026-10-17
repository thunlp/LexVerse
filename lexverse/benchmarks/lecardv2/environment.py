"""Execute LeCaRDv2 retrieval across BM25, dual-encoders, KELLER sub-facts, and LLM query rewrites."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from lexverse.environments.base import ExecutionEnvironment
from lexverse.interaction.schema import EnvironmentResult
from lexverse.runtime.results import atomic_write_json
from .tasks import (
    PROMPT_TEMPLATES, PROTOCOL, TASK_TYPE, identifier, load_candidate_pids, resolve_data_layout,
)


def parse_trec_submission(path, allowed_pids, *, top_k=1000):
    runs = {}
    seen = {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6 or fields[1] != "Q0":
            raise ValueError(f"invalid TREC line at {path}:{line_no}")
        qid, pid, rank, score, tag = identifier(fields[0]), identifier(fields[2]), int(fields[3]), float(fields[4]), fields[5]
        if pid not in allowed_pids or not math.isfinite(score):
            raise ValueError(f"invalid PID or score at {path}:{line_no}")
        if pid in seen.setdefault(qid, set()):
            raise ValueError(f"duplicate PID {pid} for query {qid} at {path}:{line_no}")
        seen[qid].add(pid)
        runs.setdefault(qid, []).append((rank, pid, score, tag))
    for qid, rows in runs.items():
        if top_k is not None and len(rows) != top_k:
            raise ValueError(f"query {qid} has {len(rows)} rows, expected {top_k}")
        if [r for r, _, _, _ in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"query {qid} ranks are not 1..{len(rows)}")
    return runs


def sort_scored_candidates(pids, scores, *, tie_break="pid_asc", top_k=None, tag="lexverse"):
    if tie_break == "str_desc":
        ordered = sorted(zip(pids, scores), key=lambda item: (float(item[1]), str(item[0])), reverse=True)
    elif tie_break == "did_desc":
        ordered = sorted(zip(pids, scores), key=lambda item: (-float(item[1]), -int(item[0])))
    else:
        ordered = sorted(zip(pids, scores), key=lambda item: (-float(item[1]), int(item[0])))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [(rank, int(pid), float(score), tag) for rank, (pid, score) in enumerate(ordered, 1)]


class BM25CorpusIndex:
    def __init__(self, paths, *, document_field="qw", k1=1.2, b=0.75):
        import jieba

        self.paths = paths
        self.document_field = document_field
        self.k1, self.b = float(k1), float(b)
        self.stopwords = set(paths["stopwords"].read_text(encoding="utf-8").splitlines())
        self._jieba = jieba
        self._load_or_build()

    def tokenize(self, text):
        return [t for t in self._jieba.cut(text) if t.strip() and t not in self.stopwords]

    def _load_or_build(self):
        csr_dir = self.paths["root"] / "baseline" / f"csr_cache_{self.document_field}"
        required = ("pids.npy", "doc_lens.npy", "offsets.npy", "doc_indices.npy", "tfs.npy", "vocab.json")
        if all((csr_dir / name).is_file() for name in required):
            self.pids = np.load(csr_dir / "pids.npy")
            self.doc_lens = np.load(csr_dir / "doc_lens.npy").astype(np.float32)
            self.offsets = np.load(csr_dir / "offsets.npy")
            self.doc_indices = np.load(csr_dir / "doc_indices.npy", mmap_mode="r")
            self.tfs = np.load(csr_dir / "tfs.npy", mmap_mode="r")
            self.vocab = json.loads((csr_dir / "vocab.json").read_text(encoding="utf-8"))
        else:
            pids, doc_lens, postings = [], [], {}
            tok_key = f"tokenized_corpus_{self.document_field}"
            if tok_key in self.paths and self.paths[tok_key].is_file():
                for idx, line in enumerate(self.paths[tok_key].read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    pids.append(identifier(obj["id"]))
                    tokens = obj["contents"].split()
                    doc_lens.append(len(tokens))
                    for term, tf in Counter(tokens).items():
                        d_list, tf_list = postings.setdefault(term, ([], []))
                        d_list.append(idx)
                        tf_list.append(tf)
            elif "candidate_dir" in self.paths:
                for idx, cpath in enumerate(sorted(self.paths["candidate_dir"].glob("*.json"), key=lambda p: int(p.stem))):
                    obj = json.loads(cpath.read_text(encoding="utf-8"))
                    pids.append(identifier(obj["pid"]))
                    tokens = self.tokenize(str(obj.get(self.document_field, "")))
                    doc_lens.append(len(tokens))
                    for term, tf in Counter(tokens).items():
                        d_list, tf_list = postings.setdefault(term, ([], []))
                        d_list.append(idx)
                        tf_list.append(tf)
            else:
                raise ValueError("BM25CorpusIndex requires tokenized_corpus or candidate_dir")

            terms = sorted(postings)
            offsets = np.zeros(len(terms) + 1, dtype=np.int64)
            total = sum(len(postings[t][0]) for t in terms)
            doc_indices = np.empty(total, dtype=np.int32)
            tfs = np.empty(total, dtype=np.int16)
            pos = 0
            for i, term in enumerate(terms):
                d_list, tf_list = postings[term]
                n = len(d_list)
                doc_indices[pos : pos + n] = d_list
                tfs[pos : pos + n] = np.clip(tf_list, 1, 32767)
                pos += n
                offsets[i + 1] = pos

            self.pids = np.array(pids, dtype=np.int64)
            self.doc_lens = np.array(doc_lens, dtype=np.float32)
            self.offsets, self.doc_indices, self.tfs = offsets, doc_indices, tfs
            self.vocab = {term: i for i, term in enumerate(terms)}
            if len(pids) >= 10000:
                try:
                    csr_dir.mkdir(parents=True, exist_ok=True)
                    np.save(csr_dir / "pids.npy", self.pids)
                    np.save(csr_dir / "doc_lens.npy", self.doc_lens.astype(np.int32))
                    np.save(csr_dir / "offsets.npy", self.offsets)
                    np.save(csr_dir / "doc_indices.npy", self.doc_indices)
                    np.save(csr_dir / "tfs.npy", self.tfs)
                    atomic_write_json(csr_dir / "vocab.json", self.vocab)
                except OSError:
                    pass

        self.pid_to_idx = {int(pid): idx for idx, pid in enumerate(self.pids.tolist())}
        avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 1.0
        self.norm = self.k1 * (1.0 - self.b + self.b * (self.doc_lens / max(avgdl, 1e-9)))

    def search(self, query_text, *, top_k=1000, candidate_pids=None, tie_break="pid_asc"):
        n_docs = len(self.pids)
        scores = np.zeros(n_docs, dtype=np.float32)
        for term, q_tf in Counter(self.tokenize(query_text)).items():
            term_idx = self.vocab.get(term)
            if term_idx is None:
                continue
            start, end = int(self.offsets[term_idx]), int(self.offsets[term_idx + 1])
            df = end - start
            if df <= 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            d_arr = self.doc_indices[start:end]
            tf_arr = self.tfs[start:end].astype(np.float32)
            scores[d_arr] += float(idf * q_tf) * (tf_arr * (self.k1 + 1.0)) / (tf_arr + self.norm[d_arr])

        if candidate_pids is not None:
            pool_scores = [float(scores[self.pid_to_idx[pid]]) for pid in candidate_pids]
            return sort_scored_candidates(
                candidate_pids, pool_scores, tie_break=tie_break, top_k=min(top_k, len(candidate_pids)), tag="llm_bm25",
            )
        if top_k < n_docs:
            part = np.argpartition(-scores, top_k)[:top_k]
            sub_scores, sub_pids = scores[part], self.pids[part]
        else:
            sub_scores, sub_pids = scores, self.pids
        order = np.lexsort((sub_pids, -sub_scores))[:top_k]
        return [(rank, int(sub_pids[pos]), float(sub_scores[pos]), "llm_bm25") for rank, pos in enumerate(order, 1)]


def _load_local_encoder(model_cfg):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from lexverse.providers.local import resolve_source

    preferred = model_cfg.load_parameters.get("device_map")
    device = preferred if preferred not in {None, "auto", "cuda"} else ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0" and torch.cuda.is_available():
        best_idx = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
        device = f"cuda:{best_idx}"
    model_dir = str(resolve_source(model_cfg)[0])
    trust = bool(model_cfg.load_parameters.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=trust)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=trust).to(device).eval()
    return tokenizer, model, device


class EncoderRetriever:
    def __init__(self, config, paths):
        self.bench = config.benchmark
        self.paths = paths
        self.model_cfg = config.models.default
        self.pooling = self.bench.get("pooling", "cls")
        self.normalize = bool(self.bench.get("normalize", False))
        self.max_length = int(self.bench.get("max_length", 512))
        self.batch_size = int(self.bench.get("batch_size", 32 if self.max_length > 512 else 128))
        self.doc_field = self.bench.get("document_field", "fact")
        self.padding_mode = self.bench.get("padding", "max_length" if self.pooling == "cls" else "longest")
        self.use_amp = bool(self.bench.get("amp", self.max_length > 512))
        self._tokenizer = self._model = self._device = None
        self._doc_vecs = {}

    def encode_texts(self, texts):
        import torch
        import torch.nn.functional as F

        if self._model is None:
            self._tokenizer, self._model, self._device = _load_local_encoder(self.model_cfg)
        out = []
        padding = "max_length" if self.padding_mode == "max_length" else True
        use_cuda_amp = self.use_amp and self._device.startswith("cuda")
        for i in range(0, len(texts), self.batch_size):
            inp = self._tokenizer(
                texts[i : i + self.batch_size], padding=padding, max_length=self.max_length,
                truncation=True, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda_amp):
                hidden = self._model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])["last_hidden_state"].float()
                if self.pooling == "cls":
                    reps = hidden[:, 0, :]
                else:
                    mask = inp["attention_mask"].unsqueeze(-1).float()
                    reps = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                if self.normalize:
                    reps = F.normalize(reps, p=2, dim=-1)
            out.append(reps.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty((0, 768), dtype=np.float32)

    def score_query(self, query_text, candidate_pids):
        missing = sorted(set(candidate_pids) - set(self._doc_vecs))
        if missing:
            cdir = self.paths["candidate_dir"]
            texts = [str(json.loads((cdir / f"{pid}.json").read_text(encoding="utf-8")).get(self.doc_field, "")) for pid in missing]
            for pid, vec in zip(missing, self.encode_texts(texts)):
                self._doc_vecs[pid] = vec
        q_vec = self.encode_texts([query_text])[0]
        d_mat = np.stack([self._doc_vecs[pid] for pid in candidate_pids], axis=0)
        return (d_mat @ q_vec).astype(np.float64).tolist()


def _keller_valid_subfact(text):
    return (
        text.split("：", 1)[0].endswith("起因、经过、结果分别为")
        and not all(i in text for i in ("起因：无", "经过：无", "结果：无"))
        and ("抱歉" not in text)
    )


class KellerSubfactRetriever:
    def __init__(self, config, paths):
        self.bench = config.benchmark
        self.model_cfg = config.models.default
        self.max_crime_num = int(self.bench.get("max_crime_num", 4))
        self.max_length = int(self.bench.get("max_length", 512))
        self.normalize = bool(self.bench.get("normalize", False))
        self.batch_size = int(self.bench.get("batch_size", 256))
        self.qf_raw = json.loads(paths["keller_query_fact"].read_text(encoding="utf-8"))
        self.df_raw = json.loads(paths["keller_doc_fact"].read_text(encoding="utf-8"))
        self._tokenizer = self._model = self._device = None
        self._str_vecs = {}

    def _encode_unique(self, strings):
        import torch
        import torch.nn.functional as F

        missing = [s for s in dict.fromkeys(strings) if s not in self._str_vecs]
        if not missing:
            return
        if self._model is None:
            self._tokenizer, self._model, self._device = _load_local_encoder(self.model_cfg)
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]
            inp = self._tokenizer(
                batch, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                reps = self._model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])["last_hidden_state"][:, 0, :].float()
                if self.normalize:
                    reps = F.normalize(reps, p=2, dim=-1)
                for s, vec in zip(batch, reps.cpu().numpy()):
                    self._str_vecs[s] = vec

    def _build_slots(self, raw_dict):
        items = [(k, v.split("：", 1)[-1]) for k, v in raw_dict.items() if _keller_valid_subfact(v)]
        slots, mask = [""] * self.max_crime_num, [0.0] * self.max_crime_num
        for i in range(min(self.max_crime_num, len(items))):
            slots[i] = f"{items[i][0]}的相关情节：{items[i][1]}"
            mask[i] = 1.0
        return slots, mask

    def score_query(self, qid, candidate_pids):
        q_slots, q_mask = self._build_slots(self.qf_raw.get(str(qid), {}))
        d_slots_by_pid, d_masks_by_pid = {}, {}
        all_strs = list(q_slots)
        for pid in candidate_pids:
            d_slots, d_mask = self._build_slots(self.df_raw.get(str(pid), {}))
            d_slots_by_pid[pid], d_masks_by_pid[pid] = d_slots, d_mask
            all_strs.extend(d_slots)
        self._encode_unique(all_strs)

        q_vecs = np.stack([self._str_vecs[s] for s in q_slots], axis=0)
        q_mask_arr = np.array(q_mask, dtype=np.float32)
        q_denom = float(sum(q_mask_arr))
        scores = []
        for pid in candidate_pids:
            if q_denom <= 0:
                scores.append(0.0)
                continue
            d_vecs = np.stack([self._str_vecs[s] for s in d_slots_by_pid[pid]], axis=0)
            sim = q_vecs @ d_vecs.T
            sim = np.where(np.array(d_masks_by_pid[pid], dtype=bool)[None, :], sim, -1e4)
            scores.append(float((sim.max(axis=1) * q_mask_arr).sum() / q_denom))
        return "\n".join(s for s, m in zip(q_slots, q_mask) if m > 0), scores


class LeCaRDv2Environment(ExecutionEnvironment):
    name = "lecardv2_retrieval"

    def __init__(self, config, provider=None):
        self.config = config
        self.provider = provider
        self.bench = config.benchmark
        self.mode = self.bench.get("mode", "ranked_submission")
        self.scope = self.bench.get("candidate_scope", "full_corpus")
        self.top_k = self.bench.get("top_k", 1000 if self.scope == "full_corpus" else 40)
        self.tie_break = self.bench.get(
            "tie_break",
            "str_desc" if self.scope == "keller_pool" else ("did_desc" if self.scope == "labeled_pool" else "pid_asc"),
        )
        self.offline = bool(config.generation.get("offline", False))
        self.paths = resolve_data_layout(self.bench, offline=self.offline)
        self.candidate_pids = load_candidate_pids(self.paths)
        self.allowed_pids = set(self.candidate_pids)
        self._submission_cache = self._keller_bm25_cache = self._bm25_index = self._encoder = self._keller = None

    async def prepare(self, context=None):
        del context
        if self.mode == "ranked_submission" and self._submission_cache is None:
            sub_path = Path(self.bench["submission_path"]).expanduser().resolve()
            self._submission_cache = parse_trec_submission(
                sub_path, self.allowed_pids, top_k=self.top_k if self.scope == "full_corpus" else None,
            )
        elif self.mode == "encoder_retrieval" and self._encoder is None:
            self._encoder = EncoderRetriever(self.config, self.paths)
        elif self.mode == "keller_subfact_retrieval" and self._keller is None:
            self._keller = KellerSubfactRetriever(self.config, self.paths)
        elif self.mode in {"bm25_retrieval", "llm_query_rewrite"}:
            if (
                self.mode == "bm25_retrieval" and self.scope == "keller_pool"
                and "keller_bm25" in self.paths and not self.bench.get("recompute_bm25", False)
            ):
                if self._keller_bm25_cache is None:
                    self._keller_bm25_cache = json.loads(self.paths["keller_bm25"].read_text(encoding="utf-8"))
            elif self._bm25_index is None:
                self._bm25_index = BM25CorpusIndex(
                    self.paths, document_field=self.bench.get("document_field", "qw"),
                    k1=float(self.bench.get("bm25_k1", 1.2)), b=float(self.bench.get("bm25_b", 0.75)),
                )

    def _write_result(self, work_dir, qid, ranked, *, answer=None, extra_meta=None, extra_state=None, dialog_history=None):
        trec_path = work_dir / "ranking.trec"
        trec_path.write_text(
            "".join(f"{qid} Q0 {pid} {rank} {score:.9g} {tag}\n" for rank, pid, score, tag in ranked),
            encoding="utf-8",
        )
        top_10 = [pid for _, pid, _, _ in ranked[:10]]
        meta = {
            "qid": qid, "mode": self.mode, "candidate_scope": self.scope,
            "top_k": len(ranked), "top_10_pids": top_10,
            "ranking_sha256": hashlib.sha256(trec_path.read_bytes()).hexdigest(),
            **(extra_meta or {}),
        }
        atomic_write_json(work_dir / "ranking.json", meta)
        return EnvironmentResult(
            status="completed",
            answer=answer or ",".join(str(pid) for pid in top_10),
            dialog_history=dialog_history or [],
            final_state={"mode": self.mode, "qid": qid, "top_k": len(ranked), **(extra_state or {})},
            artifacts={"ranking_trec": str(trec_path), "ranking_meta": str(work_dir / "ranking.json")},
        )

    async def run(self, task, work_dir):
        if task.source.task_type != TASK_TYPE or task.metadata.get("protocol") != PROTOCOL:
            raise ValueError("incompatible LeCaRDv2 task protocol")
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        qid = identifier(task.raw_payload["qid"])
        fact = str(task.raw_payload["fact"])
        pool_pids = task.raw_payload.get("candidate_pids")
        target_pids = pool_pids if pool_pids is not None else self.candidate_pids

        if self.mode == "ranked_submission":
            if qid not in self._submission_cache:
                raise ValueError(f"query {qid} missing from submission TREC file")
            return self._write_result(work_dir, qid, self._submission_cache[qid])

        if self.mode == "bm25_retrieval":
            if self._keller_bm25_cache is not None and pool_pids is not None and str(qid) in self._keller_bm25_cache:
                q_scores = self._keller_bm25_cache[str(qid)]
                scores = [float(q_scores.get(str(pid), 0.0)) for pid in pool_pids]
                ranked = sort_scored_candidates(
                    pool_pids, scores, tie_break=self.tie_break, top_k=min(self.top_k, len(pool_pids)), tag="bm25",
                )
            else:
                ranked = [
                    (r, p, s, "bm25")
                    for r, p, s, _ in self._bm25_index.search(fact, top_k=self.top_k, candidate_pids=pool_pids, tie_break=self.tie_break)
                ]
            return self._write_result(work_dir, qid, ranked)

        if self.mode == "encoder_retrieval":
            scores = self._encoder.score_query(fact, target_pids)
            ranked = sort_scored_candidates(
                target_pids, scores, tie_break=self.tie_break, top_k=min(self.top_k, len(target_pids)), tag="encoder_zs",
            )
            return self._write_result(work_dir, qid, ranked)

        if self.mode == "keller_subfact_retrieval":
            rewrite_summary, scores = self._keller.score_query(qid, target_pids)
            ranked = sort_scored_candidates(
                target_pids, scores, tie_break=self.tie_break, top_k=min(self.top_k, len(target_pids)), tag="keller_zs",
            )
            tpl = self.bench.get("prompt_template", "keller_extract")
            return self._write_result(
                work_dir, qid, ranked, answer=rewrite_summary,
                extra_meta={"prompt_template": tpl, "prompt_paper": PROMPT_TEMPLATES.get(tpl, {}).get("paper"), "rewritten_query": rewrite_summary},
                extra_state={"prompt_template": tpl},
            )

        if self.provider is None:
            raise ValueError("LeCaRDv2 llm_query_rewrite mode requires a model provider")
        messages = list(task.input.messages) or [{"role": "user", "content": task.input.prompt}]
        response = await self.provider.generate(messages)
        rewrite_text = (response.content or "").strip()
        if not rewrite_text:
            raise ValueError(f"model returned empty query rewrite for query {qid}")
        ranked = self._bm25_index.search(
            f"{fact}\n{rewrite_text}", top_k=self.top_k, candidate_pids=pool_pids, tie_break=self.tie_break,
        )
        tpl = self.bench.get("prompt_template", "when_rules_learn_human1")
        return self._write_result(
            work_dir, qid, ranked, answer=rewrite_text,
            extra_meta={
                "prompt_template": tpl, "prompt_paper": PROMPT_TEMPLATES.get(tpl, {}).get("paper"),
                "original_fact": fact, "rewritten_query": rewrite_text,
            },
            extra_state={"prompt_template": tpl},
            dialog_history=[*messages, {"role": "assistant", "content": rewrite_text, "metadata": dict(response.raw or {})}],
        )
