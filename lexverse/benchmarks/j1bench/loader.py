"""J1Bench data loader.

Loads J1-Eval case files (one JSON per line) and can produce single-case
slices — a subprocess-friendly form every upstream scenario reads via
`--case_database`. The upstream `remove_processed_cases()` logic is
per-file, so per-Trial slicing keeps cases isolated.

The dataset is external to the LexVerse repository and distribution. Callers
must explicitly provide a downloaded J1-Eval JSONL path. The canonical source
is the gated Hugging Face dataset `CimoInkPool/J1-Eval_Dataset`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexverse.runtime.errors import TrialError


J1_EVAL_DATASET_URL = (
    "https://huggingface.co/datasets/CimoInkPool/J1-Eval_Dataset/tree/main"
)


@dataclass(frozen=True)
class J1BenchCase:
    """One case record from a J1-Eval scenario file (whole JSON kept opaque).

    Different scenarios have different top-level keys; we only require
    `id` for identification and `raw` for round-trip serialization.
    """

    scenario: str
    id: str
    raw: dict[str, Any] = field(repr=False)


class J1BenchLoader:
    def __init__(self, case_database: Path | None = None, scenario: str = "CI") -> None:
        self.scenario = scenario
        if case_database is None:
            raise TrialError(
                "J1Bench requires benchmark.case_database to point to an explicitly "
                f"downloaded J1-Eval_{scenario}.jsonl; source: {J1_EVAL_DATASET_URL}"
            )
        self.case_database = Path(case_database)

    def load_all(self) -> list[J1BenchCase]:
        if not self.case_database.exists():
            raise TrialError(
                f"J1Bench case database not found: {self.case_database}"
            )
        cases: list[J1BenchCase] = []
        with self.case_database.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrialError(
                        f"invalid JSON at {self.case_database}:{line_no}: {exc}"
                    ) from exc
                if "id" not in obj:
                    raise TrialError(
                        f"case missing 'id' at {self.case_database}:{line_no}"
                    )
                cases.append(J1BenchCase(
                    scenario=self.scenario, id=str(obj["id"]), raw=obj
                ))
        return cases

    def by_id(self, case_id: str) -> J1BenchCase:
        for case in self.load_all():
            if case.id == case_id:
                return case
        raise TrialError(
            f"case {case_id!r} not found in {self.case_database}"
        )

    def write_single_case_jsonl(self, case: J1BenchCase, out_path: Path) -> Path:
        """Emit a single-case JSONL for upstream `--case_database`.

        Upstream reads the whole file, so one Trial gets one file with
        exactly one case; that keeps parallel Trials isolated.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps(case.raw, ensure_ascii=False) + "\n")
        return out_path
