#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier

python3 - <<'PY'
import json
from pathlib import Path

memo_path = Path("/artifacts/review_memo.md")
if not memo_path.is_file():
    payload = {
        "reward": 0.0,
        "all_pass_score": 0.0,
        "passed_criteria": 0.0,
        "total_criteria": 4.0,
    }
    Path("/logs/verifier/reward.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path("/logs/verifier/reward.txt").write_text("0.0\n", encoding="utf-8")
    raise SystemExit(0)

text = memo_path.read_text(encoding="utf-8").lower()
checks = [
    ("uncapped" in text or "unlimited" in text) and "12" in text,
    "30" in text and ("notice" in text or "cure" in text),
    "new york" in text and "aaa" in text,
    "section 4" in text and "section 5" in text,
]
passed = sum(1 for c in checks if c)
total = len(checks)
rate = round(passed / total, 4)
all_pass = 1.0 if passed == total else 0.0

payload = {
    "reward": rate,
    "all_pass_score": all_pass,
    "passed_criteria": float(passed),
    "total_criteria": float(total),
}
Path("/logs/verifier/reward.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
Path("/logs/verifier/reward.txt").write_text(f"{rate}\n", encoding="utf-8")
PY
