#!/usr/bin/env bash
set -euo pipefail
python /tests/grade.py /workspace/complaint_preparation.docx /logs/verifier
python /tests/judge.py /workspace/complaint_preparation.docx /inputs /logs/verifier
