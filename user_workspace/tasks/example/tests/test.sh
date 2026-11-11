#!/usr/bin/env bash
set -euo pipefail
python /tests/grade.py /work/complaint_preparation.docx /logs/verifier
python /tests/judge.py /work/complaint_preparation.docx /documents /logs/verifier
