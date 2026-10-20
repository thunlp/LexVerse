#!/usr/bin/env bash
set -euo pipefail

mkdir -p /artifacts

cat <<'EOF' > /artifacts/review_memo.md
# Legal Risk Review Memorandum: Technology Services Agreement

## 1. Liability Cap & Consequential Damages (Section 4)
- **Issue Identified**: Section 4 imposes a severe asymmetry: Provider's liability is capped at 1 month of fees, whereas Customer faces **unlimited and uncapped** liability, including for indirect and consequential damages.
- **Recommendation**: Establish a **mutual** limitation of liability capped at **12 months of fees** paid or payable, with a mutual waiver of indirect, special, and consequential damages.

## 2. Unilateral Termination Without Notice (Section 5)
- **Issue Identified**: Provider can terminate immediately for convenience without refund, while Customer is locked in for 24 months with no termination right.
- **Recommendation**: Require at least **30 days' prior written notice** for termination, add a **cure period** (e.g., 30 days) for material breach, and require a pro-rata refund of any unearned prepaid fees.

## 3. Governing Law & Dispute Resolution (Section 6)
- **Summary**: Governed by **New York** law with binding arbitration before the **AAA** in New York, NY.
- **Conclusion**: Do not execute in current form; revise Sections 4 and 5 before signing.
EOF
