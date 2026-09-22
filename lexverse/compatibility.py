"""Report dependency conflicts without mutating the environment.

``rouge`` and ``rouge_chinese`` share an import path but tokenize differently,
so a mixed installation cannot exactly reproduce both upstream score sets.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class RougeState:
    rouge_installed: bool
    rouge_chinese_installed: bool
    rouge_provides: str | None      # __module__ of rouge.Rouge, or None
    conflicting: bool               # both installed and rouge_is_chinese


def rouge_state() -> RougeState:
    rouge_installed = importlib.util.find_spec("rouge") is not None
    rouge_chinese_installed = importlib.util.find_spec("rouge_chinese") is not None
    provides: str | None = None
    if rouge_installed:
        try:
            from rouge import Rouge  # type: ignore
            provides = getattr(Rouge, "__module__", None)
        except Exception:
            provides = None
    conflicting = (
        rouge_installed
        and rouge_chinese_installed
        and provides is not None
        and "rouge_chinese" in provides
    )
    return RougeState(
        rouge_installed=rouge_installed,
        rouge_chinese_installed=rouge_chinese_installed,
        rouge_provides=provides,
        conflicting=conflicting,
    )


def rouge_warning() -> str | None:
    st = rouge_state()
    if st.conflicting:
        return (
            "rouge_chinese shadows plain rouge: `from rouge import Rouge` "
            f"resolves to {st.rouge_provides!r}. LexEval Rouge_L and LawBench "
            "tasks 1-1/3-8/2-4/2-8 cannot both be exact in this environment. "
            "Install lexverse[lexeval] and lexverse[lawbench] in separate "
            "virtualenvs for exact parity. Scores here are still computed by "
            "the upstream evaluator — only the tokenizer differs."
        )
    return None
