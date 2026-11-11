"""Check the submission contract only; this is not a legal-quality score."""
import json
import re
import sys
from pathlib import Path
from docx import Document
from docx.table import Table

HEADINGS = ("案件与金额梳理", "诉讼请求与风险分析", "民事起诉状草稿", "材料目录与证明目的")
NUMBER = r"[0-9０-９一二三四五六七八九十百]+"
PREFIX = re.compile(
    rf"^(?:第{NUMBER}[章节部分]+\s*|[（(]{NUMBER}[）)]\s*|{NUMBER}[.．、)）]\s*|{NUMBER}\s+)"
)


def heading_name(text):
    return PREFIX.sub("", text.strip(), count=1).rstrip("：:").strip()


def heading_level(paragraph):
    # Word's built-in heading names are stored in English even in Chinese Word.
    match = re.fullmatch(r"Heading ([1-9])", paragraph.style.name)
    return int(match[1]) if match else None


def check(path):
    if not path.is_file():
        return {"docx_readable": False}
    try:
        doc = Document(path)
    except Exception:
        return {"docx_readable": False}
    checks = {"docx_readable": True, **dict.fromkeys(HEADINGS, False)}
    current = None
    section_level = 1
    for paragraph in doc.iter_inner_content():
        if isinstance(paragraph, Table):
            if current and any(len(cell.text.strip()) >= 20 for row in paragraph.rows for cell in row.cells):
                checks[current] = True
            continue
        text = paragraph.text.strip()
        level = heading_level(paragraph)
        heading = heading_name(text)
        if level is not None:
            heading = re.split("[：:]", heading, maxsplit=1)[0].strip()
        if heading in HEADINGS:
            current = heading
            section_level = level or 1
        elif level is not None:
            if level <= section_level:
                current = None
        elif current and len(text) >= 20:
            checks[current] = True
    return checks


if __name__ == "__main__":
    checks = check(Path(sys.argv[1]))
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.json").write_text(json.dumps({"submission_valid": int(all(checks.values()))}))
    (output / "checks.json").write_text(json.dumps({
        "scope": "仅验证DOCX与章节基本内容，不评价法律正确性或完整性", "checks": checks
    }, ensure_ascii=False, indent=2))
