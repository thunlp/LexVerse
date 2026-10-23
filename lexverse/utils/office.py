"""Structural validation for Office document files."""

from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile
from zlib import error as ZlibError


def validate_office_artifact(path: Path) -> None:
    parts = {
        "docx": ("word/document.xml", "wordprocessingml", "document", "document"),
        "xlsx": ("xl/workbook.xml", "spreadsheetml", "workbook", "sheet"),
        "pptx": ("ppt/presentation.xml", "presentationml", "presentation", "presentation"),
    }
    part, namespace, root, kind = parts[path.suffix.lstrip(".").lower()]
    try:
        with ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Corrupt ZIP member")
            # Parse every XML part so a valid ZIP with broken document XML fails too.
            xml = {name: ElementTree.fromstring(archive.read(name))
                   for name in archive.namelist() if name.endswith((".xml", ".rels"))}
            types = xml["[Content_Types].xml"]
            content_type = f"application/vnd.openxmlformats-officedocument.{namespace}.{kind}.main+xml"
            if not any(node.get("PartName") == "/" + part and node.get("ContentType") == content_type
                       for node in types.findall("{http://schemas.openxmlformats.org/package/2006/content-types}Override")):
                raise ValueError("Office document type does not match its extension")
            if not any(node.get("Type", "").endswith("/officeDocument")
                       and node.get("Target", "").lstrip("/") == part
                       and node.get("TargetMode", "Internal") == "Internal"
                       for node in xml["_rels/.rels"]):
                raise ValueError("Missing Office document relationship")
            if xml[part].tag not in {
                f"{{http://schemas.openxmlformats.org/{namespace}/2006/main}}{root}",
                f"{{http://purl.oclc.org/ooxml/{namespace}/main}}{root}",
            }:
                raise ValueError("Invalid Office document root")
    except (BadZipFile, KeyError, ElementTree.ParseError, RuntimeError, NotImplementedError,
            ZlibError, ValueError) as exc:
        raise ValueError(f"Invalid Office artifact: {path.name}") from exc
