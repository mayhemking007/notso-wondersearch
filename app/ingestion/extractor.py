"""Extraction returns page-scoped text; OCR is deliberately out of scope."""
from pathlib import Path
import re

SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}


def normalize(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    return "\n\n".join(
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    )


def extract(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        import pymupdf
        with pymupdf.open(path) as document:
            return [(i + 1, normalize(page.get_text())) for i, page in enumerate(document)]
    return [(None, normalize(path.read_text(encoding="utf-8-sig")))]
