"""Deterministic overlapping windows preserving paragraph breaks."""
import re


def chunk(text: str, words: int = 200, overlap: int = 30) -> list[str]:
    if words < 1 or not 0 <= overlap < words:
        raise ValueError("Require 0 <= overlap < words")
    tokens = list(re.finditer(r"\S+", text))
    passages = []
    start = 0
    while start < len(tokens):
        end = min(start + words, len(tokens))
        passages.append(text[tokens[start].start():tokens[end - 1].end()])
        if end == len(tokens):
            break
        start = end - overlap
    return passages
