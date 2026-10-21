"""Safe FTS MATCH construction: punctuation never becomes query syntax."""
import re
from app.indexing.sqlite_index import SQLiteIndex


def fts_query(query: str) -> str:
    tokens = list(dict.fromkeys(re.findall(r"[^\W_]+", query.lower(), flags=re.UNICODE)))[:64]
    return " OR ".join('"' + token + '"' for token in tokens)


def retrieve(index: SQLiteIndex, query: str, limit: int) -> list[dict]:
    expression = fts_query(query)
    if not expression:
        return []
    with index.connect() as db:
        return [dict(row) for row in db.execute("""
            SELECT p.*, bm25(passages_fts) AS bm25_score
            FROM passages_fts JOIN passages p ON p.rowid=passages_fts.rowid
            WHERE passages_fts MATCH ? ORDER BY bm25_score ASC, p.passage_id ASC LIMIT ?
        """, (expression, limit))]
