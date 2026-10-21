"""SQLite metadata and ordinary lexical postings. No model imports here."""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import logging
import os
import sqlite3
from collections.abc import Iterable, Iterator

from app.ingestion.extractor import SUPPORTED, extract
from app.ingestion.chunker import chunk

log = logging.getLogger(__name__)


def stable_id(value: str) -> str:
    return sha256(value.encode()).hexdigest()


class SQLiteIndex:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY, root TEXT NOT NULL,
                    source_path TEXT NOT NULL, fingerprint TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS passages (
                    rowid INTEGER PRIMARY KEY, passage_id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                    source_path TEXT NOT NULL, page_number INTEGER, passage_text TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS passages_document ON passages(document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
                    passage_text, content='passages', content_rowid='rowid',
                    tokenize='unicode61');
                CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages BEGIN
                    INSERT INTO passages_fts(rowid,passage_text) VALUES(new.rowid,new.passage_text);
                END;
                CREATE TRIGGER IF NOT EXISTS passages_ad AFTER DELETE ON passages BEGIN
                    INSERT INTO passages_fts(passages_fts,rowid,passage_text)
                    VALUES('delete',old.rowid,old.passage_text);
                END;
                CREATE TABLE IF NOT EXISTS expansion_cache (
                    cache_key TEXT PRIMARY KEY, expansions TEXT NOT NULL);
            """)
            if db.execute("PRAGMA user_version").fetchone()[0] < 1:
                self._migrate_path_ids(db)
                db.execute("PRAGMA user_version=1")

    @staticmethod
    def _migrate_path_ids(db: sqlite3.Connection) -> None:
        """Merge legacy Windows path spellings without touching benchmark IDs."""
        groups: dict[str, list[dict]] = {}
        for row in db.execute("SELECT * FROM documents"):
            document = dict(row)
            root, source = Path(row["root"]), Path(row["source_path"])
            if not root.is_absolute() or not source.is_absolute():
                continue
            root, source = root.resolve(), source.resolve()
            try:
                relative = source.relative_to(root)
            except ValueError:
                continue
            root_key = os.path.normcase(str(root))
            identity = stable_id(f"{root_key}:{os.path.normcase(relative.as_posix())}")
            document.update(canonical_id=identity, canonical_root=root_key)
            groups.setdefault(identity, []).append(document)
        for identity, variants in groups.items():
            winner = next((d for d in variants if d["document_id"] == identity), variants[0])
            if len(variants) == 1 and winner["document_id"] == identity and winner["root"] == winner["canonical_root"]:
                continue
            passages = [(r["page_number"], r["passage_text"]) for r in db.execute(
                "SELECT page_number,passage_text FROM passages WHERE document_id=? ORDER BY rowid",
                (winner["document_id"],))]
            db.executemany("DELETE FROM documents WHERE document_id=?", [(d["document_id"],) for d in variants])
            db.execute("INSERT INTO documents VALUES(?,?,?,?)",
                       (identity, winner["canonical_root"], winner["source_path"], winner["fingerprint"]))
            db.executemany(
                "INSERT INTO passages(passage_id,document_id,source_path,page_number,passage_text) VALUES(?,?,?,?,?)",
                [(stable_id(f"{identity}:{i}"), identity, winner["source_path"], page, text)
                 for i, (page, text) in enumerate(passages)])

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def replace_document(self, document_id: str, root: str, source: str,
                         fingerprint: str, passages: list[tuple[int | None, str]]) -> None:
        self.replace_documents([(document_id, root, source, fingerprint, passages)])

    def replace_documents(self, documents: Iterable[tuple[str, str, str, str, list[tuple[int | None, str]]]]) -> None:
        """Atomically write one or many documents, keeping official benchmark IDs."""
        with self.connect() as db:
            for document_id, root, source, fingerprint, passages in documents:
                db.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
                db.execute("INSERT INTO documents VALUES(?,?,?,?)", (document_id, root, source, fingerprint))
                db.executemany(
                    "INSERT INTO passages(passage_id,document_id,source_path,page_number,passage_text) VALUES(?,?,?,?,?)",
                    [(stable_id(f"{document_id}:{i}"), document_id, source, page, text)
                     for i, (page, text) in enumerate(passages)],
                )

    def index_directory(self, directory: str | Path, words: int = 200, overlap: int = 30) -> dict:
        started = perf_counter()
        root = Path(directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Document path must be a directory")
        # Raising on traversal errors prevents destructive pruning after an incomplete scan.
        root_key = os.path.normcase(str(root))
        def traversal_error(error: OSError) -> None:
            raise error
        files = []
        for parent, _, names in os.walk(root, onerror=traversal_error, followlinks=False):
            for name in names:
                path = Path(parent) / name
                if path.suffix.lower() in SUPPORTED and not path.is_symlink():
                    files.append(path)
        with self.connect() as db:
            existing = {r["document_id"]: r["fingerprint"] for r in db.execute(
                "SELECT document_id,fingerprint FROM documents WHERE root=?", (root_key,))}
        seen: set[str] = set()
        indexed = passages_count = skipped = 0
        errors = []
        for path in sorted(files):
            relative_key = os.path.normcase(path.relative_to(root).as_posix())
            doc_id = stable_id(f"{root_key}:{relative_key}")
            seen.add(doc_id)
            try:
                fingerprint = stable_id(f"v1:{words}:{overlap}:" + sha256(path.read_bytes()).hexdigest())
                if existing.get(doc_id) == fingerprint:
                    skipped += 1
                    continue
                passages = [(page, part) for page, text in extract(path)
                            for part in chunk(text, words, overlap)]
                self.replace_document(doc_id, root_key, str(path), fingerprint, passages)
                indexed += 1
                passages_count += len(passages)
            except Exception as exc:
                log.warning("Could not index %s: %s", path, exc)
                errors.append({"source": str(path), "error": str(exc)})
        deleted = set(existing) - seen
        with self.connect() as db:
            db.executemany("DELETE FROM documents WHERE document_id=?", [(x,) for x in deleted])
        return {"documents_indexed": indexed, "passages_indexed": passages_count,
                "documents_skipped": skipped, "documents_deleted": len(deleted),
                "errors": errors, "elapsed_ms": round((perf_counter() - started) * 1000, 2)}

    def counts(self) -> dict:
        with self.connect() as db:
            return {name: db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                    for name in ("documents", "passages")}

    def cached_expansions(self, key: str) -> list | None:
        with self.connect() as db:
            row = db.execute("SELECT expansions FROM expansion_cache WHERE cache_key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def cache_expansions(self, key: str, expansions: list) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO expansion_cache VALUES(?,?)", (key, json.dumps(expansions)))
