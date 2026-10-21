"""Environment configuration shared by CLI and API."""
from dataclasses import dataclass, field
from pathlib import Path
import os


def flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SEARCH_DATA_DIR", "data")))
    expansion_enabled: bool = field(default_factory=lambda: flag("SEARCH_EXPANSION_ENABLED", True))
    rerank_enabled: bool = field(default_factory=lambda: flag("SEARCH_RERANK_ENABLED", True))
    offline: bool = field(default_factory=lambda: flag("SEARCH_OFFLINE", False))
    expansion_model: str = field(default_factory=lambda: os.getenv("SEARCH_EXPANSION_MODEL", "naver/splade-cocondenser-ensembledistil"))
    expansion_revision: str = field(default_factory=lambda: os.getenv("SEARCH_EXPANSION_REVISION", "main"))
    rerank_model: str = field(default_factory=lambda: os.getenv("SEARCH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    rerank_revision: str = field(default_factory=lambda: os.getenv("SEARCH_RERANK_REVISION", "main"))
    expansion_limit: int = field(default_factory=lambda: int(os.getenv("SEARCH_EXPANSION_LIMIT", "3")))
    candidates: int = field(default_factory=lambda: int(os.getenv("SEARCH_CANDIDATES", "20")))
    max_length: int = field(default_factory=lambda: int(os.getenv("SEARCH_MAX_LENGTH", "512")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("SEARCH_BATCH_SIZE", "8")))
    torch_threads: int = field(default_factory=lambda: int(os.getenv("SEARCH_TORCH_THREADS", "4")))
    chunk_words: int = 200
    overlap_words: int = 30
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if not 0 <= self.expansion_limit <= 8:
            raise ValueError("Expansion limit must be between 0 and 8")
        if not 1 <= self.candidates <= 200 or not 16 <= self.max_length <= 512:
            raise ValueError("Candidates must be 1–200; maximum sequence length 16–512")
        if self.batch_size < 1 or self.torch_threads < 1:
            raise ValueError("Batch size and CPU threads must be positive")
        if not 0 <= self.overlap_words < self.chunk_words or self.rrf_k < 1:
            raise ValueError("Invalid chunking or RRF settings")

    @property
    def model_cache(self) -> Path:
        return self.data_dir / "models"
