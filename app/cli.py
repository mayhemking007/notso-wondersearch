"""Run with python -m app.cli."""
import json
import logging
import typer
from app.retrieval.search import SearchService

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def emit(value: dict) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


@app.command()
def index(docs: str = typer.Option(..., "--docs")) -> None:
    """Index local PDF, text, and Markdown documents without neural inference."""
    emit(SearchService().index_directory(docs))


@app.command()
def search(query: str, top_k: int = typer.Option(5, "--top-k")) -> None:
    """Search with query-time expansion and optional reranking."""
    emit(SearchService().search(query, top_k))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the local HTTP API."""
    import uvicorn
    uvicorn.run("app.main:app", host=host, port=port, workers=1)


@app.command()
def benchmark(dataset: str = "scifact", limit: int = 30) -> None:
    """Evaluate the shared pipeline on official SciFact test queries."""
    if dataset != "scifact":
        raise typer.BadParameter("Only scifact is supported")
    from app.benchmark.scifact import run
    emit(run(limit=limit))


@app.command("download-models")
def download_models() -> None:
    """Download and validate configured models before going offline."""
    service = SearchService()
    if service.settings.expansion_enabled:
        service.expander._load()
    if service.settings.rerank_enabled:
        service.reranker.rank("warmup", [{"passage_text": "A sample passage.", "passage_id": "warmup"}])
    emit(service.health())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        app()
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc
