from typer.testing import CliRunner
from app.cli import app


def test_cli_help_and_lexical_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEARCH_EXPANSION_ENABLED", "false")
    monkeypatch.setenv("SEARCH_RERANK_ENABLED", "false")
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Idempotency keys prevent duplicate orders", encoding="utf-8")
    result = runner.invoke(app, ["index", "--docs", str(docs)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["search", "idempotency", "--top-k", "1"])
    assert result.exit_code == 0, result.output
    assert '"rerank_score": null' in result.output
