from pathlib import Path

from fuel.cli import main


def test_download_model_command(monkeypatch, capsys, tmp_path):
    called = {}

    def fake_download(model_path=None, force=False):
        called["model_path"] = model_path
        return Path(model_path or "models/test.gguf")

    monkeypatch.setattr("fuel.cli.ensure_gemma_model", fake_download)

    exit_code = main(["download-model", "--model-path", str(tmp_path / "model.gguf")])

    assert exit_code == 0
    assert called["model_path"] == str(tmp_path / "model.gguf")
    assert "Model ready" in capsys.readouterr().out


def test_benchmark_command(monkeypatch, capsys, tmp_path):
    def fake_run_benchmark(limit=20, output_path=None, preset="default"):
        return {
            "results": [{"index": 1, "ratio": 0.5}],
            "avg_compression": 0.5,
            "output_path": output_path or str(tmp_path / "results.json"),
        }

    monkeypatch.setattr("fuel.cli.run_benchmark", fake_run_benchmark)

    exit_code = main(["benchmark", "--limit", "2", "--output", str(tmp_path / "out.json")])

    assert exit_code == 0
    assert "Avg compression" in capsys.readouterr().out
