import numpy as np

from fuel.compress import _embed_gemma


def test_embed_gemma_accepts_list_data(monkeypatch):
    class FakeModel:
        def create_embedding(self, text):
            return {"data": [{"embedding": [1.0, 0.0]}]}

    monkeypatch.setattr("fuel.compress._load_gemma", lambda: FakeModel())

    arr = _embed_gemma(["hello"])

    assert arr.shape == (1, 2)
    assert np.isfinite(arr).all()
