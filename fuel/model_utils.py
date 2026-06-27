from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def ensure_gemma_model(model_path: Optional[str] = None, force: bool = False) -> Path:
    """Download the Gemma GGUF model if it is missing.

    The default path is models/gemma-3-270m-it-q4_k_m.gguf.
    """
    if model_path is None:
        model_path = "models/gemma-3-270m-it-q4_k_m.gguf"

    path = Path(model_path)
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    url = "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf"
    command = ["wget", "-O", str(path), url]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("wget is required to download the Gemma model") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to download Gemma model: {exc.stdout}") from exc

    return path
