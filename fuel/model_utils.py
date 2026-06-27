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


def generate_with_model(
    prompt: str,
    model_path: Optional[str] = None,
    max_tokens: int = 128,
    n_gpu_layers: int = 0,
    n_ctx: int = 2048,
    verbose: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    n_threads: Optional[int] = None,
    n_batch: Optional[int] = None,
) -> dict:
    """Generate text using a local GGUF model via llama-cpp-python."""
    if model_path is None:
        model_path = "models/gemma-3-270m-it-q4_k_m.gguf"

    model_file = Path(model_path)
    if not model_file.exists():
        model_file = ensure_gemma_model(model_path=str(model_file))

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is required for model generation") from exc

    llm_kwargs = {
        "model_path": str(model_file),
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": verbose,
        "embedding": False,
    }
    if n_threads is not None:
        llm_kwargs["n_threads"] = n_threads
    if n_batch is not None:
        llm_kwargs["n_batch"] = n_batch

    llm = Llama(**llm_kwargs)
    messages = [{"role": "user", "content": prompt}]
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"text": text, "model_path": str(model_file)}
