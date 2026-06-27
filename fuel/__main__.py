from __future__ import annotations

import argparse

from .model_utils import ensure_gemma_model


def main() -> None:
    parser = argparse.ArgumentParser(description="llmfuel helpers")
    parser.add_argument("--download-model", action="store_true", help="Download the Gemma GGUF model")
    parser.add_argument("--model-path", default=None, help="Optional model download path")
    args = parser.parse_args()

    if args.download_model:
        path = ensure_gemma_model(model_path=args.model_path)
        print(f"Model ready at {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
