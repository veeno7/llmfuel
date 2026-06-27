from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .model_utils import ensure_gemma_model, generate_with_model


def run_benchmark(limit: int = 20, output_path: str | None = None, preset: str = "default", aggressiveness: str = "balanced") -> dict:
    from .compress import CoTDeduper

    samples = [
        {"question": "What is 2+2?", "answer": "#### 4"},
        {"question": "Solve 3x+1=10", "answer": "#### 3"},
    ]

    results = []
    for idx, ex in enumerate(samples[:limit], start=1):
        steps = [
            ex["question"],
            ex["question"],
            f"Answer: {ex['answer']}",
        ]
        deduper = CoTDeduper(preset=preset, aggressiveness=aggressiveness)
        deduped = deduper.dedup(steps)
        ratio = deduper.compression_ratio(steps, deduped)
        results.append({"index": idx, "ratio": ratio, "original_tokens": sum(len(s.split()) for s in steps), "compressed_tokens": sum(len(s.split()) for s in deduped)})

    output = output_path or "benchmarks/benchmark_results.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return {
        "results": results,
        "avg_compression": sum(r["ratio"] for r in results) / len(results) if results else 0.0,
        "output_path": output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmfuel", description="llmfuel local CoT deduplication helpers")
    subparsers = parser.add_subparsers(dest="command")

    download = subparsers.add_parser("download-model", help="Download the Gemma GGUF model")
    download.add_argument("--model-path", default=None, help="Optional path for the downloaded model")
    download.add_argument("--force", action="store_true", help="Re-download even if the file exists")

    benchmark = subparsers.add_parser("benchmark", help="Run a simple local benchmark")
    benchmark.add_argument("--limit", type=int, default=20, help="Number of sample cases")
    benchmark.add_argument("--output", default=None, help="Path to write benchmark JSON")
    benchmark.add_argument("--preset", default="default", choices=["default", "pi"], help="Deduper preset")
    benchmark.add_argument("--aggressiveness", default="balanced", choices=["balanced", "aggressive", "max"], help="How aggressively to collapse repeated reasoning")

    generate = subparsers.add_parser("generate", help="Generate text from a local GGUF model")
    generate.add_argument("--prompt", required=True, help="Prompt to send to the model")
    generate.add_argument("--model-path", default=None, help="Optional path to the GGUF model")
    generate.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    generate.add_argument("--n-gpu-layers", type=int, default=0, help="Number of GPU layers to offload")
    generate.add_argument("--ctx-size", type=int, default=2048, help="Context window size")
    generate.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    generate.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    generate.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty")
    generate.add_argument("--n-threads", type=int, default=None, help="Number of CPU threads")
    generate.add_argument("--n-batch", type=int, default=None, help="Batch size for prompt processing")
    generate.add_argument("--verbose", action="store_true", help="Enable verbose llama.cpp logging")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "download-model":
        path = ensure_gemma_model(model_path=args.model_path, force=args.force)
        print(f"Model ready at {path}")
        return 0

    if args.command == "benchmark":
        result = run_benchmark(limit=args.limit, output_path=args.output, preset=args.preset)
        print(f"Saved {len(result['results'])} benchmark entries to {result['output_path']}")
        print(f"Avg compression: {result['avg_compression']:.2f}")
        return 0

    if args.command == "generate":
        result = generate_with_model(
            prompt=args.prompt,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.ctx_size,
            verbose=args.verbose,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            n_threads=args.n_threads,
            n_batch=args.n_batch,
        )
        print(result["text"])
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
