from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .model_utils import ensure_gemma_model


def run_benchmark(limit: int = 20, output_path: str | None = None, preset: str = "default") -> dict:
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
        deduper = CoTDeduper(preset=preset)
        deduped = deduper.dedup(steps)
        ratio = deduper.compression_ratio(steps, deduped)
        results.append({"index": idx, "ratio": ratio})

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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
