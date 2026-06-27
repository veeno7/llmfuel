import json
import re
from fuel import CoTDeduper

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def extract_answer(text: str) -> float | None:
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def load_gsm8k_sample():
    if load_dataset is None:
        raise RuntimeError("datasets package is unavailable")

    try:
        return load_dataset("gsm8k", "main", split="test[:50]")
    except Exception as exc:
        print("Warning: unable to load GSM8K from Hugging Face:", exc)
        print("Using a small fallback sample instead.")
        return [
            {
                "question": "If one hen lays 3 eggs every day, how many eggs does she lay in 7 days?",
                "answer": "#### 21",
            },
            {
                "question": "A car travels 180 miles in 3 hours. What is its average speed in miles per hour?",
                "answer": "#### 60",
            },
            {
                "question": "A rectangle is 5 meters long and 3 meters wide. What is its area?",
                "answer": "#### 15",
            },
        ]


if __name__ == "__main__":
    gsm8k = load_gsm8k_sample()
    deduper = CoTDeduper(preset="pi")  # fast MiniLM

    results = []
    for idx, ex in enumerate(gsm8k, start=1):
        fake_cot = ex["question"] + "\nStep 1...\nStep 1 again...\nAnswer"
        steps = [s for s in fake_cot.split("\n") if s]
        deduped = deduper.dedup(steps)
        target = extract_answer(ex["answer"])

        results.append({
            "index": idx,
            "question": ex["question"],
            "target": target,
            "original_steps": len(steps),
            "deduped_steps": len(deduped),
            "ratio": len(deduped) / len(steps) if steps else None,
        })

    out_path = "benchmarks/gsm8k_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    avg_compression = sum(r["ratio"] for r in results if r["ratio"] is not None) / len(results)
    print(f"Saved {len(results)} results to {out_path}")
    print(f"Avg compression: {avg_compression:.2f}")
