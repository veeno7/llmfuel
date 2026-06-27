from fuel import CoTDeduper


def test_dedup_removes_paraphrased_repeats():
    steps = [
        "Solve 2+2.",
        "To solve it, I add two and two.",
        "The answer is 4.",
    ]

    deduper = CoTDeduper(preset="pi")
    out = deduper.dedup(steps)

    assert len(out) == 2
    assert out[0] == "Solve 2+2."
    assert out[1] == "The answer is 4."


def test_dedup_removes_paraphrased_facts():
    steps = [
        "The capital of France is Paris.",
        "Paris is the capital city of France.",
        "The Eiffel Tower is in Paris.",
    ]

    deduper = CoTDeduper(preset="pi")
    out = deduper.dedup(steps)

    assert len(out) == 2
    assert out[0] == "The capital of France is Paris."
    assert out[1] == "The Eiffel Tower is in Paris."


def test_dedup_removes_math_paraphrases():
    steps = [
        "The derivative of x squared is 2x.",
        "Differentiating x^2 gives 2x.",
        "So the final answer is 2x.",
    ]

    deduper = CoTDeduper(preset="pi")
    out = deduper.dedup(steps)

    assert len(out) == 2
    assert out[0] == "The derivative of x squared is 2x."
    assert out[1] == "So the final answer is 2x."
