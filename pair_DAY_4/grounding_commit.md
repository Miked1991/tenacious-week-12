# Grounding Commit

**Author:** Mikias Dagem
**Date:** 2026-05-08

---

## Change 1 — ValueError guard in score_dimension()

**Pointer:** `scoring/scoring_evaluator.py` — `score_dimension()` dispatcher.

### What changed — ValueError guard

The original dispatcher used a string match against known dimension names and fell through to `return 1.0` for any key that did not match. The fall-through was the default branch — there was no explicit handling for the unrecognized case, no log statement, and no signal to the caller that the key had been silently ignored.

The fall-through was replaced with an explicit guard:

```python
raise ValueError(
    f"Unrecognized rubric dimension: '{dimension}'. "
    f"Registered dimensions: {sorted(SCORER_REGISTRY.keys())}"
)
```

### Why it mattered — ValueError guard

A silent `return 1.0` on an unrecognized key awards the full weight of that dimension to every output without evaluating any of them. The corruption is one-directional — it can only inflate scores, never deflate them — and it leaves no trace in the output. Replacing it with a `ValueError` converts silent corruption into a loud failure that surfaces immediately and points directly to the offending key.

---

## Change 2 — Load-time key validation in benchmark runner

**Pointer:** `scoring/run_benchmark.py` — task JSONL load step.

### What changed — load-time validation

A validation pass was added immediately after the task JSONL is loaded and before any scoring begins. The pass iterates every unique dimension key referenced across all tasks and checks each against `SCORER_REGISTRY`. If any key is unregistered, the run aborts with a descriptive error listing the unrecognized keys and the registered alternatives.

### Why it mattered — load-time validation

A `ValueError` raised inside `score_dimension()` at example 500 of a 1,000-example benchmark is better than silent corruption, but it still wastes the work done on examples 1–499 and leaves a partial results file that could be mistaken for a complete run. The load-time check moves the failure to startup, before any compute is spent, and makes the error impossible to miss. The two guards are complementary: load-time validation prevents the problem at scale; the in-function `ValueError` is a secondary defense for keys that reach the scorer through any path not gated by the load step.

---

## Change 3 — Caveat in benchmark results

**Pointer:** `benchmark_results/tenacious_bench_v0.1.json` — metadata section.

### What changed — results caveat

The results file reported a mean rubric score of 81.4 with no qualification on how the scorer handled unexpected input. A caveat was added to the metadata section clarifying that the score was produced by the pre-fix dispatcher, that any task JSONL keys mismatched against the scorer registry during that run would have received silent full-mark awards, and that the figure cannot be confirmed as corruption-free until the benchmark is re-run with the validated dispatcher and load-time key check in place.

### Why it mattered — results caveat

Publishing an unqualified score produced by an unaudited scorer misrepresents the reliability of the measurement to anyone using the figure for model selection or comparison. The caveat makes the limitation explicit and flags that the number should not be used for downstream decisions until the clean re-run is complete.
