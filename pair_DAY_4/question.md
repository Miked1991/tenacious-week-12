# Pair Day 4 — Research Question

**Author:** Mikias Dagem
**Date:** 2026-05-08

---

## Context

In `scoring/scoring_evaluator.py` (Tenacious-Bench v0.1), the `score_dimension()` dispatcher selects a scoring function by matching a string key against a fixed set of rubric dimension names. When the key does not match any known dimension, the function falls through and returns `1.0` — silently awarding full marks with no error raised and no warning logged.

The task JSONL files that drive the benchmark pass dimension names as plain strings. A single-character typo — `no_bench_words` instead of `no_bench_word`, `funding_tier` instead of `funding_tiers` — causes the dispatcher to miss every intended scoring function for that dimension. The model's output is never evaluated against the rubric. The dimension receives a perfect score. The composite benchmark score is inflated by the full weight of the missed dimension, with nothing in the logs to indicate the key was unrecognized.

`benchmark_results/tenacious_bench_v0.1.json` reports a mean rubric score of 81.4. Whether that figure reflects actual model quality or partially reflects silent full-mark awards from unmatched keys is currently unknown, because the dispatcher has no mechanism to distinguish "scored correctly" from "key not found."

## Question

In a string-keyed rubric dispatcher, why does a silent permissive default — returning a fixed high score rather than raising an error — systematically inflate benchmark scores rather than introduce random noise, and what is the correct defensive pattern for a deterministic evaluator to guarantee that every task is scored against only its intended dimensions?

Knowing this will let me:

1. Replace the fall-through default in `score_dimension()` with an explicit guard that raises `ValueError` on any unrecognized key, converting silent corruption into a loud failure
2. Add a validation pass at benchmark load time that cross-checks every dimension key in the task JSONL against the registered scorer set before a single example is evaluated
3. Add a note to `benchmark_results/tenacious_bench_v0.1.json` clarifying that the reported 81.4 score cannot be confirmed as corruption-free until the dispatcher is audited and the benchmark is re-run
