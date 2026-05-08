# Sign-off

**Author:** Mikias Dagem
**Date:** 2026-05-08
**Gap status:** Closed

---

## Gap-closure judgment

The gap is fully closed. `score_dimension()` in `scoring/scoring_evaluator.py` now raises `ValueError` on any unrecognized dimension key instead of silently returning 1.0. A validation pass added to the benchmark load step cross-checks every dimension key in the task JSONL against the registered scorer set before a single example is evaluated — a key mismatch now aborts the run at startup rather than corrupting it silently at scale. `benchmark_results/tenacious_bench_v0.1.json` now carries an explicit caveat that the reported 81.4 mean rubric score was produced by the pre-fix dispatcher and cannot be confirmed as corruption-free until the benchmark is re-run with the validated scorer.

The published 81.4 figure may or may not reflect actual model quality. The fix does not retroactively validate that number — it means future runs will produce scores that are guaranteed to reflect only the intended rubric dimensions, not the accidental awarding of full marks to unrecognized keys.

---

## What the asker now understands

**Before:** The fall-through 1.0 return felt like a minor implementation gap — probably harmless because any errors would average out across enough examples. My working assumption was that an occasional wrong score is noise, and noise cancels over a large benchmark.

**After:** The failure mode has no opposing pressure. The fall-through always returns 1.0, never 0.0. Every unrecognized key pulls the composite score up — there is no case where it pulls it down. That is not noise; it is a one-directional ceiling applied to every task that contains a mismatched key. The inflation is systematic and invisible: a corrupted 81.4 is indistinguishable from a legitimately earned 81.4 by looking at the score distribution alone. The only way to detect corruption after the fact is to audit every key in the task JSONL against the registered scorer set — which is exactly what the load-time validation pass now does before any scoring begins.

The deeper discipline: a deterministic evaluator must be loud on unexpected input. A scorer that silently awards full marks on unrecognized keys is not a conservative fallback — it is an optimistic one, and optimism in a measurement instrument is a systematic error.
