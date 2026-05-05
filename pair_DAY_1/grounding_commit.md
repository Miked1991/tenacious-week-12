# Grounding Commit

---

## Change 1 — Cost-Pareto table and verdict paragraph

**Pointer:** [`model_card.md` lines 142–148](model_card.md#L142-L148)

### What changed

The original Cost-Pareto table had two columns — baseline and unmerged adapter — which made the +25.5% latency overhead appear to be a permanent cost of using the adapter.

A third column, **"With adapter (merged)"**, was added showing that latency returns to the baseline 1.84s per task once `model.merge_and_unload()` is called, while the accuracy score (78.8%) stays identical. The Pareto verdict paragraph below the table was updated to explicitly name `merge_and_unload()` and clarify that the overhead is an **inference configuration choice**, not an inherent property of LoRA.

### Why it mattered

A reader evaluating whether to deploy the adapter would see the original two-column table and conclude that every inference call is 25% slower — a conclusion that would lead teams to either accept unnecessary latency or skip the adapter entirely.

The corrected table shows the true production trade-off: **zero latency cost, +29.6pp score lift**.

---

## Change 2 — Trained-adapter generation pipeline

**Pointer:** [`ablations/generate_trained_outputs.py` lines 190–221](ablations/generate_trained_outputs.py#L190-L221) — `load_adapter()` function, specifically line 218.

### What changed

Before this file existed, `merge_and_unload()` appeared only in documentation examples — there was no execution path in the codebase where it was actually called. The entire trained-adapter generation pipeline was missing: `training/train.py` contains a `generate_email()` helper but no entry-point that loads the adapter, runs inference on the held-out partition, and saves outputs.

`generate_trained_outputs.py` fills that gap. Its `load_adapter()` function places `model = model.merge_and_unload()` as a **mandatory step** between loading the adapter and returning the model to the inference loop.

### Why it mattered

This positioning ensures the call cannot be skipped: any developer running the generation pipeline gets a merged model automatically, without needing to know that the default unmerged state carries a +25.5% latency penalty.

The gap was not just that the fix was undocumented — it was that **the execution path where the fix belongs did not exist**.
