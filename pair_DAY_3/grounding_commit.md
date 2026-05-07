# Grounding Commit

**Author:** Mikias Dagem
**Date:** 2026-05-07

---

## Change 1 — Probe-level split in training data generation

**Pointer:** `training_data/generate_full_training_data.py` — split logic replacing the variant-level shuffle.

### What changed — probe-level split

The original split shuffled individual examples randomly, distributing all 10 Magpie variants of each probe across both train and eval with no constraint. This meant that for any given probe, 8 variants typically landed in train and 2 in eval — different surface phrasings of the same underlying company profile, same Crunchbase signals, same correct ICP segment, same required funding reference.

The split was rewritten to operate at the probe level: all variants of a given probe stay on one side of the boundary. Train and eval now contain entirely different probes, not different phrasings of the same probes.

### Why it mattered — probe-level split

The old split made eval loss measure memorization, not generalization. A model that had seen probe_42 in 8 paraphrases during training would score well on probe_42's remaining 2 eval variants because it had already absorbed the underlying company profile — not because it had learned to reason about ICP signals in general. Checkpoint selection based on that loss metric was selecting the deepest memorizer, not the best-generalizing model.

---

## Change 2 — Checkpoint callback switched from eval loss to rubric score

**Pointer:** `training/train.py:207` — `TrainingArguments` metric configuration.

### What changed — checkpoint callback

The checkpoint callback was using `eval_loss` as `metric_for_best_model`, with `load_best_model_at_end=True`. This loaded and published whichever checkpoint had the lowest eval loss — a metric that was corrupted by the variant-level split.

The callback was updated to:

```python
metric_for_best_model = "eval_rubric_score"
greater_is_better = True
```

The rubric callback logs `eval_rubric_score` via `trainer.log()` inside an `on_evaluate` method, using the scoring logic from `scoring_evaluator.py` against the true held-out set.

### Why it mattered — checkpoint callback

The held-out set existed the entire time but was never wired into checkpoint selection. The published checkpoint at step 441 was selected on a corrupted signal. Switching to rubric score against the clean held-out set means checkpoint selection now reflects actual generalization performance.

---

## Change 3 — Eval loss caveat in model card

**Pointer:** `model_card.md` — eval metrics section.

### What changed — model card caveat

The model card reported the best eval loss of 0.0213 at checkpoint-441 without any qualification. A reader would interpret this as the generalization loss on a held-out set.

A caveat was added clarifying that the reported 0.0213 is a variant-level eval loss — computed on paraphrases of training probes — and cannot be interpreted as the generalization loss. The rubric score against the true held-out set is the figure to use for comparing checkpoint quality.

### Why it mattered — model card caveat

Publishing an unqualified eval loss number that is structurally inflated by data leakage misrepresents model quality to anyone reading the card. The caveat makes the limitation explicit so downstream readers can weight the number appropriately.
