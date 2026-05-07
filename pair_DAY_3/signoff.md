# Sign-off

**Author:** Mikias Dagem
**Date:** 2026-05-07
**Gap status:** Closed

---

## Gap-closure judgment

The gap is fully closed. `generate_full_training_data.py` now splits at the probe level — all Magpie variants of a given probe stay on one side of the boundary, so train and eval contain structurally independent examples. `train.py:207` now selects checkpoints against `eval_rubric_score` from the true held-out set rather than against the contaminated eval loss. `model_card.md` now carries an explicit caveat that the reported 0.0213 is a variant-level figure and cannot be read as generalization loss.

The checkpoint at step 441 that was published to HuggingFace was selected on a corrupted signal. The fix does not retroactively validate that checkpoint — it means future runs will select on a signal that actually measures what it claims to measure.

---

## What the asker now understands

**Before:** The eval loss of 0.0213 felt suspiciously low but I hadn't diagnosed why. My working assumption was that the task might just be easy — that the model had genuinely learned to segment ICP categories well enough that near-zero loss was achievable. The split was random at the example level, which seemed fine.

**After:** The split being random at the example level is the problem, not the solution. When 10 Magpie variants of the same probe exist and you shuffle them randomly into train and eval, you get 8 in train and 2 in eval on average — and those 2 eval examples are paraphrases of training examples. The model hasn't seen those exact phrasings, but it has absorbed the underlying company profile, the correct ICP segment, and the required funding reference. Scoring well on the eval set requires nothing more than memorizing the probe's core facts. A variant-level split turns eval loss into a memorization metric and checkpoint selection into a competition for deepest memorization.

The clean held-out set existed the entire time. The gap wasn't a missing resource — it was a wiring failure: a valid signal that was never connected to the decision it should have been driving.

**The discipline:** having a held-out set is not enough. The metric that drives checkpoint selection must be computed against that set. A held-out set that sits unused while a contaminated signal picks the checkpoint is functionally equivalent to having no held-out set at all.
