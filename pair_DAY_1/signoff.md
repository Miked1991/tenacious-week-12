# Sign-off

**Gap status:** Closed

---

## Gap-closure judgment

The gap is fully closed. `merge_and_unload()` now exists in the actual execution path — [`ablations/generate_trained_outputs.py:150`](ablations/generate_trained_outputs.py#L150) calls it once after loading the adapter, before the inference loop begins. Any run of the evaluation pipeline now automatically uses merged inference.

The previous state was documentation-only: the call appeared in code comments and guide examples, but no generation entry-point existed for the trained adapter, so it was never actually executed.

---

## What the asker now understands

**Before:** The asker knew that `merge_and_unload()` existed and understood what it computed, but treated it as an optional optimization a developer could choose to apply.

**After:** An optimization that exists only in documentation is not closed — it is deferred. The gap is only closed when the correct behavior is the **default behavior** that any developer gets by running the standard pipeline.

The new script makes `lora_merged=True` the only available path. There is no way to run the trained-adapter evaluation and accidentally get unmerged inference, because the generation script does not expose an unmerged option.

**The discipline:** a gap is closed when the code enforces the fix, not when the docs describe it.
