# Evening Call Summary – Day 1

**Pair:** Mikias Dagem and Melaku
**Topic:** Inference-time mechanics

---

**Asker's question:**

> How can a LoRA-adapted Qwen2.5-1.5B model achieve a 2.41× inference speedup despite identical merged parameter counts — and how can I experimentally decompose that gain across prefill, decode, quantization, kernel fusion, and KV-cache behavior?

---

## Feedback raised

The original explainer covered the LoRA slowdown conceptually but lacked empirical evidence. The asker requested a concrete benchmark to ground the claims in measured data.

## Changes made

- Added a benchmark comparing three configurations — base model, unmerged LoRA adapter, and merged LoRA adapter — under identical generation settings.
- Clarified that the latency penalty comes from **runtime adapter overhead** (the extra matrix multiplications per forward pass), not from a higher parameter count.
- Expanded discussion of decode-phase sensitivity, kernel fusion, and memory-bandwidth effects.
- Revised explainer now includes measured latency and tokens-per-second comparisons alongside a clearer explanation of why `merge_and_unload()` restores near-baseline inference performance.

## Outcome

Both partners signed off. The question is marked **closed**.
