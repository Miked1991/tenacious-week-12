# Tweet / LinkedIn Thread

---

### Tweet 1 / Hook

I fine-tuned Qwen2.5-1.5B with LoRA and clocked a +25.5% latency overhead.

I blamed the adapter. I was wrong.

One line of code wiped out that penalty entirely — with zero change to model outputs.

Here's what I missed 🧵

---

### Tweet 2 / The mechanism

LoRA adds two small matrices A and B to each layer.

Unmerged, every single forward pass recomputes:

- W·x (base)
- A·B·x (adapter)
- then adds them

For 100 generated tokens, that adapter multiply runs **101 times per layer**.

`merge_and_unload()` collapses W' = W + AB once — then A and B are gone forever.

---

### Tweet 3 / The experiment

Same model. Same adapter. Two modes: merged vs. unmerged.

The 2.41× speedup breaks down like this:

- **Decode phase** (~60–80%) — adapter math runs once per token; merging kills it entirely
- **Prefill** (~10–20%) — removes one extra GEMM on the input prompt
- **Quantization** (~5–15%) — merged weights compress cleaner
- **Kernel fusion + KV-cache** — smaller but non-zero gains

Decode dominates because it's memory-bound and repeated for every output token.

---

### Tweet 4 / The lesson

The merged and unmerged models have the **exact same parameter count**.

The speedup isn't from fewer parameters — it's from fewer *operations*.

That distinction matters when you write a model card. I called the +25.5% overhead "inherent to LoRA." It isn't. It's a configuration choice.

---

### Tweet 5 / When NOT to merge

Merging is irreversible. Keep the adapter unmerged if you need:

- **Dynamic adapter swapping** — serving multiple fine-tunes on one base model
- **Multi-tenant inference** — different users, different adapters, same GPU
- **Continued training** — you still need A and B as separate parameters

Otherwise: merge. The +25.5% penalty is configurable, not fundamental.

---

### Tweet 6 / Takeaway

Before you blame a technique, check your configuration.

`merge_and_unload()` — one call, 2.41× faster, identical outputs.

The model card now distinguishes inherent costs from configurable ones. That's the real fix.
