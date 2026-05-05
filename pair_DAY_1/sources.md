# Sources

---

## Two Canonical Sources

- **BoostLoRA (arXiv:2604.27308)** — [https://arxiv.org/html/2604.27308v1](https://arxiv.org/html/2604.27308v1)
  Theoretical foundation establishing that merged LoRA carries zero per-token runtime overhead. Used to back the core mechanistic claim that `W' = W + AB` eliminates adapter arithmetic entirely after a one-time merge.

- **HuggingFace PEFT `merge_and_unload()` documentation** — [https://huggingface.co/docs/peft/v0.13.2/en/developer_guides/merge_peft_adapters](https://huggingface.co/docs/peft/v0.13.2/en/developer_guides/merge_peft_adapters)
  Official API reference confirming the merge-and-discard behaviour: adapter matrices A and B are folded into the base weight and then freed from memory, leaving only W' at inference time.

---

## Tool / Pattern Used

**Controlled ablation** — toggling one variable at a time (unmerged → merged, then incrementally adding Flash Attention, quantization, kernel fusion, and KV-cache improvements) while holding all other settings fixed. Prefill and decode latency are measured separately at each step using `time.perf_counter` for end-to-end timing and `torch.profiler` for per-kernel CUDA breakdown.
