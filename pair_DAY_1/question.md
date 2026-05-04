# Sharpen Question

**Author:** Mikias Dagem
**Date:** 05/04/2026

In my Week 11 model card for a fine-tuned Qwen2.5-1.5B classifier, I documented a +25.5% latency overhead when using the LoRA adapter in its default unmerged state. I claimed this was an inherent cost of adaptation. However, I now realize that merging the LoRA weights into the base model (`merge_and_unload()`) eliminates the two extra matrix multiplications per layer (`B @ A @ x`) with zero change to inference outputs.

**Question:** Why does the unmerged path force recomputation of `B @ A @ x` at every forward pass, and what specific inference-time conditions (e.g., dynamic adapter swapping, multi-tenant serving) would justify keeping the adapter unmerged despite the +25.5% penalty?

Knowing this would let me rewrite my model card’s “Performance” section to distinguish inherent vs. configurable costs and add a decision tree for when to merge. 
