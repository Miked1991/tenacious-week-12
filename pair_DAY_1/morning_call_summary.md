# Morning Call Summary – Day 1

**Pair:** Mikias Dagem and Melaku
**Topic:** Sharpening the inference-time question

---

## What we discussed

The original draft question focused too narrowly on the Unsloth `FastLanguageModel.for_inference()` API, framing the observed speedup as a library-specific optimization. During the call, we identified that the deeper gap was not "what Unsloth does," but why inference latency can change dramatically even when LoRA weights are merged and parameter count stays constant.

We debated whether the root cause was quantization, LoRA merging, or transformer inference phases, and agreed the question needed to target a systems-level decomposition — covering prefill and decode phases, kernel fusion, KV-cache behavior, and memory-bandwidth bottlenecks.

## Changes made

- Reframed the question away from a specific library API toward a general inference-mechanics explanation.
- Named the core paradox explicitly: identical merged model size, large latency difference.
- Tightened the connection to the Week 11 final report's Pareto-dominance claim so the grounding commit path was clear.

## Key clarification

Melaku confirmed that his slowdown occurs **pre-merge** and is most likely caused by runtime adapter injection overhead, not by increased model size.

## Outcome

Both questions were refined to isolate inference-time mechanisms rather than general performance observations. The sharpened question was ready to hand off to the explainer.
