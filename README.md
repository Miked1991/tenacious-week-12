# TRP1 Week 12 — Knowledge Gap Formulation

| | |
|---|---|
| **Program** | 10 Academy TRP1 |
| **Week** | 12 |
| **Dates** | May 5–6, 2026 |
| **Participant** | Mikias Dagem |

---

## Overview

Week 12 uses a structured peer-teaching format. Each day, two participants take turns as **Asker** and **Explainer**. The Asker formulates a precise knowledge gap tied to their actual project work; the Explainer closes it with a focused, mechanism-driven explainer. Every session ends with a **grounding commit** — a concrete edit to a real portfolio artifact that pays the insight back into the work.

---

## Day 1 — LoRA Inference Mechanics

| | |
|---|---|
| **Date** | May 5, 2026 |
| **Partner** | Melaku |
| **Status** | Complete ✅ |

### Mikias as Asker

**Question:** In my Week 11 model card for a fine-tuned Qwen2.5-1.5B classifier, I documented a +25.5% latency overhead when using the LoRA adapter in its default unmerged state and claimed this was an inherent cost of adaptation. Why does the unmerged path force recomputation of `B @ A @ x` at every forward pass, and what specific inference-time conditions would justify keeping the adapter unmerged despite the penalty?

**Gap closed:** The overhead is not inherent — it is an inference configuration choice. The unmerged path computes `Wx` and `ABx` separately on every forward pass and adds them; `merge_and_unload()` folds `W' = W + AB` once before inference and discards `A` and `B`, cutting the per-layer arithmetic in half with zero change to output values. The +25.5% penalty only survives in cases that require live adapter swapping (multi-tenant serving, dynamic adapter routing) — neither of which applies to the Tenacious pipeline. The true production trade-off is zero latency cost against a +29.6pp accuracy lift.

**Grounding commit:** Added a third column — *"With adapter (merged)"* — to the Cost-Pareto table in `model_card.md`, showing that latency returns to the 1.84s baseline once `merge_and_unload()` is called. Created `ablations/generate_trained_outputs.py` with `merge_and_unload()` placed as a mandatory step in `load_adapter()`, making the merged path the only available execution path for trained-adapter evaluation.

---

### Mikias as Explainer

**Question received from Melaku:** How can a LoRA-adapted Qwen2.5-1.5B model achieve a 2.41× inference speedup despite identical merged parameter counts — and how can I experimentally decompose that gain across prefill, decode, quantization, kernel fusion, and KV-cache behavior?

**Key points delivered:**

- The speedup comes from eliminating per-token adapter arithmetic, not from reducing parameter count. For 100 generated tokens, the unmerged model runs the adapter multiplication 101 times per adapted layer; the merged model runs it zero times.
- Decode phase dominates (~60–80% of the gain) because it is memory-bound and repeats per output token. Prefill contributes ~10–20%.
- Quantization adds a further 5–15% speedup because merged weights compress cleanly; unmerged adapters can cause numerical spillover.
- Kernel fusion reduces per-layer CUDA launches from three (Wx, ABx, addition) to one.
- KV-cache reuse is restored fully after merging; some unmerged implementations break the static computation graph and force cache recomputation.
- Provided a complete five-step benchmark protocol (TTFT, TPOT, profiler, quantization sweep, cache-reuse loop) for Melaku's model card.

**Artifacts:** [pair_DAY_1/](pair_DAY_1/)

---

## Day 2 — Agent Reliability and Tool-Use Internals

| | |
|---|---|
| **Date** | May 6, 2026 |
| **Partner** | Nahom Desalegn |
| **Status** | Complete ✅ |

### Mikias as Asker

**Question:** The Tenacious conversion engine calls Cal.com, Resend, Crunchbase, PDL, and Africa's Talking with no retry logic — a single timeout means permanent, silent failure. What distinguishes a transient from a permanent API failure, why does exponential backoff with jitter prevent retry storms, what is a retry budget and why does it need a ceiling, and would the same retry strategy apply to `booking_handler.py`, `email_outreach.py`, and `enrichment_pipeline.py` — or does each file need a different approach?

**Gap closed:** Retry only on transient signals (429, 5xx, `TimeoutError`, `ConnectError`, `RemoteProtocolError`) — any other 4xx is a permanent failure and retrying it wastes budget. Exponential backoff with jitter desynchronises concurrent retriers after a shared outage, preventing the thundering-herd re-collapse that fixed-interval retries cause. The `min(base × 2^attempt, cap)` ceiling keeps worst-case blocked time predictable; without it, a single slow external API can exhaust the thread pool. The three files are not equivalent: `email_outreach.py` and `enrichment_pipeline.py` are safe to retry (idempotent by nature or with a key); `booking_handler.py` requires an idempotency key before retrying is safe — the missing key is the actual bug, not the missing retry.

**Grounding commit:** Created `agent/retry.py` (67 lines, stdlib only — `functools`, `time`, `random`, `httpx`) exposing an `@http_retry` decorator. Applied across six agent files: `booking_handler.py`, `email_outreach.py`, `enrichment_pipeline.py`, `hubspot_sync.py`, `sms_handler.py`, and `signals_research.py`. Every bare single-shot `httpx.post`/`httpx.get` is now wrapped — a Cal.com timeout no longer silently drops a qualified prospect.

---

### Mikias as Explainer

**Question received from Nahom:** At the token level, when a model using function-calling selects a tool, is the description string part of the selection computation — or is it processed only after the tool-name token is already committed?

**Key points delivered:**

- Tool schemas (name, description, JSON Schema) are serialised into the context window and processed during the prefill forward pass. Every description token attends to every other token through self-attention; the resulting key-value pairs are stored in the KV cache before a single output token is generated.
- This makes description strings **causally upstream** of the tool-name logit distribution — they shape which tool gets selected, not just how arguments are filled in afterward.
- The *BiasBusters* paper (ICLR 2026) found description quality is the strongest single driver of tool selection; a separate study found well-edited descriptions produce 10× more usage than original ones, with names and parameters held constant.
- Constrained decoding (token masking at the name position) changes *when* description quality matters most but does not eliminate the causal effect — the probability mass the mask operates on is still shaped by descriptions during prefill.
- Practical rule: treat tool descriptions as contracts, not comments. Vague descriptions force the model onto name-only signal, which is weaker and leads to silent misroutes.

**Artifacts:** [pair_DAY_2/](pair_DAY_2/)

---

## Repository Structure

```
week-12/
├── README.md
├── pair_DAY_1/
│   ├── question.md               ← Mikias's sharpened question to Melaku
│   ├── explainer.md              ← Mikias's LoRA inference explainer for Melaku
│   ├── morning_call_summary.md
│   ├── evening_call_summary.md
│   ├── signoff.md
│   ├── grounding_commit.md
│   ├── sources.md
│   └── thread.md
└── pair_DAY_2/
    ├── question.md               ← Mikias's retry logic question to Nahom
    ├── explainer.md              ← Mikias's tool-selection explainer for Nahom
    ├── morning_call_summary.md
    ├── evening_call_summary.md
    ├── signoff.md
    ├── grounding_commit.md
    ├── sources.md
    └── thread.md
```
