# TRP1 Week 12 — Knowledge Gap Formulation

| | |
|---|---|
| **Program** | 10 Academy TRP1 |
| **Week** | 12 |
| **Dates** | May 5–8, 2026 |
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

## Day 3 — SFT Eval Corruption and ORPO Gradient Mechanics

| | |
|---|---|
| **Date** | May 7, 2026 |
| **Partner** | Kemeriya Major |
| **Status** | Complete ✅ |

### Mikias as Asker

**Question:** In `training/train.py`, the SFT script splits examples randomly at the variant level across `sft_train.jsonl` and `sft_eval.jsonl`. The actual data is built from 10 Magpie variants per probe, distributed roughly 8/2 across train and eval. This means every eval example shares the same underlying probe — same Crunchbase signals, same correct ICP segment, same required funding reference — as training examples. `model_card.md` reports a best eval loss of 0.0213 at checkpoint-441, selected via `load_best_model_at_end=True`. When you split train and eval at the variant level rather than the probe level, how does this corrupt the eval loss signal used for checkpoint selection — and what should replace it when a true held-out set already exists?

**Gap closed:** A variant-level split makes eval loss measure memorization, not generalization. A model that has seen probe_42 in 8 paraphrases during training scores well on probe_42's 2 remaining eval variants because it has absorbed the underlying company profile — not because it has learned to reason about ICP signals in general. Checkpoint selection on that signal selects the deepest memorizer. The fix is a probe-level split in `generate_full_training_data.py` (all variants of a probe stay on one side) combined with replacing the checkpoint callback in `train.py:207` with rubric score from `scoring_evaluator.py` against the true held-out set. The held-out set existed the entire time — the gap was a wiring failure, not a missing resource.

**Grounding commit:** Rewrote the split logic in `training_data/generate_full_training_data.py` to operate at the probe level. Updated `train.py:207` to use `metric_for_best_model = "eval_rubric_score"` and `greater_is_better = True`, with the rubric callback logging under the exact key via `trainer.log()` inside `on_evaluate`. Added a caveat to `model_card.md` clarifying that the reported 0.0213 eval loss is variant-level and cannot be interpreted as generalization loss.

---

### Mikias as Explainer

**Question received from Kemeriya:** In the ORPO training run documented in `submission_report.md` Section 7, signal-grounding preference pairs produced +11.8 pts while dual-control pairs produced −4.5 pts at the same β=0.2. Which of three hypotheses — gradient overshoot from β pressure, data sparsity in the 33 dual-control pairs, or early-stop artifact — is consistent with ORPO's gradient mechanics, and what would the loss curves look like under each?

**Key points delivered:**

- TRL's `ORPOTrainer` computes each sequence's log probability as a per-token average — it divides summed log-probs by sequence length T. This is correct design for handling variable-length sequences, but it creates a structural penalty when chosen and rejected differ at only one position.
- Signal-grounding pairs have errors distributed across ~30 tokens in a 100-token response. After dividing by T, the preference gradient contribution is ~0.30. Dual-control pairs differ at exactly one terminal token — after dividing by T, the contribution is ~0.01, which falls below the optimizer's effective noise floor at standard learning rates.
- The model receives no usable update from dual-control pairs. Small SFT-driven adjustments from other batch examples accumulate and produce the regression. More steps deepen it — the broken calculation doesn't converge, it compounds.
- Neither overshoot nor early-stop explain the pattern. Overshoot would show rapid loss descent followed by instability; early-stop would show improvement that simply plateaus. Consistent regression from step 1 points to a zero-magnitude gradient, which is the dilution diagnosis.
- β=0.2 does not help — β scales both sides of the loss proportionally and does not touch the 1/T denominator. The fix is structural: override the loss mask for terminal-error pairs so the denominator equals 1, then raise β to 0.5 and add gradient clipping at `max_norm=0.5` for the first ~200 steps.

**Artifacts:** [pair_DAY_3/](pair_DAY_3/)

---

## Day 4 — Benchmark Scorer Integrity and LLM Judge Reliability

| | |
|---|---|
| **Date** | May 8, 2026 |
| **Partner** | Ephrata Wolde |
| **Status** | Complete ✅ |

### Mikias as Asker

**Question:** In `scoring/scoring_evaluator.py` (Tenacious-Bench v0.1), the `score_dimension()` dispatcher falls through to return `1.0` for any dimension name it does not recognize — silently awarding full marks rather than raising an error or logging a warning. When a task JSONL contains a typo (e.g., `no_bench_words` instead of `no_bench_word`), the scorer returns a perfect score for that dimension with no indication the rubric key was unrecognized. Why does a silent permissive default in a string-keyed rubric dispatcher systematically inflate benchmark scores rather than introduce random noise, and what is the correct defensive pattern for a deterministic evaluator to guarantee that every task is scored against only its intended dimensions?

**Gap closed:** The inflation is systematic, not random, because the fall-through always returns 1.0 and never 0.0 — every unrecognized key pulls the composite score up with no opposing pressure. This is not noise that averages out; it is a one-directional ceiling applied to every task containing a mismatched key, and the corruption is invisible in the score distribution. The correct defensive pattern has two layers: a `ValueError` guard inside `score_dimension()` that converts any unrecognized key into a loud failure, and a load-time validation pass that cross-checks every dimension key in the task JSONL against the registered scorer set before a single example is evaluated. The load-time check is the primary defense at scale; the in-function guard is a secondary backstop for any key that reaches the scorer through a path not gated by the load step.

**Grounding commit:** Replaced the fall-through `return 1.0` in `score_dimension()` with an explicit `ValueError` listing the unrecognized key and registered alternatives. Added a load-time validation pass to `scoring/run_benchmark.py` that aborts the run at startup if any task JSONL key is unregistered. Added a caveat to `benchmark_results/tenacious_bench_v0.1.json` stating that the reported 81.4 mean rubric score was produced by the pre-fix dispatcher and cannot be confirmed as corruption-free until the benchmark is re-run.

---

### Mikias as Explainer

**Question received from Ephrata Wolde:** My tenacious-bench uses an LLM as the judge to score model outputs. I report those scores as if they are reliable measurements, but I never tested whether the judge itself is consistent. What is inter-rater reliability, how do you measure whether an LLM judge is trustworthy, and what does an unreliable judge do to benchmark results?

**Key points delivered:**

- An LLM judge samples from a probability distribution over score tokens at inference time. At temperature > 0, two calls on identical input can return different scores. The score is a sample, not a measurement — its reliability depends on how concentrated that distribution is.
- IRR has two components that must be diagnosed separately: intra-rater reliability (does the judge agree with itself across repeated calls?) and inter-rater reliability (does the judge agree with a reference standard?). A judge can pass the first and fail the second if its biases are systematic and consistent.
- Unreliable judges produce systematic inflation, not random noise. Position bias and length bias are one-directional: a judge that favors the final-position response favors it every time, with no opposing case. There is no averaging-out. Studies find up to 30% systematic evaluation deviation from position effects alone.
- The minimum viable validation protocol is three steps in order: (1) intra-rater stability on 50 examples, (2) Judge Sensitivity Score (JSS) across 10 prompt rephrasing variants — target ≥ 0.85, (3) Quadratic Weighted Kappa (QWK) against 30 human annotations — target ≥ 0.80 for any decision that depends on ranking.
- Until those three pass, checkpoint selection on the scores picks the model that best exploits judge biases; cross-run comparisons conflate prompt drift with model improvement; and published numbers are not reproducible by anyone running a different judge.

**Artifacts:** [pair_DAY_4/](pair_DAY_4/)

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
├── pair_DAY_2/
│   ├── question.md               ← Mikias's retry logic question to Nahom
│   ├── explainer.md              ← Mikias's tool-selection explainer for Nahom
│   ├── morning_call_summary.md
│   ├── evening_call_summary.md
│   ├── signoff.md
│   ├── grounding_commit.md
│   ├── sources.md
│   └── thread.md
├── pair_DAY_3/
│   ├── question.md               ← Mikias's eval corruption question to Kemeriya
│   ├── explainer.md              ← Mikias's ORPO averaging explainer for Kemeriya
│   ├── morning_call_summary.md
│   ├── evening_call_summary.md
│   ├── signoff.md
│   ├── grounding_commit.md
│   ├── sources.md
│   └── thread.md
└── pair_DAY_4/
    ├── question.md               ← Mikias's scorer integrity question to Ephrata
    ├── explainer.md              ← Mikias's LLM judge reliability explainer for Ephrata
    ├── morning_call_summary.md
    ├── evening_call_summary.md
    ├── signoff.md
    ├── grounding_commit.md
    ├── sources.md
    └── thread.md
```
