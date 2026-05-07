# Pair Day 3 — Twitter Thread

**Author:** Mikias Dagem
**Date:** 2026-05-07
**Topic:** Why ORPO's per-token averaging silently kills gradient signal for certain preference pairs

---

**1/**
I trained with ORPO on two types of preference pairs.

One gave me +11.8 pts. The other gave me −4.5 pts.

Same β. Same optimizer. Same sequence length. Same training run.

The difference came down to one number in the loss function 🧵

---

**2/**
ORPO computes the log probability of each response by summing per-token log-probs and dividing by sequence length T.

This is deliberate — without normalization, a 200-token response dominates a 50-token one in every batch purely by length.

But this design has a consequence that isn't in the paper.

---

**3/**
Signal-grounding pairs have errors distributed across ~30 tokens in a 100-token response.

Dual-control pairs differ in exactly one terminal token — the action category — while everything else is identical.

Both get divided by T = 100.

Watch what happens to the gradient.

---

**4/**
For signal-grounding: 30 differing tokens / 100 = 0.30 preference signal reaching the optimizer.

For dual-control: 1 differing token / 100 = 0.01.

At standard learning rates, 0.01 falls below the effective noise floor. The optimizer receives the update, rounds it to near zero, and moves on.

The model never corrects the one token that's actually wrong.

---

**5/**
This is why β=0.2 doesn't fix it.

β scales the preference term relative to the SFT cross-entropy term — it controls balance, not magnitude.

Multiplying a near-zero gradient by 0.2 yields a smaller near-zero gradient. The denominator is still T. β doesn't touch the denominator.

---

**6/**
The fix is structural, not a hyperparameter tweak.

Override the loss mask for terminal-error pairs so `mask.sum(dim=1)` equals 1, not T. This makes the denominator match the number of tokens actually carrying the preference signal.

Then raise β to 0.5 and add gradient clipping at `max_norm=0.5` for the first ~200 steps to absorb the now-unmasked spike.

Without the mask change, more steps deepen the regression. The broken calculation doesn't converge — it compounds.

---

## Medium Post

### ORPO's Per-Token Averaging Has a Structural Blind Spot — Here's Where It Breaks

By Mikias Dagem | 2026-05-07

---

ORPO is elegant. It removes the need for a reference model by computing a log-odds ratio between chosen and rejected responses directly, then combining it with standard SFT cross-entropy. One loss, one model, no KL divergence term to tune separately.

But there is a structural property of how ORPO computes that log-odds ratio that creates a silent failure mode for a specific class of preference pairs. It took a +11.8 pt gain and a −4.5 pt regression in the same training run for me to find it.

---

### How TRL Computes the Preference Signal

TRL's `ORPOTrainer` computes each sequence's log probability by summing per-token cross-entropy across non-padding positions and dividing by the number of non-padding tokens:

```text
log_p(response) = Σ log p(xₜ | x<t) / T
```

This produces a per-token average rather than a raw sum. The design is correct: without this normalization, longer responses would mechanically dominate shorter ones in every batch. A 200-token response would contribute twice the gradient magnitude of a 100-token response regardless of how wrong or right either one is.

The full ORPO loss is:

```text
L = −log σ(log_p(chosen) − log_p(rejected)) + β · CE(chosen)
```

The preference gradient that flows back is proportional to `log_p(chosen) − log_p(rejected)`. Both terms carry the 1/T denominator.

---

### Where It Breaks

The 1/T normalization is correct on average, but it assumes the preference signal is distributed across the response. When it isn't — when chosen and rejected differ at exactly one position — the signal gets divided by the full sequence length whether or not that length reflects the number of positions that are actually informative.

Consider two pair types at T = 100:

**Signal-grounding pairs:** Chosen and rejected responses differ across roughly 30 tokens. The wrong version gets ICP segmentation wrong, references the wrong funding tier, uses the wrong scoring threshold. The errors are spread throughout the response. After dividing by 100, the preference gradient contribution is approximately 0.30.

**Dual-control pairs:** Chosen and rejected responses are identical except for the final action token — one says "qualify," the other says "deprioritize." 99 tokens are the same. 1 token differs. After dividing by 100, the preference gradient contribution is approximately 0.01.

At standard learning rates, 0.01 falls below the optimizer's effective noise floor. The parameter update is computed, but it's small enough that floating-point rounding and other batch-level gradient contributions swamp it. The model receives no usable signal from dual-control pairs. It doesn't learn to correct the one token that's wrong, because the correction signal never survives the averaging step with enough magnitude to move weights.

---

### Why β Doesn't Help

The natural instinct is to raise β. β controls the balance between the preference term and the SFT cross-entropy term — a higher β should push harder on the preference signal.

The problem is that β scales the entire preference term, including its already-diluted magnitude. If Δlog_p is 0.01 because of the 1/T denominator, multiplying by β = 0.5 gives 0.005. Multiplying by β = 2.0 gives 0.02. Neither value has recovered the signal from below the noise floor.

β is a balance parameter. It doesn't touch the denominator. The denominator is where the problem lives.

---

### The Regression Is Mechanically Predictable

The −4.5 pt regression from dual-control training isn't surprising once you trace through the math. Dual-control pairs generate near-zero preference gradients. But they still contribute to the SFT cross-entropy term. The SFT term pushes the model to reproduce the chosen response's language — which is almost identical to the rejected response's language, since 99 of 100 tokens match. Small SFT-driven adjustments accumulate across steps, slightly shifting weights in directions not optimized for the preference task. Other batch examples push back in other small directions. Over 750 steps, the net effect is a consistent degradation: not catastrophic collapse, just steady erosion of the calibration that produced the baseline score.

More steps don't fix this. They deepen it. The broken calculation doesn't converge — it compounds.

---

### The Fix

The denominator must reflect the number of tokens actually carrying the preference signal, not the full sequence length.

For terminal-error pairs, override the loss mask before `compute_logps` so that `mask.sum(dim=1)` equals 1 instead of T:

```python
if is_terminal_only_pair(chosen_ids, rejected_ids):
    loss_mask = terminal_token_mask(chosen_ids)   # denominator = 1
else:
    loss_mask = full_sequence_mask(chosen_ids)    # denominator = T
```

With this change, the dual-control preference gradient is no longer diluted. The terminal token's signal reaches the optimizer at full magnitude.

Two follow-on adjustments:

- **Raise β to 0.5.** The preference signal is now real. It can be weighted meaningfully relative to the SFT term.
- **Add gradient clipping at `max_norm=0.5` for the first ~200 steps.** The unmasked terminal gradient will be larger than the trainer has seen. Clipping prevents a spike in the terminal token's logits while the optimizer adjusts to the new signal magnitude.

Apply this before the 750-step dual-control run. Without it, more training deepens the regression.

---

### The Broader Principle

ORPO's per-token averaging is the right design for the general case. It solves a real problem: length-biased gradients that would otherwise make long responses dominate short ones regardless of quality.

But general-case design decisions create edge-case failure modes. When your preference pairs are structurally asymmetric — when some pairs have distributed errors and others have concentrated errors at a single position — the same normalization that stabilizes the general case silently zeroes out the signal for the concentrated case.

The fix isn't to remove the averaging. It's to make the denominator reflect the actual structure of each pair. That's a one-function change to `compute_logps`, and it restores the gradient signal that the averaging was inadvertently discarding.

---

### References

1. Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model. *arXiv:2403.07691*
2. Rafailov, R., et al. (2023). Direct Preference Optimization. *arXiv:2305.18290*
3. Hugging Face TRL. (2024). *ORPOTrainer documentation*. huggingface.co/docs/trl/orpo_trainer
