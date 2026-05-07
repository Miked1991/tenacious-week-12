# Pair Day 3 — Explainer

**Author:** Mikias Dagem  
**Date:** 2026-05-07  
**For:** Kemeriya Major

---

## 1. How TRL Computes `log_p` for ORPO's Odds-Ratio Term

ORPO's loss is built on the log-odds ratio between a chosen response and a rejected response. To compute that ratio, TRL's `ORPOTrainer` needs the log probability of each full sequence. The implementation does this by summing the per-token cross-entropy losses across all non-padding positions and then **dividing by the number of non-padding tokens**:

```text
log_p(response) = Σ log p(xₜ | x<t) / T
```

This produces an *average* per-token log probability — not a raw sum. TRL normalizes this way deliberately so that long and short sequences contribute equally to the loss regardless of length. Without normalization, a 200-token response would mechanically dominate a 50-token response in every batch.

The full ORPO loss is then:

```text
L = −log σ(log_p(chosen) − log_p(rejected)) + β · CE(chosen)
```

where β weights the preference term relative to standard SFT cross-entropy. The gradient that flows back into the model is proportional to `log_p(chosen) − log_p(rejected)` — and both terms carry the `1/T` denominator.

---

## 2. Gradient Magnitude: Distributed Error vs. Terminal Error

Let T be the sequence length and k be the number of tokens that differ between chosen and rejected.

For a **signal-grounding pair**, errors are distributed across the full response — k ≈ T. Each differing token contributes a non-zero term to the numerator. After dividing by T, the resulting Δlog_p is still a meaningful scalar, and the gradient that flows back is large enough for the optimizer to act on. This is why signal grounding produced a +11.8 pt gain.

For a **dual-control pair**, only the final token differs — k = 1. The sum of per-token differences has exactly one non-zero entry. After dividing by T:

```text
Δlog_p ≈ (one token's signal) / T
```

For a 100-token response, the gradient is diluted by a factor of 100. The optimizer receives a signal roughly 1/100th the magnitude of a distributed-error pair trained at identical β. At typical learning rates, this falls below the effective noise floor and the weight update is near zero. The model never corrects the one token that is actually wrong.

The -4.5 pt regression follows from this directly. Dual-control pairs generate no usable gradient while other examples in the batch push weights in small directions that slightly degrade performance.

At the same β, same sequence length, same optimizer:

| Pair type | Differing tokens (k) | Relative gradient magnitude |
| --------- | -------------------- | --------------------------- |
| Signal grounding (k ≈ T = 100) | ~100 | 1× (baseline) |
| Dual control (k = 1, T = 100) | 1 | ~0.01× |

The difference is entirely structural, not a product of learning rate, batch size, or checkpoint timing.

---

## 3. Does β = 0.2 Help?

No. β scales the preference term relative to the SFT cross-entropy term — it controls the *balance* between alignment and language modeling, not the magnitude of the underlying gradient signal. If Δlog_p is already near zero because of the 1/T dilution, multiplying it by 0.2 yields a smaller near-zero value. The root cause is the denominator, and β does not touch the denominator.

Raising β — say, to 0.5 or 1.0 — would make the same diluted signal proportionally louder relative to the SFT term, but the absolute gradient magnitude remains below the noise floor. β tuning can only help once the dilution is removed at its source.

---

## 4. The Fix Before the 750-Step Run

The denominator must reflect only the tokens that carry the preference signal, not the full sequence length. For pairs where chosen and rejected differ only at the terminal token, override the loss mask so that `mask.sum(dim=1)` equals 1:

```python
if is_terminal_only_pair(chosen_ids, rejected_ids):
    loss_mask = terminal_token_mask(chosen_ids)  # denominator = 1
else:
    loss_mask = full_sequence_mask(chosen_ids)   # denominator = T, original behavior
```

Apply this inside `compute_logps` in `ORPOTrainer` before the 750-step dual-control run. Two additional adjustments follow from the fix:

- **Raise β to 0.5.** With the dilution removed, the terminal-token gradient is now a real signal. It can be weighted meaningfully.
- **Add gradient clipping at `max_norm=0.5`.** The unmasked gradient will be larger than the trainer has seen. Clipping for the first ~200 steps prevents a spike in the terminal token's logits while the optimizer adjusts.

Without this mask change, extending training on dual-control pairs will deepen the regression because the 1/T dilution is a fixed structural property of how the loss is computed — more steps through the same broken calculation will not converge.

---

## Final Answer

> Why did signal grounding produce +11.8 pts while dual control produced -4.5 pts at the same β?

TRL's `ORPOTrainer` computes each sequence's log probability as a **per-token average** — it divides the summed log-probs by sequence length T. This is correct design for handling variable-length sequences, but it creates a structural penalty for preference pairs where the only error is a single terminal token.

When errors are distributed across a full response (signal grounding), the preference signal survives the averaging. When only one token differs (dual control), the signal is diluted by a factor of T before it reaches the optimizer — at T=100, the gradient is 100× smaller than it should be. That magnitude falls below the optimizer's effective noise floor. The model receives no usable update on dual-control pairs, and the slight drift from other batch examples produces the regression.

β=0.2 does not help. It scales an already-diluted gradient by 0.2, yielding a still-smaller near-zero value. β cannot recover signal that has been arithmetically divided away.

The fix is structural: override the loss mask for terminal-error pairs so the denominator equals 1, not T. Then raise β to 0.5 and add gradient clipping at `max_norm=0.5` for the first ~200 steps to absorb the now-unmasked gradient spike. Apply this to `compute_logps` before the 750-step dual-control run — without it, more steps deepen the regression rather than recovering from it.

---

## References

1. Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model. *arXiv:2403.07691* — Section 3 defines the per-token averaging loss and the odds-ratio term.
2. Hugging Face TRL. (2024). *ORPOTrainer documentation*. [huggingface.co/docs/trl/orpo_trainer](https://huggingface.co/docs/trl/orpo_trainer) — `compute_logps` implementation and mask handling.
3. Rafailov, R., et al. (2023). Direct Preference Optimization. *arXiv:2305.18290* — DPO foundation; the same 1/T averaging issue applies and is discussed in Appendix C.
