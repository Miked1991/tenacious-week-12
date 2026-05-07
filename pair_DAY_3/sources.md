# Sources

**Author:** Mikias Dagem
**Date:** 2026-05-07

---

## Two Canonical Sources

- **ORPO: Monolithic Preference Optimization without Reference Model — Hong et al. (2024), arXiv:2403.07691**
  The primary mechanistic source. Section 3 defines the per-token averaging loss: log probability is computed as the sum of per-token log-probs divided by sequence length T. This is the denominator that creates the structural gradient asymmetry between distributed-error pairs (k ≈ T) and terminal-error pairs (k = 1). The paper also defines the full ORPO loss combining the odds-ratio preference term with standard SFT cross-entropy, and explains the role of β in balancing the two.

- **HuggingFace TRL — ORPOTrainer documentation and `compute_logps` implementation**
  [huggingface.co/docs/trl/orpo_trainer](https://huggingface.co/docs/trl/orpo_trainer)
  Confirms how the 1/T averaging is implemented in practice: `compute_logps` sums per-token cross-entropy across non-padding positions and divides by `mask.sum(dim=1)`. Used to show that the fix — overriding the mask for terminal-error pairs so the denominator equals 1 instead of T — is a localized change to one function, not a rewrite of the training loop.

---

## Tool / Pattern Used

**Per-token gradient magnitude decomposition** — computing the ratio of Δlog_p between two pair types at fixed β, sequence length, and optimizer settings to show that the gradient magnitude difference is a deterministic function of how many tokens differ between chosen and rejected, not a product of learning rate, batch composition, or checkpoint timing. The key calculation: with T = 100 and k differing tokens, the preference gradient contribution scales as k/T. For signal-grounding pairs (k ≈ 30), this is ~0.30; for dual-control pairs (k = 1), this is 0.01 — a 30× difference that places dual-control gradients below the optimizer's effective noise floor at standard learning rates.
