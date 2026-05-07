# Evening Call Summary — Pair Day 3

**Pair:** Mikias Dagem & Kemeriya Major
**Written by:** Mikias Dagem
**Date:** 2026-05-07

---

Going through Kemeriya's variant-level leakage explainer, the probe_42 walkthrough was the part that actually worked — seeing 8 train variants and 2 eval variants of the same company profile laid out concretely made the mechanism land in a way that "data leakage" as a general concept hadn't. I also said the sentence "0.0213 is very likely contamination-driven memorization" was the most useful line in the explainer, because it gave me a specific claim to anchor to rather than a vague suggestion that something might be wrong. What I pushed back on was the fix section. It told me what to change — `metric_for_best_model` in `TrainingArguments` — but not how to wire it up. I didn't know what the custom rubric callback looked like inside Trainer or what key the metric needed to be logged under for the argument to find it. Kemeriya revised the section to add the two specific kwargs and a note that the callback must log under exactly `eval_rubric_score` via `trainer.log()` inside an `on_evaluate` method.

When Kemeriya went through my per-token averaging explainer, she said the gradient magnitude calculation was what she needed. I'd worked through the concrete ratio — roughly 30 differing tokens in a signal-grounding pair versus 1 in a dual-control pair, both 100 tokens long — and she said that number was what she couldn't derive from reading the memo on her own. The confirmation that β=0.2 doesn't help also landed: the memo's reasoning had implied that raising β would push categorical errors into correction, and I showed that β scales both sides of the loss proportionally without touching the 1/T denominator. What she pushed back on was the ending. The explainer diagnosed the dilution but stopped there. Her question was direct: now that she knows it's dilution, should she rewrite the dual-control pairs to be shorter, or is there a TRL flag for it? I added a fix section naming both options — shorter completions to reduce T, or token-level weighting to put high loss weight on just the action token — and flagged SimPO's length-normalized reward as an alternative, with a note that it drops the SFT component, which matters for catastrophic forgetting at 0.8B scale.

Everything was revised before the call ended. 