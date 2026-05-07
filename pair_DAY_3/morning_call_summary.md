# Morning Call Summary — Pair Day 3

**Pair:** Mikias Dagem & Kemeriya Major
**Written by:** Mikias Dagem
**Date:** 2026-05-07

---

My question going into the call was a symptom, not a diagnosis — something like "why is my eval loss so low at 0.0213?" Kemeriya immediately pushed back: is the eval set actually held out, or does it share the same underlying probes as the training set? I knew the split was random at the example level but hadn't thought through what that meant structurally. Once she named it — 8 variants of each probe going into train, 2 into eval, all from the same underlying company profile — the question became precise. It wasn't about low loss in general. It was about whether a variant-level split turns eval loss into a memorization metric rather than a generalization metric, and what should replace it for checkpoint selection given that a clean held-out set already exists. Adding that last condition changed the question from "how do I get better eval" to "why am I not using what I already have." That's what made it researchable.

Kemeriya's question had the same problem from the other direction — she was describing an outcome, not a mechanism. Her draft was "why did dual-control regress after ORPO training?" I asked her what she had in the repo that could actually explain it. She pointed to a β change in her memo, from 0.1 to 0.2, but hadn't connected it to the regression numbers. I pushed further: does the memo say what β does at the gradient level, not just what it's intended to do? She couldn't answer that without knowing the mechanics. So we rebuilt the question around the specific tension between the β=0.2 justification in `memo_07_orpo.md` and the −4.5 pt dual-control result in `submission_report.md` Section 7, replaced "why did it regress" with three competing hypotheses — gradient overshoot, data sparsity in the dual-control dimension, early-stop artifact — and added the diagnostic condition: what would the loss curves look like under each one?

Both questions were locked before the call ended.
