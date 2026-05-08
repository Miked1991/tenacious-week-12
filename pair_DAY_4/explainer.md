# Pair Day 4 — Explainer

**Author:** Mikias Dagem
**Date:** 2026-05-08
**For:** Ephrata Wolde

---

**Question received from Ephrata:** My tenacious-bench uses an LLM as the judge to score model outputs. I report those scores as if they are reliable measurements, but I never tested whether the judge itself is consistent. What is inter-rater reliability, how do you measure whether an LLM judge is trustworthy, and what does an unreliable judge do to benchmark results?

---

## 1. What Inter-Rater Reliability Actually Measures for an LLM Judge

Inter-rater reliability (IRR) asks a single question: if you ran the same evaluation twice — same prompt, same response, different call — would you get the same score? For human raters, this is a question about individual judgement consistency. For an LLM judge, it is a question about whether the score is a function of the response's quality or a function of irrelevant variables: token sampling randomness, positional ordering of options, prompt phrasing, or context window content that has nothing to do with what's being evaluated.

IRR in this context has two components that need to be measured separately:

**Intra-rater reliability** — does the same judge return the same verdict on the same input across repeated calls? At temperature > 0, an LLM judge samples from a distribution at the scoring token. If that distribution is diffuse, two calls on identical input can return different scores. The score is not a measurement; it is a sample.

**Inter-rater reliability** — does your judge agree with a separate judge (another model, or a human expert) applying the same rubric? Low inter-rater agreement means the score is not anchored to the rubric's intended criteria — it reflects whatever the judge model has learned to associate with high-quality text, which may not match what the rubric is actually trying to capture.

High intra-rater reliability is a necessary precondition, not a sufficient one. A judge can be perfectly consistent and consistently wrong if it is systematically biased toward specific surface features.

---

## 2. How to Measure Trustworthiness Before Trusting the Leaderboard

The goal is not to collect every available IRR metric — it is to answer whether the scores in `benchmark_results/tenacious_bench_v0.1.json` are measuring what they claim to measure.

**Intra-rater stability (run first):** Re-score a random sample of 50 examples with the same judge at the same temperature and compare outputs. Any disagreement is sampling noise, not signal. If the judge is deterministic (temperature = 0, greedy decoding), this passes by construction — but it still tells you nothing about bias.

**Sensitivity to irrelevant variation (run second):** Rephrase the scoring prompt surface without changing its semantic content — swap "Score from 1 to 5" for "Rate on a scale of 1 to 5", reorder the rubric criteria, change capitalization. A judge whose scores shift on these perturbations is treating prompt surface as evidence about response quality. The Judge Sensitivity Score (JSS) quantifies this: it is the fraction of examples where the verdict is stable across prompt rephrasing. A JSS below 0.85 on a 5-point scale means you have a noisy instrument.

**Agreement with a reference standard (run third):** Score the same 50 examples with a second judge — a different model, or a small set of human annotations. Compute Quadratic Weighted Kappa (QWK) between the two judge series. QWK is appropriate here because your rubric is ordinal: a score of 3 is closer to a score of 4 than to a score of 1, and the metric should penalize large disagreements more than small ones. For a benchmark making selection decisions (checkpoint comparisons, model ranking), aim for QWK ≥ 0.80. Below 0.60, the judge is not reliably separating quality levels.

---

## 3. Why an Unreliable Judge Inflates Scores Systematically, Not Randomly

This is the part that is not intuitive. If a judge is noisy, you might expect the noise to cancel out across enough examples — some scores too high, some too low, net effect near zero. That is not what the literature shows, and there is a structural reason it cannot be true.

LLM judges have learned biases from pretraining that operate in one direction on specific surface features. Position bias is the clearest example: when a judge scores two candidate responses presented in order, it systematically prefers the response in the final position because its training data had more text in which the last speaker's contribution is treated as authoritative (conclusion, resolution, summary). This bias is not random — it is a consistent directional shift. Studies find up to 30% systematic evaluation deviation from position effects alone.

Length bias operates the same way: judges trained on human text where longer responses correlate with effort tend to award higher scores to longer outputs, regardless of content quality. If your benchmark's held-out responses are longer than average, the judge will systematically over-score them.

The consequence is that the corruption is not detectable by looking at score distributions. A systematically inflated leaderboard looks exactly like a legitimately high-scoring leaderboard. The only way to detect it is to measure agreement against a reference that does not share the same biases — a human judge, or a judge trained on different data.

---

## 4. The Minimum Viable Protocol for Tenacious-Bench

Given that `tenacious_bench_v0.1.json` reports an 81.4 mean rubric score and the judge has not been validated, three checks establish whether that number is trustworthy before it is used to make any model selection decisions:

1. **Temperature audit:** Confirm the judge runs at temperature = 0. If not, switch it and re-score the 50-example sample. Record whether any scores change.
2. **Perturbation sweep:** Run the 50-example sample through three rephrasings of the scoring prompt. Compute JSS. If JSS < 0.85, the score variance is coming from prompt surface, not response quality — the 81.4 figure is partially noise.
3. **QWK against a human reference:** Annotate 30 examples yourself using the rubric. Compute QWK against the judge's scores on the same 30. If QWK < 0.70, the judge is not measuring the rubric's intended criteria reliably enough to use for model selection.

Until these three checks pass, the 81.4 figure cannot be interpreted as a measurement of model quality. It is a measurement of what the judge tends to award — which may or may not correlate with what the rubric is trying to capture.

---

## Final Answer

> If you use an LLM as a judge without testing its consistency, what specifically breaks?

An LLM judge without IRR validation produces scores that are a mix of response quality and the judge's learned surface-feature biases. The biases are directional, not random, so they inflate scores in systematic ways — position bias, length bias, and prompt-phrasing sensitivity each push scores in a consistent direction rather than introducing symmetric noise. Because the inflation is systematic, it is invisible in the score distribution: an inflated 81.4 looks identical to a legitimately earned 81.4.

The three things that break concretely are: model selection (you pick the checkpoint that best exploits judge biases, not the one that best satisfies the rubric), comparisons across runs (if prompt phrasing changed between runs, the score difference reflects prompt drift, not model improvement), and external reporting (a number published without a QWK figure against a human reference cannot be reproduced or challenged by anyone reading the paper).

The fix is not to distrust LLM judges — it is to run the three-step validation protocol before interpreting any score as a measurement. A judge that passes intra-rater stability, JSS ≥ 0.85, and QWK ≥ 0.80 is a reliable measurement instrument. One that has not been tested is a hypothesis.

---

## References

1. Haldar, R., & Hockenmaier, J. (2025). Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks. *EMNLP 2025*.
2. Bellibatlu, R. R., et al. (2026). JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems. *arXiv:2604.23478*.
3. Choi, J., et al. (2026). Diagnosing LLM Judge Reliability Using Item Response Theory. *arXiv:2602.00521*.
4. Landesberg, E. (2026). When LLM Judge Scores Look Good but Best-of-N Decisions Fail. *arXiv:2603.12520*.
