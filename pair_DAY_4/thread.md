# Pair Day 4 — Twitter Thread

**Author:** Mikias Dagem
**Date:** 2026-05-08
**Topic:** Why an untested LLM judge corrupts benchmark results — and what to run before you trust the numbers

---

**1/**
You ran a benchmark. Your LLM judge scored every output. You reported the result.

But you never tested whether the judge itself is consistent.

Here's what you're actually reporting. 🧵

---

**2/**
An LLM judge converts a model output to a score by sampling from a probability distribution over score tokens.

At temperature > 0, two calls on identical input can return different scores.

The score is not a measurement. It is a sample. Whether those samples cluster tightly or scatter determines whether your reported number means anything.

---

**3/**
"Reliability" breaks into two separate questions that need different tests.

**Intra-rater:** does the same judge agree with itself across repeated calls on the same input?

**Inter-rater:** does your judge agree with a reference standard — another model, or a human expert?

A judge can pass the first and fail the second. Stable ≠ accurate.

---

**4/**
The instinct is to assume noise averages out.

If the judge is wrong sometimes, it cancels across enough examples. Right?

It doesn't — because the failure mode is one-directional. A judge biased toward long responses awards more points to long responses every time. There is no opposing pressure. The inflation compounds, not cancels.

Studies find up to 30% systematic evaluation deviation from position bias alone. That is not noise in the signal. That is the signal being wrong.

---

**5/**
Two metrics that tell you what you need to know before you publish:

**JSS (Judge Sensitivity Score):** fraction of examples where the verdict is stable across prompt rephrasing. Run the same scoring prompt in 10 surface variants. Target ≥ 0.85.

**QWK (Quadratic Weighted Kappa):** agreement with a human reference, weighted by ordinal distance. Annotate 30 examples yourself and compute QWK against your judge. Target ≥ 0.80 for any decision that depends on the ranking.

These take an afternoon. Running them before publishing is the only way to know if your number is a measurement or an artifact.

---

**6/**
Until those pass:

The reported score is a hypothesis. Checkpoint selection on those scores picks the model that best exploits judge biases. Rankings are not reproducible — anyone running a different judge will get different numbers, and both of you will be equally wrong.

Test the ruler before trusting the measurements.

---

## Medium Post

### Your LLM Judge Has Never Been Tested — Here's What That Does to Your Benchmark

By Mikias Dagem | 2026-05-08

---

When you deploy an LLM as a judge in your evaluation pipeline, you are making an implicit claim: that the scores it produces are reliable measurements of response quality. That claim has a precondition. The judge itself must be tested before you trust what it produces. Most benchmarks skip this step. The consequences are predictable and severe, but they are invisible in the output — which is what makes them dangerous.

---

### The Sampling Problem

An LLM judge does not compute a score the way a rule-based function does. It samples from a probability distribution over score tokens. At temperature > 0, the token at the scoring position is drawn probabilistically — two calls on identical input can produce different scores. How different depends on how concentrated that distribution is at the scoring token. If the judge is highly confident, scores cluster tightly and the variance is negligible. If the judge is uncertain, scores scatter across the scale and the number you report is wherever the sampling happened to land on that particular call.

This makes a single-call judge score a sample, not a measurement. The question before you report any benchmark result is: what is the variance of that sample, and is it small enough to be ignored?

Most pipelines don't ask.

---

### Intra-Rater vs. Inter-Rater: Why You Need Both

Reliability in evaluation has two components that need to be diagnosed separately.

**Intra-rater reliability** asks whether the same judge returns the same score on the same input across repeated calls. This is a test of sampling stability — it tells you whether the judge's probability distribution over score tokens is concentrated or diffuse. A judge running at temperature = 0 with greedy decoding passes this test by construction. A judge at temperature = 0.7 may not.

**Inter-rater reliability** asks whether your judge agrees with a reference standard — another model, or a human expert using the same rubric. This is a test of validity: is the judge measuring what the rubric claims to measure, or is it measuring something else (text length, vocabulary sophistication, whether the response ends with a confident summary)?

A judge can pass intra-rater reliability and fail inter-rater reliability. A perfectly stable but systematically biased judge scores identically on every call and consistently wrong relative to human judgment. Passing intra-rater stability does not mean the scores are trustworthy — it only means they are reproducible. Reproducible wrong answers are still wrong.

---

### Why Bias Is Systematic, Not Random

The common intuition is that judge errors should average out. If the judge is wrong sometimes, the noise should cancel across enough examples. This is true for symmetric, random errors. It is not true for systematic biases.

LLM judges have documented biases that operate in one direction on specific surface features. Position bias is the clearest example: a judge scoring two candidate responses presented in sequence will systematically prefer the response in the final position, because its training data contains more text in which the last speaker's contribution is framed as the authoritative conclusion. This preference does not fluctuate — it applies consistently every time the judge sees a response in that position. There is no opposing case where the judge randomly disfavors the final position to balance it out.

Length bias operates the same way. A judge trained on human-written text where longer responses correlate with effort tends to award higher scores to longer outputs regardless of content quality. If your benchmark's model outputs are longer than the examples in your judge's calibration distribution, the judge will systematically over-score them.

The consequence is that corrupted benchmark scores are invisible by inspection. A mean rubric score inflated by position bias looks identical to a legitimately earned mean rubric score. You cannot detect the corruption by looking at the numbers. You can only detect it by measuring agreement against a reference that does not share the same biases.

---

### Two Metrics and One Protocol

The minimum viable validation protocol before trusting any benchmark result involves two metrics and three steps.

**Step 1 — Intra-rater stability.** Re-score a random sample of 50 examples with the same judge on a second call. If the judge is deterministic, this passes automatically and you can move on. If scores change, the variance is sampling noise — quantify it and decide whether it is acceptable given your reporting precision.

**Step 2 — Judge Sensitivity Score.** Rephrase the scoring prompt in 10 surface variants without changing its semantic content. Run the same 50 examples through each variant. Compute the fraction of examples where the verdict is stable across all 10. This is the JSS. Target ≥ 0.85. A JSS below that threshold means a meaningful fraction of your scores are determined by prompt surface rather than response quality — your number is partly a measurement of how you phrased the evaluation question.

**Step 3 — QWK against a human reference.** Annotate 30 examples yourself using the rubric. Compute Quadratic Weighted Kappa (QWK) between your annotations and the judge's scores on the same 30 examples. QWK is the right metric here because your rubric is ordinal: a score of 3 is closer to a score of 4 than to a score of 1, and the metric penalizes large disagreements more than small ones. For any benchmark used to make model selection decisions, target QWK ≥ 0.80. Below 0.60, the judge is not reliably separating quality levels, and any ranking it produces cannot be used to choose between models.

These three steps take an afternoon. They should be completed before any benchmark number is reported as a result.

---

### What Happens Without Them

Without validation, three things break in ways that are hard to recover from after the fact.

**Model selection fails.** Checkpoint selection on unvalidated scores picks the model that best exploits judge biases, not the model that best satisfies the rubric. If the judge favors long responses, the selected checkpoint is the one that learned to generate long text — which may or may not be the one that generates correct text.

**Cross-run comparisons are invalid.** If the scoring prompt changed between runs, any score difference reflects prompt sensitivity (measured by JSS) rather than model improvement. You cannot tell from the numbers alone whether a score went up because the model got better or because the rephrased prompt favored the outputs it was now generating.

**Published numbers are not reproducible.** A benchmark result published without a QWK figure against a human reference cannot be challenged or verified by an external reader. Anyone running a different judge will get different numbers. Both results will claim to measure the same rubric, and neither will have the validation data to adjudicate between them.

---

### The Rule

A judge that has not been validated is a hypothesis about what high-quality responses look like, encoded in a model's weights. It may be a good hypothesis. It may not. You do not know until you test it.

Run intra-rater stability. Run a JSS sweep. Compute QWK against 30 human annotations. If all three pass, you have a reliable instrument and your benchmark result is a measurement. If any one fails, you have an artifact — and you now know exactly what to fix before you report anything.

Test the ruler before trusting the measurements.

---

### References

1. Haldar, R., & Hockenmaier, J. (2025). Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks. *EMNLP 2025*.
2. Bellibatlu, R. R., et al. (2026). JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems. *arXiv:2604.23478*.
3. Choi, J., et al. (2026). Diagnosing LLM Judge Reliability Using Item Response Theory. *arXiv:2602.00521*.
4. Landesberg, E. (2026). When LLM Judge Scores Look Good but Best-of-N Decisions Fail. *arXiv:2603.12520*.
