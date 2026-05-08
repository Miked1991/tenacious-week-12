# Sources

**Author:** Mikias Dagem
**Date:** 2026-05-08

---

## Two Canonical Sources

- **Haldar, R., & Hockenmaier, J. (2025). Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks. EMNLP 2025.**
  The primary mechanistic source for intra-rater reliability in LLM judges. Documents that popular LLM judges have unexplained score variance exceeding 90% for certain models — meaning the score is dominated by sampling noise and surface-feature biases rather than response quality. Provides the theoretical grounding for why a single-call judge score is a sample from a distribution, not a measurement, and why that distribution's width matters before any benchmark figure can be interpreted as meaningful.

- **Bellibatlu, R. R., et al. (2026). JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems. arXiv:2604.23478.**
  The source for the Judge Sensitivity Score (JSS) metric and the ≥ 0.85 threshold. Defines JSS as the fraction of examples where the judge's verdict is stable across surface-level rephrasing of the scoring prompt, and provides the perturbation protocol (10 rephrased prompt variants per example) used in the three-step validation sequence delivered to Ephrata.

---

## Tool / Pattern Used

**Per-component IRR decomposition** — separating intra-rater stability (does the judge agree with itself?) from inter-rater agreement (does the judge agree with a reference standard?) before running either diagnostic. The key insight driving the explainer structure: a judge that passes intra-rater stability can still fail inter-rater agreement if its biases are systematic and consistent. Collapsing both into a single "reliability" question produces a test that can pass for the wrong reason — a perfectly stable but consistently wrong judge scores 100% on intra-rater stability and 0% on what actually matters for benchmark validity. Separating the two forces a diagnosis at the right level before any threshold (JSS ≥ 0.85, QWK ≥ 0.80) is applied.
