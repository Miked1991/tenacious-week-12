# Sources

**Author:** Mikias Dagem  
**Date:** 2026-05-06

---

## Two Canonical Sources

- **BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models (ICLR 2026)**
  The primary empirical basis for the explainer's core claim. Found that semantic alignment between user queries and tool descriptions is the strongest driver of tool selection in function-calling LLMs — stronger than the tool name itself. Also established that small perturbations to descriptions can flip tool choices, and that repeated pre-training exposure to a single description amplifies provider-level bias.

- **Tool Preferences in Agentic LLMs are Unreliable (arXiv:2505.18135v2)**
  The source for the 10× usage finding: tools with carefully edited descriptions receive more than ten times the usage of tools with their original descriptions, with names and parameters held constant. This result isolates the causal effect of the description as a variable, ruling out name or schema effects.

---

## Tool / Pattern Used

**Token-level probability analysis of autoregressive generation** — tracing how the model's next-token probability distribution is shaped by the full input context (user query + all tool definitions) before a single output token is produced. The explainer uses this mechanism to show that there is no isolated name-selection stage: descriptions are embedded as tokens and influence the probability mass over tool name tokens from the first generation step, not after the name is already committed.
