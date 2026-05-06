# Pair Day 2 — Explainer

**Author:** Mikias Dagem  
**Date:** 2026-05-06  
**For:** My pair partner Nahom Desalegn — a plain-language walkthrough of how tool descriptions drive tool selection in function-calling LLMs.

---

## Do Tool Descriptions Affect Which Tool Gets Selected?

Yes — and more than most people expect. Tool description strings are part of the selection computation at the token level. They are the **primary driver** of tool selection, not just a label the model reads after the decision is already made.

Here is how it actually works, step by step.

---

## 1. Tool Definitions Are Tokens That Shape the Model's State

Before generating any response, the model receives the full conversation context, which includes the user query and the complete definition of every available tool. Each tool definition contains a name, a description, and a JSON Schema for its parameters. A list of 30–40 tools can easily consume a couple of thousand tokens.

Those tokens are not passive background text. They are embedded into the model's hidden representations. By the time the model starts generating output, its internal state has already been shaped by every tool description — encoding how well each one aligns with the user's request.

There is no isolated "name selection stage" that happens before descriptions are read. The selection unfolds across multiple token predictions, all informed by the full context.

---

## 2. The Model Generates the Tool Name One Token at a Time

Transformer models are autoregressive — they produce output one token at a time. Selecting a tool looks roughly like this:

- The model generates a token that signals the start of a tool call (e.g. a special `<tool_call>` marker).
- It then generates the first token of the tool name (e.g. `hubspot`).
- It continues generating tokens until the full tool name is complete.

At every single step, the probability distribution over all possible next tokens is computed using the entire input context — including all descriptions. The tool name that gets chosen is the one with the highest conditional probability given that full context. Descriptions are baked into that probability from the start.

---

## 3. Descriptions Are the Primary Driver, Not a Post-Selection Check

Controlled research confirms this. The *BiasBusters* paper (ICLR 2026) found that **semantic alignment between the user query and tool metadata — especially descriptions — is the strongest driver of tool selection**. Small changes to a description can flip which tool gets chosen entirely.

A separate study found that **tools with well-edited descriptions receive over 10 times more usage than tools with their original descriptions**. That result was measured with tool names and parameters held constant, isolating the causal effect of the description alone.

---

## 4. Why Bad Descriptions Cause Wrong Tool Calls

When descriptions are vague or missing, the model's probability distribution across tool names becomes flat or incorrectly peaked.

Example: if you have two tools — `search_products` ("Searches product catalog by keyword") and `get_product_details` ("Retrieves detailed specification for a single product by ID") — and a user asks "find me a red backpack under $50," the model evaluates the full description context before committing to any name. A poorly written description for `search_products` could easily cause the model to call `get_product_details` instead, silently misrouting the request.

This is why Anthropic, OpenAI, and other providers explicitly recommend writing clear, precise descriptions: the model depends on them during both tool selection and argument generation.

---

## 5. Direct Answer to the Research Question

> At the token level, when a model using function-calling selects a tool, is the description string part of the selection computation — and if so, how — or is it processed only after the tool name token is already committed?

**Descriptions are part of the selection computation from the very first token.** They are not processed after the tool name is committed. Every tool definition — name, description, and parameter schema — is embedded as tokens in the input context. The model's step-by-step generation uses that full context to compute the probability of every possible next token. The tool name that gets generated is the one with the highest probability given everything in context, including the descriptions.

There is no separate selection stage that runs before descriptions are considered.

---

## Key Takeaway for Our System

When we build agents that select tools, writing clear descriptions is not optional polish — it is the mechanism the model uses to route correctly. Vague descriptions force the model to rely on tool names alone, which is a much weaker signal and leads to silent misroutes.

---

## References

1. Blankenstein, T., et al. (2026). *BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models.* ICLR 2026.
2. *Tool Preferences in Agentic LLMs are Unreliable.* arXiv:2505.18135v2.
3. Airbyte. (2026). *Agent Tools Explained: How Tools & Connectors Work in Agent Systems.*
4. OpenAI / Anthropic official documentation on function calling (2026).
