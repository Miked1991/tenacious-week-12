# Pair Day 2 — Twitter Thread

**Author:** Mikias Dagem  
**Date:** 2026-05-06  
**Topic:** How tool descriptions drive tool selection in function-calling LLMs

---

**1/**
Most people think the model picks a tool by its *name*, then reads the description afterward.

That's backwards.

The description is what determines which tool gets called in the first place. Here's why that matters 🧵

---

**2/**
Before generating a single output token, the model ingests the full list of available tools — names, descriptions, and parameter schemas — all as tokens.

By the time it starts writing a response, its internal state has already been shaped by every description in that list.

There is no "name-first, description-second" stage.

---

**3/**
Transformer models generate output one token at a time.

Selecting a tool = generating the tool name token-by-token.

At every step, the model computes a probability distribution over all possible next tokens using the **full context** — including every tool description.

The name that gets generated is the one with the highest probability given all of that.

---

**4/**
The research backs this up hard.

The BiasBusters paper (ICLR 2026) found that semantic alignment between the user query and tool descriptions is the **strongest driver** of tool selection.

A separate study found tools with well-edited descriptions get **10× more usage** than tools with original descriptions — with names and parameters held constant.

---

**5/**
What happens when descriptions are vague?

The model's probability distribution over tool names goes flat or peaks on the wrong tool.

Two tools: `search_products` and `get_product_details`. Bad descriptions → the model silently calls the wrong one. No error. No warning. Just a misroute.

---

**6/**
The takeaway for anyone building agents:

Writing clear, precise tool descriptions isn't polish — it's the mechanism the model uses to route correctly.

Treat descriptions like function signatures for the model's decision-making, not like optional comments in your code.

---

## Medium Post

### Your Tool Descriptions Are Doing More Work Than You Think

By Mikias Dagem | 2026-05-06

---

When developers first encounter function-calling in LLMs, they naturally focus on tool names and parameter schemas. The description field often gets treated as documentation — something you fill in for the humans reading the code, not something the model actually uses to make decisions.

That assumption is wrong, and it explains a surprising number of silent routing failures in production agents.

---

## The Common Misconception

Most people's mental model goes something like this: the model reads the user's message, scans the list of tool names, picks the best match, and *then* reads the description to fill in the arguments.

In this model, descriptions are validators. They run after the decision is already made.

This is not how it works.

---

## What Actually Happens at the Token Level

Transformer models are autoregressive — they generate output one token at a time. Before generating anything, the model receives the entire input context: the user's message, conversation history, and the full definition of every available tool. Each tool definition includes its name, description, and JSON Schema.

A list of 30–40 tools can consume thousands of tokens. Those tokens are not passive. They are embedded into the model's hidden representations, shaping its internal state before a single output token is produced.

By the time the model starts generating a response, it has already processed every description. The tool name that eventually gets generated is the one with the highest conditional probability given that full context — descriptions included.

There is no isolated name-selection stage that runs before the descriptions are read.

---

## What the Research Shows

This isn't just a theoretical claim. Controlled studies have measured the causal effect of descriptions on tool selection directly.

The *BiasBusters* paper, published at ICLR 2026, found that **semantic alignment between user queries and tool descriptions is the strongest driver of tool selection** — stronger than the tool name itself. Critically, small perturbations to a description can flip which tool gets selected, even when the name remains unchanged.

A separate study found that **tools with carefully edited descriptions receive more than ten times the usage** of tools with their original descriptions. That result controlled for names and parameters, isolating the description as the variable.

The causal effect is well established. Descriptions do not inform the model after selection — they reshape the probability distribution that leads to selection in the first place.

---

## Why This Causes Silent Failures

When descriptions are vague, generic, or missing, the model's probability distribution over tool names becomes flat or incorrectly peaked.

Consider two tools: `search_products`, described as "Searches product catalog by keyword," and `get_product_details`, described as "Retrieves detailed specification for a single product by ID." If a user asks "find me a red backpack under fifty dollars," the model evaluates both descriptions before generating any name token. A poorly written description for `search_products` could easily shift probability mass toward `get_product_details`.

No exception is raised. No warning is logged. The wrong tool gets called, the wrong arguments get generated, and the user gets a confused or empty response.

This class of failure is invisible precisely because the model never "knows" it made the wrong choice — it made the highest-probability choice given the information it had.

---

## What to Do About It

Treat tool descriptions the way you treat function signatures in statically typed code: they are contracts, not comments.

A good description does three things:

1. **States what the tool does** in plain language that maps to the kinds of queries a user might make.
2. **Distinguishes it from similar tools** — especially any tool that handles adjacent use cases.
3. **Specifies what it does not do**, so the model doesn't route edge cases to it incorrectly.

Vague descriptions like "handles product operations" or "manages user data" force the model to rely on the tool name alone. That is a much weaker signal, and it leads to unreliable routing at scale.

---

## The Broader Principle

Descriptions are the interface between natural language and structured tool calls. The model uses them — not just as documentation, but as the primary mechanism for deciding what action to take.

If you are building agents that route across multiple tools, the quality of your descriptions is a direct lever on the reliability of your system. It is also one of the easiest levers to pull: no architecture changes, no retraining, just clearer writing.

Write descriptions as if the model will use them to make decisions. Because it will.

---

## References

1. Blankenstein, T., et al. (2026). *BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models.* ICLR 2026.
2. *Tool Preferences in Agentic LLMs are Unreliable.* arXiv:2505.18135v2.
3. Airbyte. (2026). *Agent Tools Explained: How Tools & Connectors Work in Agent Systems.*
4. Anthropic & OpenAI official documentation on function calling (2026).
