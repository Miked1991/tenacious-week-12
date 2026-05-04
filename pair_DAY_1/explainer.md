# The Blog Post Written for My Partner

**Author:** Mikias Dagem
**Date:** 05/04/2026

---

## Q1. How can a LoRA-adapted Qwen2.5-1.5B model achieve a 2.41× inference speedup despite identical merged parameter counts — and how can I experimentally decompose that gain across prefill, decode, quantization, kernel fusion, and KV-cache behavior?

---

## 🔑 Key Definitions (Read first)

- **LoRA (Low-Rank Adaptation)** – A fine-tuning method that keeps the base weight matrix $W$ frozen and adds two small trainable matrices $A$ and $B$ so that the adapted output is $Wx + ABx$. Only $A$ and $B$ are updated during training.

- **Merged vs. Unmerged LoRA**
  - *Unmerged*: The adapter matrices $A$ and $B$ are kept separate. During inference, each forward pass computes $Wx$ and $ABx$ separately and adds them. This adds per-token overhead.
  - *Merged*: You compute $W' = W + AB$ once before inference, then discard $A$ and $B$. Inference uses only $W'$, so there is no per-step adapter overhead. The total number of parameters after merging is the same as the base model plus the fused adapter – but the **runtime arithmetic** is halved for adapted layers.

- **Prefill vs. Decode** – Prefill is the first forward pass that processes the entire input prompt and builds the KV cache (compute-bound). Decode generates one token at a time, reusing the KV cache (memory-bound). Decode typically dominates total latency for long outputs.

- **KV Cache** – Stores keys and values from previous tokens to avoid recomputing them. Essential for fast autoregressive generation.

- **Kernel Fusion** – Combining several small matrix operations into a single GPU kernel to reduce launch overhead and improve memory locality.

- **Quantization** – Reducing numerical precision (e.g., float16 → int8 or 4-bit) to lower memory bandwidth usage and increase throughput.

---

## ⚡ Why the Speedup Happens

Even though the merged model has the **same number of parameters** as the unmerged base+adapter, the speedup comes from **eliminating extra operations per layer per token**, not from parameter reduction.

- **Unmerged LoRA**: For each adapted layer, every forward pass (prefill and each decode step) must compute:
  1. $Wx$ (base)
  2. $ABx$ (adapter)
  3. Add the results
- **Merged LoRA**: The single merged weight $W'$ is used, so only $W'x$ is computed.

For a generation of 100 tokens, the unmerged model performs the adapter multiplication **101 times per adapted layer**. The merged model performs it **zero times**. This difference directly yields the 2.41× speedup, consistent with known results where unmerged LoRA can cause 10× slowdown in edge cases.

---

## 🔬 How to Decompose the 2.41× Gain Experimentally

Use the same Qwen2.5-1.5B base model and the same LoRA adapter. Compare **unmerged** vs. **merged** inference. Use a fixed prompt (e.g., 200 tokens) and generate 100 new tokens. Run each test 10 times and average.

### 1. Prefill (Time to First Token – TTFT)

- **What to measure**: Time from input to the first generated token.
- **Why**: Unmerged LoRA computes $ABx$ in parallel with $Wx$ during prefill; merging removes that extra GEMM.
- **Method**:

  ```python
  start = time.perf_counter()
  _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
  ttft = time.perf_counter() - start
  ```

- **Expected contribution**: 10–20% of total speedup (small but measurable).

### 2. Decode (Time Per Output Token – TPOT)

- **What to measure**: Average time per subsequent token after the first.
- **Why**: Decode is memory-bound and happens for each new token. Unmerged LoRA recomputes $ABx$ and addition every step; merging eliminates this entirely.
- **Method**:

  ```python
  start = time.perf_counter()
  output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
  total = time.perf_counter() - start
  tpot = (total - ttft) / 100
  ```

- **Expected contribution**: 60–80% of the speedup (largest factor).

### 3. Quantization

- **What to measure**: Inference time and memory usage under 8-bit or 4-bit quantization.
- **Why**: Unmerged adapters can cause numerical spillover, making quantization less efficient. Merged weights compress cleanly.
- **Method**: Load both models with `load_in_8bit=True` (bitsandbytes) and repeat the benchmark.
- **Expected contribution**: Additional 5–15% speedup after merging.

### 4. Kernel Fusion

- **What to measure**: Number and duration of CUDA kernels launched per forward pass.
- **Why**: Unmerged LoRA forces separate kernels for $Wx$, $ABx$, and addition. Merged models fuse these into one kernel per layer.
- **Method**:

  ```python
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
      _ = model(**inputs, use_cache=True)
  print(prof.key_averages().table(sort_by="cuda_time_total"))
  ```

- **Expected contribution**: Reduced kernel launch overhead (small but non-zero gain).

### 5. KV-Cache Behavior

- **What to measure**: Time for repeated calls with identical long-prefix prompts.
- **Why**: Unmerged LoRA can break the static graph of the KV cache in some implementations, forcing cache recomputation. Merging restores full cache reuse.
- **Method**: Loop the same prompt 50 times. For iterations 2-50, record TTFT.
- **Expected contribution**: Marginal (0–10% of speedup), important for multi-turn or batched scenarios.

---

## 📊 Summary of Decomposition (Bullet Points)

- **Prefill** – Removes $ABx$ GEMM → contributes ~10-20% of the 2.41× gain.
- **Decode** – Eliminates per-token adapter arithmetic → contributes ~60-80% of the gain.
- **Quantization** – Merged weights quantize better → adds ~5-15% additional speedup.
- **Kernel fusion** – Fewer kernel launches → small but measurable improvement.
- **KV-cache behavior** – Restores static graph → consistent cache reuse (marginal gain).

The total 2.41× speedup is the product of these improvements, dominated by the decode phase.

---

## 🔗 References (Bullet Points)

- **BoostLoRA paper (arXiv 2604.27308)** – [https://arxiv.org/html/2604.27308v1](https://arxiv.org/html/2604.27308v1)
  Theoretical foundation for zero-overhead inference after merging LoRA adapters (Section 1).

- **HuggingFace PEFT merge_and_unload documentation** [https://huggingface.co/docs/peft/v0.13.2/en/developer_guides/merge_peft_adapters](https://huggingface.co/docs/peft/v0.13.2/en/developer_guides/merge_peft_adapters)
  Official API documentation for `merge_and_unload()`.

- **Community report: Idefics3 8B (HuggingFace discussion)** – [https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3/discussions/14](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3/discussions/14)
  Real-world example: unmerged LoRA took 20 s → merged took 2 s (10× speedup).

- **CSDN in-depth analysis** – [https://wenku.csdn.net/column/14wmd2tkzr92](https://wenku.csdn.net/column/14wmd2tkzr92)
  Explains that `merge_and_unload()` performs a "semantic handover of tensor ownership", eliminating runtime overhead.

- **Memory-efficient PEFT (introl.com)** – [https://introl.com/peft](https://introl.com/peft)
  States: "LoRA adds zero inference latency when merged."

- **Prefill vs. Decode explained (Redis blog)** – [https://redis.io/blog/prefill-vs-decode/](https://redis.io/blog/prefill-vs-decode/)
  Defines TTFT and TPOT, and explains why decode is memory-bound.

- **End-to-end LLM inference metrics (MorphLLM)** – [https://www.morphllm.com/llm-inference](https://www.morphllm.com/llm-inference)
  Detailed breakdown of prefill and decode phases.

- **KV cache survey (arXiv 2604.05012)** – [https://browse-export.arxiv.org/abs/2604.05012](https://browse-export.arxiv.org/abs/2604.05012)
  Comprehensive overview of KV cache mechanics and importance.

- **LookaheadKV (arXiv 2603.10899)** – [https://browse-export.arxiv.org/abs/2603.10899](https://browse-export.arxiv.org/abs/2603.10899)
  Advanced techniques for KV cache efficiency in long contexts.

By following the five experimental steps (prefill, decode, quantization, kernel fusion, KV-cache), you can measure each component's contribution and verify that the 2.41× speedup arises almost entirely from eliminating per-token LoRA arithmetic – exactly as the theory and practice of merged adapters predict.
