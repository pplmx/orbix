# Performance expectations

Orbix is a learning project, and these are the shapes to expect:

- **CPU-only.** No GPU support. For GPU throughput, use vLLM, SGLang, or TensorRT-LLM.

- **Prefill throughput improves with prompt length, within the window.** On Llama 3.2 1B Q2_K
  with 32 workers, the per-token cost fell from 52 ms at a 155-token prompt to 40 ms at 614
  (medians of five interleaved runs). Attention is O(n^2) but parallelises over positions, while
  the projections are O(n) over columns that do not grow, so the growing term is the one that
  gains workers.

- **Treat the numbers here as indicative, not as a benchmark.** They came from a shared machine
  whose load drifted more than the effects measured. `orbix bench` prints its own conditions.

- **Reference kernels are not optimized per instruction.** No SIMD and no assembly, because a
  kernel that cannot be read in the repository's own language defeats the point. The speed
  comes from parallelism over independent outputs instead, which took decoding on stories15M
  from 65 to 235 tok/s on a 128-core EPYC.

- **Parallelism is over independent outputs, never over a reduction.** The split picks which
  goroutine computes which output element -- rows for a matvec, columns for a transposed product
  -- and never splits the dot product, so float32 addition is never reassociated and the
  parallel result is bit-identical to the serial one. A split reduction would be faster on paper
  and would silently change every number compared against llama.cpp.

- **A model runs in its trained context, and no larger.** The rope table is built for that
  window, so `--ctx` accepts a smaller value and refuses a larger one. Going past it needs rope
  scaling, which this engine does not implement. A budget larger than the window is cut to fit,
  and `--stats` says when that happened.

- **Scheduling turns concurrency into throughput.** A round prefills every pending prompt and decodes every active sequence in one pass, so total throughput scales with concurrency: on stories15M-q4_0 with 32 generated tokens per request it goes 239 tok/s at one stream to 655 tok/s at sixteen, medians of seven interleaved runs. Per-stream speed still drops as streams are added, because the model is one shared resource.

- **A shared prompt costs almost nothing the second time.** With `--prefix-cache`, a request
  whose prompt starts with blocks the pool already computed attaches them instead of prefilling
  again: 101.0 tok/s against 139.5 for four sequential requests sharing one prompt, with the 96
  reused positions reported. This is the case a chat application has -- a system prompt and
  earlier turns re-sent verbatim.

The intended comparison is not "Orbix vs vLLM on an A100". It is "what does it take to make LLM inference work in pure Go, and what do I learn by building it?"

## Roadmap

The roadmap is a learning path as much as a feature list. Each version adds one conceptual layer.

- [x] `v0.1` — GGUF loader, F32/F16/BF16/Q4_0/Q8_0 decode, reference kernels, SentencePiece
  tokenizer, Llama forward pass, `orbix run` and `model inspect`.
- [x] `v0.2` — Continuous batching and a paged KV cache with prefix reuse, the 32-weight block
  formats, Qwen2.
- [x] `v0.3` — `orbix serve` with SSE, client-disconnect cancellation, the `priority` field,
  Mistral's sliding window, chat templates, mixture of experts.
- [x] `v0.4` — The five K-quants a checkpoint carries (`Q2_K`-`Q6_K`) plus IQ4_NL, and a prefix
  index that records which block continues which, so eviction gives up a chain's end.
  `Q8_K` is the one K-quant not decoded, because it is a conversion-tool intermediate.
- [x] `v0.5` — `bench --suite`, `/metrics`, `--otlp-endpoint`, `serve --log-format`.
- [x] `v1.0` — A CI matrix building five targets and testing two, and the reading path this
  file documents. "Stable" means the interfaces are settled enough to read as a whole.

Every version on that list is complete, which means less than a green checklist suggests: each
layer has an implementation correct against its reference and small enough to read. It does not
mean production-ready -- there is no GPU path, no queuing beyond one process, and no architecture
beyond the Llama family. The roadmap was a reading order, and it is finished.

## Design principles

1. **No CGo, ever.** `CGO_ENABLED=0` is the only supported configuration.
2. **Scheduling and caching are first-class.** The kernel is not the whole engine; continuous
   batching and KV reuse are what separate a serving engine from a script.
3. **GGUF is the model format.** No proprietary formats, no Python conversion step.
4. **Go conventions are load-bearing.** Explicit interfaces, directional dependencies,
   `context.Context` everywhere, no reflection-heavy magic.
