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

  How much of that a given `--workers` value buys, measured on gemma-3-270m-it-Q8_0 with a
  10-token prompt and 20 decode steps each, one process:

  | `--workers` | ms/token | speed-up |
  | ----------- | -------- | -------- |
  | 1           | 315.83   | 1.00x    |
  | 2           | 141.33   | 2.23x    |
  | 4           | 70.16    | 4.50x    |
  | 8           | 42.31    | 7.46x    |
  | 16          | 30.64    | 10.31x   |
  | 32          | 34.32    | 9.20x    |
  | 64          | 32.27    | 9.79x    |
  | 128         | 25.67    | 12.30x   |

  The curve is not monotone: 32 workers measured slower than 16, and 64 only just beat 16. The
  end of the table is the figure for a machine like this one, 12.3x on 128 cores.

  Why 32 sits above 16 is not established. `parts()` splits a projection into
  `min(workers, work/workPerWorker)` pieces, so the number of pieces stops growing once the work
  per piece would fall below 16384 multiply-adds, and a plausible reading is that the middle rows
  sit where more goroutines share a fixed total rather than each getting a full slice. That is an
  inference from reading the split rule, not a measurement -- the numbers above are what was
  measured. What the last row rests on is firmer: the kernels are bound by reading weights, so
  once the bus is saturated more workers cannot read the same bytes faster, and 12.3x is that
  ceiling rather than a scheduling choice.

  The default is 0, meaning `GOMAXPROCS`, so the last row is what an unconfigured server already
  does. A smaller value is for a host running several processes -- see the note on
  `--workers` in configuration.md.

- **Parallelism is over independent outputs, never over a reduction.** The split picks which
  goroutine computes which output element -- rows for a matvec, columns for a transposed product
  -- and never splits the dot product, so float32 addition is never reassociated and the
  parallel result is bit-identical to the serial one. A split reduction would be faster on paper
  and would silently change every number compared against llama.cpp.

- **A model runs in its trained context, and no larger.** The rope table is built for that
  window, so `--ctx` accepts a smaller value and refuses a larger one. Going past it needs rope
  scaling beyond the linear one: `rope.scale_linear` is applied, and any other
  `rope.scaling.type` is refused at load rather than ignored, because running a YaRN-trained
  model on the plain rotation produces plausible text instead of an error. A budget larger than
  the window is cut to fit, and `--stats` prints a note naming the window, the budget asked for
  and the count actually produced.

- **Scheduling turns concurrency into throughput.** A round prefills every pending prompt and decodes every active sequence in one pass, so total throughput scales with concurrency: on stories15M-q4_0 with 32 generated tokens per request it goes 239 tok/s at one stream to 655 tok/s at sixteen, medians of seven interleaved runs. Per-stream speed still drops as streams are added, because the model is one shared resource.

- **The same shape holds on a real checkpoint, not only on the tiny one.** Measured on
  gemma-3-270m-it-Q8_0 with a 4-token prompt and 30 generated tokens, one process, four load
  levels: total throughput rises with concurrency, and per-request latency rises with it too.

  | concurrent requests | tok/s | per-request latency |
  | ------------------- | ----- | ------------------- |
  | 1                   | 30.7  | 975 ms              |
  | 2                   | 48.2  | 1244 ms             |
  | 4                   | 69.8  | 1717 ms             |
  | 8                   | 82.4  | 2909 ms             |

  Every request in a burst finishes at the same time -- at concurrency 4 the four latencies were
  1724, 1724, 1724 and 1725 ms -- because a round advances all active sequences together. That
  is why throughput gains and per-request latency loses: the model is one shared resource, and
  more streams divide it.

  **This section used to say the reverse**: "latency per request fell from 1493 ms to 496 ms
  across that range and then flattened between four and eight". A shared model cannot make each
  request faster by adding requests to it, so that sentence was not a stale measurement but a
  description of something that cannot happen. The earlier throughput figures (20.1, 36.8, 60.5,
  61.5 tok/s) were replaced by the row above rather than kept beside it: they came from a run
  whose load was not recorded, and mixing two runs in one table is how this document acquired
  numbers nobody could reproduce. Both figures here come from one process, one checkpoint, and
  the same prompt as the table above.

  The plateau the old paragraph attributed to memory bandwidth is not in this data: eight
  concurrent requests are still 18% faster than four. Whether a plateau exists at 16 or 32 was
  not measured, and the paragraph no longer claims one for a machine nobody here controlled.

  Two further figures, kept because they are what a single request actually costs: decoding is
  28.3 ms per token, and a 64-token prefill is 276 ms batched against 1799 ms one token at a
  time -- a 6.5x gain from the sub-batching that the sequential path cannot have. Both come from
  `go test -bench` with `MODEL` set. They were measured in the same session as the throughput
  numbers that turned out to be wrong, so treat them as that run's figures rather than as
  re-measured ones; the commands reproduce them for whoever sets `MODEL`.

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
