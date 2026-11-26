# Changelog

Orbix is a pure-Go, CGo-free LLM inference engine, and **this is its first tagged release**.
Everything below the summary is the full history, generated from 1393 commits: the project was
developed for a while before any version was cut, so `v0.1.0` is a starting point rather than
the sum of what changed since `v0.0.x`.

## What works today

- **GGUF**: headers through tensor data, container versions 2 and 3, and 14 quantized formats
  cross-checked against ggml's own block layout — F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0,
  Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, and IQ4_NL.
- **Models**: seven architectures — `llama`, `mistral`, `qwen2`, `qwen3`, `gemma`, `gemma2`,
  `gemma3`. Mixtral loads as `llama` with expert tensors, which is how its files declare it.
- **Inference**: continuous batching, a paged KV cache with prefix reuse, sliding-window
  attention per layer, and MoE routing.
- **Chat**: the checkpoint's own Jinja2 template, evaluated by a subset that the repository
  cross-checks against real Jinja2 rather than against hand-written expectations.
- **Serving**: an OpenAI-compatible API — completions, chat, streaming, tool calling, logprobs,
  stop sequences, and fill-in-the-middle — plus OTLP trace export and Prometheus metrics.

## What pre-alpha means here

The API may change between minor versions. The code is written for readability first: the
kernels are reference implementations, not tuned ones, and where a faster path existed it is
documented rather than assumed. This is a project for learning how inference works, and it says
so in the README.

Binaries for linux/amd64, linux/arm64, darwin/amd64, darwin/arm64 and windows/amd64 are built
with `CGO_ENABLED=0` and ship with per-file SHA-256 sums.
