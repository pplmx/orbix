# Orbix

**A pure-Go, CGo-free LLM inference engine — built to learn how LLMs actually work.**

GGUF loading, quantized kernels, KV cache, continuous batching, and OpenAI-compatible serving — one layer at a time, in Go.

[![Go Version](https://img.shields.io/badge/go-1.27%2B-00ADD8?logo=go)](https://go.dev)
[![CGo](https://img.shields.io/badge/cgo-disabled-brightgreen)](https://github.com/pplmx/orbix)
[![Status](https://img.shields.io/badge/status-learning--in--public-orange)](https://github.com/pplmx/orbix)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> **Status: learning project, pre-alpha.** The API will change. The code prioritizes readability and correctness over performance.

Orbix reads GGUF models, runs quantized transformer inference on CPU, schedules requests with
continuous batching, and serves an OpenAI-compatible API — from a single static binary with no
CGo, no shared libraries, and no external runtime.

```bash
CGO_ENABLED=0 go install github.com/pplmx/orbix/cmd/orbix@latest

orbix serve --model ./models/qwen2.5-7b-instruct-q4_0.gguf --addr :8080
```

That is the whole quick start. The server implements `/v1/chat/completions`,
`/v1/completions` (streaming and not), `/v1/models`, `/healthz` and `/metrics`, and any OpenAI
client library works against it.

## Documentation

| Document                                  | What it covers                                               |
| ----------------------------------------- | ------------------------------------------------------------ |
| [Architecture](docs/architecture.md)      | How the layers fit, the interfaces, and the reading order    |
| [Usage](docs/usage.md)                    | Serving, reverse proxies, CORS, cross-compiling              |
| [CLI reference](docs/cli.md)              | Every command, benchmark suites, the access log              |
| [Models and quantization](docs/models.md) | Supported architectures and formats, checkpoint-gated tests  |
| [Performance](docs/performance.md)        | What the numbers are, the roadmap, and the design principles |
| [Configuration](docs/configuration.md)    | Every flag and what it controls                              |
| [Contributing](CONTRIBUTING.md)           | Build commands, the gates, and the constraints CI enforces   |

## At a glance

| Layer          | Package                                         | What it is                                                    |
| -------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| GGUF container | [`internal/gguf`](internal/gguf/doc.go)         | Header, metadata, tensor directory, streamed tensor data      |
| Half precision | [`internal/f16`](internal/f16/doc.go)           | binary16, every one of the 65,536 patterns                    |
| Dequantization | [`internal/quant`](internal/quant/doc.go)       | F32/F16/BF16, Q4/Q5/Q8, IQ4_NL, Q2_K–Q6_K                     |
| Kernels        | [`internal/kernel`](internal/kernel/doc.go)     | MatMul, MatVec, RMSNorm, Softmax, SiLU                        |
| Tokenizer      | [`internal/tokenize`](internal/tokenize/doc.go) | SentencePiece and byte-level BPE                              |
| Chat templates | [`internal/chat`](internal/chat/doc.go)         | The Jinja2 subset a checkpoint's template needs               |
| Engine         | [`internal/engine`](internal/engine/doc.go)     | Continuous batching, admission policy, KV cache               |
| Transformer    | [`internal/llama`](internal/llama/doc.go)       | Llama, Qwen2/3 and Gemma 1/2/3 forward passes, RoPE, GQA, MoE |
| Tracing        | [`internal/otel`](internal/otel/doc.go)         | OTLP/HTTP+JSON spans, hand-written, no SDK                    |
| Server         | [`internal/server`](internal/server/doc.go)     | OpenAI-compatible API, SSE, `/metrics`, access log            |

## Why this exists

Most LLM inference code lives in Python + CUDA, wrapped in C++ kernels, and hidden behind
framework abstractions. That is the right choice for production and a hard place to *learn*.
Orbix asks whether inference can be understood by building it in Go, from the file format up.

- **Pure Go, no CGo.** `CGO_ENABLED=0` is a learning constraint here, not a deployment trick: if a
  C library does it, it has to be written out to be understood.
- **Readability over performance.** Kernels are written to be read. Assembly fast paths come later,
  behind the same interface.
- **One layer at a time**, so each can be studied independently.
- **Honest benchmarks.** No cherry-picked numbers; where Orbix is slower than llama.cpp,
  [the performance page](docs/performance.md) says so.

The cost, plainly: pure-Go CPU inference is slower than C++ with hand-tuned SIMD. That is not the
comparison this project is trying to win.

## License

MIT — see [LICENSE](LICENSE).
