# Orbix

**A pure-Go, CGo-free LLM inference engine — built to learn how LLMs actually work.**

GGUF loading, quantized kernels, KV cache, continuous batching, and OpenAI-compatible serving — one layer at a time, in Go.

[![Go Version](https://img.shields.io/badge/go-1.27%2B-00ADD8?logo=go)](https://go.dev)
[![CGo](https://img.shields.io/badge/cgo-disabled-brightgreen)](https://github.com/pplmx/orbix)
[![Status](https://img.shields.io/badge/status-learning--in--public-orange)](https://github.com/pplmx/orbix)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> **Status: learning project, pre-alpha.** Orbix is a personal project for learning LLM inference by building one in pure Go. The API will change. The code prioritizes readability and correctness over performance.

Orbix is a **pure-Go LLM inference engine** written to understand, layer by layer, what happens between "load a GGUF file" and "stream tokens over HTTP". It has no CGo, no shared libraries, and no external runtime. It reads GGUF models, runs quantized transformer inference on CPU, schedules requests with continuous batching, and serves an OpenAI-compatible API — all from a single static binary produced by `go build`.

This is not a production inference framework competing with vLLM or llama.cpp. It is a **learning vehicle**: a codebase where each layer of LLM inference is implemented explicitly in Go, small enough to read end to end, and structured so that every concept — quantization, attention, KV cache, batching — has a concrete, inspectable implementation.

## Contents

- [The Name](#the-name)
- [Why This Project Exists](#why-this-project-exists)
- [What You Can Learn From This Codebase](#what-you-can-learn-from-this-codebase)
- [Architecture](#architecture)
- [Core Interfaces](#core-interfaces)
- [Quick Start](#quick-start)
- [CLI](#cli)
- [Supported Models and Quantization](#supported-models-and-quantization)
- [Performance Expectations](#performance-expectations)
- [Roadmap](#roadmap)
- [Design Principles](#design-principles)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## The Name

**Orbix** = **orbit** + **-ix**.

An orbit is a stable, predictable path around a center. In Orbix, inference requests orbit a scheduler and a KV-cache manager: they are batched, cached, and streamed in a controlled loop. The `-ix` suffix nods to *inference*.

| Layer               | Name                          |
| ------------------- | ----------------------------- |
| Brand               | **Orbix**                     |
| CLI / server binary | **`orbix`**                   |
| Go package          | **`orbix`**                   |
| Environment prefix  | **`ORBIX_`**                  |
| Config              | **`orbix.yaml`**, `~/.orbix/` |

## Why This Project Exists

Most LLM inference code lives in Python + CUDA, wrapped in C++ kernels, and hidden behind framework abstractions. That is the right choice for production. It is a hard place to *learn*.

Orbix exists to answer a simple question: **can I understand LLM inference by building it in Go, from the file format up?** Go is a good language for this. It has explicit control flow, readable concurrency, a strong standard library, and no hidden magic. A GGUF parser, a quantized matmul, and a KV cache are all just code — and Go makes that code visible.

The constraints follow from the goal:

- **Pure Go, no CGo.** If a C library does it, I have to write it myself to understand it. `CGO_ENABLED=0` is not a deployment trick here; it is a learning constraint.
- **Readability over performance.** Kernels are written to be read, not to be the fastest. Assembly fast paths come later, behind the same interface, after the reference implementation is correct and understandable.
- **One layer at a time.** The project is organized so that each layer — GGUF, kernels, KV cache, scheduler, server — can be studied independently.
- **Honest benchmarks.** No cherry-picked numbers. If Orbix is slower than llama.cpp, the README says so.

**What this costs, stated plainly:** pure-Go CPU inference is slower than C++ with hand-tuned SIMD. Orbix does not pretend otherwise, and it is not trying to win a throughput benchmark. It is trying to make inference legible.

## Implementation Status

The repository is at the start of the roadmap. What exists, and what has been
verified:

| Layer          | Package             | State                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GGUF container | `internal/gguf`     | Parses header, metadata, tensor directory, and streams tensor data. Every field and byte offset cross-checked against an independent parser.                                                                                                                                                                                                                                                                                                 |
| Half precision | `internal/f16`      | binary16 to/from float32 with correct subnormals, infinities, NaNs, and round-to-nearest-even. All 65,536 bit patterns verified against a reference.                                                                                                                                                                                                                                                                                         |
| Dequantization | `internal/quant`    | F32, F16, BF16, Q4_0, Q8_0. Decoded values for every tensor in a real model compared bit for bit with a reference.                                                                                                                                                                                                                                                                                                                           |
| Kernels        | `internal/kernel`   | MatMul, MatMulTransposed, MatVec, RMSNorm, Softmax, SiLU, and elementwise helpers. A `Runner` splits the row-partitioned products across goroutines with a measured size threshold; every row is still computed by one goroutine in serial order, so results are bit-identical to the serial kernel.                                                                                                                                         |
| Tokenizer      | `internal/tokenize` | SentencePiece segmentation matching llama.cpp's implementation token for token.                                                                                                                                                                                                                                                                                                                                                              |
| Engine         | `internal/engine`   | Interleaves several sequences over one model, one token per sequence per round. Each sequence keeps its own logits copy, and each prompt is prefilled in one batched pass. Decode is still one token at a time, so concurrency lowers per-stream latency cost without yet raising total throughput.                                                                                                                                          |
| Transformer    | `internal/llama`    | Llama forward pass with adjacent-pair RoPE, grouped-query attention, and a contiguous KV cache. Batched prefill (`Prefill`) runs a prompt as one matrix-matrix sequence and returns the last position's logits; its output is bit-identical to the per-token path. Greedy and temperature sampling. Greedy decoding on a real checkpoint reproduces llama.cpp's text exactly, and per-token log-probabilities agree to within about 0.1 nat. |

## What You Can Learn From This Codebase

Each layer maps to a concept in LLM inference. The code is organized to make that mapping explicit.

| Layer               | Concept                                       | What the code shows                                                                      |
| ------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **GGUF loader**     | Model formats, quantization types, tokenizers | How a model file is structured, how tensors are laid out, how tokens become IDs          |
| **Compute kernels** | Matmul, attention, normalization, activations | What a transformer forward pass actually computes, and why quantization changes the math |
| **KVCacheManager**  | KV cache, paged allocation, prefix reuse      | Why inference is memory-bound, how attention state is stored and reused                  |
| **Scheduler**       | Continuous batching, prefill/decode, priority | Why serving is a scheduling problem, not just a compute problem                          |
| **Server**          | OpenAI-compatible API, SSE, request lifecycle | How a serving layer maps HTTP requests onto an inference engine                          |

If you are learning LLM inference, the intended reading order is top to bottom: start with GGUF, then kernels, then cache, then scheduler, then server. Each layer has a `doc.go` explaining the concept and the design choices.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                       orbix serve (HTTP)                        │
│             OpenAI-compatible /v1/chat/completions              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                            Scheduler                            │
│             continuous batching · priority · queues             │
└──────────────┬──────────────────────────────────┬───────────────┘
               │                                  │
┌─────────────────────────────┐   ┌───────────────────────────────┐
│       KVCacheManager        │   │            Backend            │
│        paged blocks         │   │      native Go (default)      │
│        radix prefix         │   │          remote HTTP          │
└─────────────────────────────┘   └───────────────┬───────────────┘
                                                  │
                                  ┌───────────────▼───────────────┐
                                  │        Compute kernels        │
                                  │       quantized matmul        │
                                  │       attention · norms       │
                                  └───────────────────────────────┘
```

- **Scheduler** owns admission, batching, and the prefill/decode lifecycle.
- **KVCacheManager** abstracts block allocation and prefix reuse. The default implementation uses paged blocks.
- **Backend** is the compute boundary. The default `native` backend runs kernels in-process.
- **Compute kernels** are where the pure-Go work happens, with reference implementations first and assembly fast paths later.

Dependencies point downward only. Kernels never import the scheduler; the scheduler never imports the server.

## Core Interfaces

```go
// Engine is the top-level entry point.
type Engine interface {
	Load(ctx context.Context, modelPath string, opts LoadOptions) error
	Scheduler() Scheduler
	Close() error
}

// Scheduler owns admission and batching. It returns a stream of results.
type Scheduler interface {
	Submit(ctx context.Context, req Request) (<-chan Event, error)
	Stats() SchedulerStats
}

// KVCacheManager abstracts cache lifecycle and prefix reuse.
type KVCacheManager interface {
	Allocate(seqLen int) (BlockHandle, error)
	Release(handle BlockHandle)
	MatchPrefix(tokens []int) (cachedLen int, handle BlockHandle)
}

// Backend is the compute boundary.
type Backend interface {
	Name() string
	Capabilities() Capabilities
	Execute(ctx context.Context, batch Batch) (<-chan Event, error)
}

// Event is the streaming unit.
type Event struct {
	Type  EventType
	Token string
	Meta  map[string]any
	Err   error
}
```

The compute kernels are internal but exposed for testing and benchmarking:

```go
package kernel

func MatMulQ4_0(dst, a []float32, b []byte, m, n, k int)
func Attention(q, k, v []float32, mask []float32, scale float32)
func RMSNorm(dst, src, weight []float32, eps float32)

// Runner evaluates the row-partitioned kernels across a bounded worker set.
func NewRunner(workers int) *Runner
func (r *Runner) MatVec(dst, a, v []float32, n, k int) error
func (r *Runner) MatMulTransposed(dst, a, b []float32, m, n, k int) error
```

## Quick Start

`orbix run` and `orbix model inspect` work today. The serving and embedding
snippets are the target API, kept here so the shape of the engine stays visible
while the upper layers are built.

### Serve a GGUF model

```bash
go install github.com/pplmx/orbix/cmd/orbix@latest

orbix serve \
  --model ./models/qwen2.5-7b-instruct-q4_0.gguf \
  --addr :8080
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Explain continuous batching."}],
    "stream": true
  }'
```

### Embed in a Go service

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/pplmx/orbix"
	"github.com/pplmx/orbix/kv/paged"
)

func main() {
	eng := orbix.NewEngine(
		orbix.WithKVCache(paged.New(paged.Config{BlockSize: 16})),
		orbix.WithMaxBatchSize(32),
	)
	if err := eng.Load(context.Background(), "./model.gguf", orbix.LoadOptions{}); err != nil {
		log.Fatal(err)
	}
	defer eng.Close()

	events, err := eng.Scheduler().Submit(context.Background(), orbix.Request{
		Prompt:    "What does a KV cache store?",
		MaxTokens: 256,
	})
	if err != nil {
		log.Fatal(err)
	}

	for ev := range events {
		switch ev.Type {
		case orbix.EventToken:
			fmt.Print(ev.Token)
		case orbix.EventDone:
			fmt.Println()
		}
	}
}
```

### Cross-compile

```bash
GOOS=linux GOARCH=arm64 go build -o orbix-arm64 ./cmd/orbix
GOOS=windows GOARCH=amd64 go build -o orbix.exe ./cmd/orbix
GOOS=linux GOARCH=riscv64 go build -o orbix-riscv64 ./cmd/orbix
```

No toolchain setup beyond Go itself.

## CLI

Only the commands marked as implemented exist today; the rest are tracked in
the [Roadmap](#roadmap) and will not appear until they work.

| Command                      | Description                                     | Status   |
| ---------------------------- | ----------------------------------------------- | -------- |
| `orbix model inspect <path>` | Print GGUF metadata, tensor types, quantization | v0.1     |
| `orbix run "<prompt>"`       | One-shot inference                              | v0.1     |
| `orbix version`              | Print version and build info                    | v0.1     |
| `orbix chat`                 | Interactive REPL with history                   | planned  |
| `orbix serve`                | Start the OpenAI-compatible inference server    | planned  |
| `orbix bench`                | Run a built-in throughput/latency benchmark     | v0.2     |

```bash
orbix run --model ./model.gguf "Once upon a time"
orbix run --model ./model.gguf --stats --max-tokens 64 "Explain KV caching."

orbix bench --model ./model.gguf --concurrency 4 --max-tokens 32

orbix model inspect ./model.gguf
orbix model inspect --json ./model.gguf   # machine-readable
```

`orbix model inspect` is also a learning tool: it prints the GGUF layout in a
readable form, which is useful when studying how a model file is structured.

## Supported Models and Quantization

Orbix targets the GGUF ecosystem. Initial support:

| Family                  | Architectures        | Status       |
| ----------------------- | -------------------- | ------------ |
| **Llama / Llama 2 / 3** | `llama`              | planned v0.1 |
| **Qwen 2 / 2.5**        | `qwen2`              | planned v0.2 |
| **Mistral / Mixtral**   | `mistral`, `mixtral` | planned v0.3 |
| **Gemma**               | `gemma`              | planned v0.4 |

| Quantization                   | Status |
| ------------------------------ | ------ |
| `F32`, `F16`                   | v0.1   |
| `Q8_0`                         | v0.1   |
| `Q4_0`, `Q4_1`                 | v0.2   |
| `Q5_0`, `Q5_1`                 | v0.3   |
| `K-quants` (`Q4_K`, `Q6_K`)    | v0.4   |

GGUF parsing and tokenization are implemented in pure Go from the start; the quantization table expands incrementally.

## Performance Expectations

Orbix is a learning project, and it is honest about where it stands:

- **CPU-only.** No GPU support. For GPU throughput, use vLLM, SGLang, or TensorRT-LLM.
- **Reference kernels are not optimized per instruction.** They are written to be readable, with no SIMD and no assembly. What they do use is more than one core: the row-partitioned products run on a bounded worker set, which took decoding on stories15M from 65 to 235 tokens per second on a 128-core EPYC. Assembly fast paths on `amd64` (AVX2) and `arm64` (NEON) are planned behind the same interfaces, after the reference implementations are correct.
- **Parallelism is over rows, never over a reduction.** Every output row is computed by one goroutine in the same order the serial kernel uses, so float32 addition is never reassociated and the parallel result is bit-identical. That is a deliberate constraint: a split reduction would be faster on paper and would silently change every number the project compares against llama.cpp.
- **Scheduling decides what concurrency buys.** Each sequence's prompt is prefilled in one batched pass, but decode still advances one token per sequence per round on one shared model. Concurrency therefore splits one stream's speed into several fair streams; a real throughput win needs batched decode, which is the next layer.
- **Benchmarks are self-run.** `orbix bench` is part of the CLI so you can measure on your own hardware, with your own model. It reports the worker count it actually used.

The intended comparison is not "Orbix vs vLLM on an A100". It is "what does it take to make LLM inference work in pure Go, and what do I learn by building it?"

## Roadmap

The roadmap is a learning path as much as a feature list. Each version adds one conceptual layer.

- [x] `v0.1` — **GGUF and first kernels**: GGUF loader, F32/F16/BF16/Q4_0/Q8_0 decode, reference matmul/RMSNorm/softmax/SiLU kernels, SentencePiece tokenizer, Llama forward pass, `orbix run`, `orbix model inspect`
- [ ] `v0.2` — **Scheduling and cache**: scheduler with continuous batching, paged `KVCacheManager`, Q4_0/Q4_1, Qwen2 — *partial: the engine interleaves sequences, prefills prompts in one batched pass, and `orbix bench` reports throughput; batched decode, paging, and Qwen2 are not done*
- [ ] `v0.3` — **Serving**: `orbix serve` with OpenAI-compatible SSE, priority queues, Q5_0/Q5_1, Mistral
- [ ] `v0.4` — **Performance paths**: K-quants, prefix-aware radix cache, `amd64` AVX2 and `arm64` NEON assembly
- [ ] `v0.5` — **Measurement and observability**: `orbix bench` suite, remote HTTP backend, Prometheus and OpenTelemetry
- [ ] `v1.0` — **Stable API**: cross-platform CI matrix, published benchmarks, documented reading path

## Design Principles

1. **No CGo, ever.** `CGO_ENABLED=0` is the default and the only supported configuration. If a feature needs C to be understood, it is out of scope.
2. **Readability before performance.** Reference implementations are written to be read. Fast paths come later, behind the same interfaces, and never replace the readable version.
3. **One layer, one concept.** Each package maps to a concept in LLM inference, with a `doc.go` explaining what it does and why.
4. **Scheduling and caching are first-class.** The kernel is not the whole engine. Continuous batching and KV reuse are what separate a serving engine from a script.
5. **GGUF is the model format.** No proprietary formats, no Python conversion step.
6. **Honest benchmarks.** No cherry-picked numbers. `orbix bench` runs on the user's hardware, with the user's model.
7. **Go conventions are load-bearing.** Explicit interfaces, directional dependencies, `context.Context` everywhere, no reflection-heavy magic.

## Configuration

Orbix reads `orbix.yaml` from the working directory or `~/.orbix/`. Environment variables use the `ORBIX_` prefix.

```yaml
model: ./models/qwen2.5-7b-q4_0.gguf
scheduler:
  max_batch_size: 32
  policy: priority
kv_cache:
  manager: paged
  block_size: 16
  max_blocks: 2048
server:
  addr: ":8080"
```

```bash
export ORBIX_MODEL=./models/llama-3-8b-q4_0.gguf
export ORBIX_SERVER_ADDR=:8080
```

## Development

```bash
git clone https://github.com/pplmx/orbix
cd orbix

make test      # CGO_ENABLED=0 go test ./...
make lint      # golangci-lint run
make build     # CGO_ENABLED=0 go build -o bin/orbix ./cmd/orbix
make bench     # run kernel and engine benchmarks
```

CI enforces `CGO_ENABLED=0` on every platform. A build that requires CGo is a build that fails.

If you are contributing with an AI coding agent, read [`AGENTS.md`](AGENTS.md) first. It records the no-CGo constraint, layering rules, interface contracts, and naming conventions.

## Contributing

This is primarily a personal learning project, but issues and design discussions are welcome — especially if you are learning LLM inference yourself and spot something unclear.

- **No CGo.** PRs that introduce `import "C"` or require `CGO_ENABLED=1` will not be merged.
- **Clarity is a feature.** If a kernel or scheduler change makes the code harder to read, it needs a strong justification.
- Keep dependencies directional: kernels must not import the scheduler; the scheduler must not import the server.
- Kernel changes must include a correctness test against the reference implementation and a benchmark.
- New model architectures go through the architecture registry; do not add model-specific branches to the scheduler or server.
- Naming: brand **Orbix**, binary **`orbix`**, package **`orbix`**.

## License

MIT © 2026 pplmx
