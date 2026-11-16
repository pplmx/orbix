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

| Layer          | Package             | State                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GGUF container | `internal/gguf`     | Parses header, metadata, tensor directory, and streams tensor data. Every field and byte offset cross-checked against an independent parser.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Half precision | `internal/f16`      | binary16 to/from float32 with correct subnormals, infinities, NaNs, and round-to-nearest-even. All 65,536 bit patterns verified against a reference.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Dequantization | `internal/quant`    | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. The 32-weight formats are pinned by exact-value tests on constructed blocks; the 256-weight K-quants add a round-trip error bound and a check against a real checkpoint tensor, because a wrong sub-block scale still decodes to plausible magnitudes.                                                                                                                                                                                                                                                                                                                                           |
| Kernels        | `internal/kernel`   | MatMul, MatMulTransposed, MatVec, RMSNorm, Softmax, SiLU, and elementwise helpers. A `Runner` splits the row-partitioned products across goroutines with a measured size threshold; every row is still computed by one goroutine in serial order, so results are bit-identical to the serial kernel.                                                                                                                                                                                                                                                                                                                                                                                 |
| Tokenizer      | `internal/tokenize` | SentencePiece segmentation matching llama.cpp's implementation token for token.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Engine         | `internal/engine`   | Batches a scheduling round: every pending prompt prefills in one pass, then every active sequence decodes together in one pass, so concurrency raises total throughput. `MaxActive` bounds a round and an admission policy (`fifo`, `shortest`, `fair`) picks which sequences fill it. Each sequence keeps its own logits copy and KV cache; the cache is a contiguous arena or paged blocks from a shared pool, with optional prefix reuse. `RenderChatPrompt` is the one chat template the REPL and the HTTP server share.                                                                                                                                                         |
| Transformer    | `internal/llama`    | Llama and Qwen2 forward passes: adjacent-pair (Llama) and NEOX (Qwen2) RoPE, optional attention bias, grouped-query attention, and two KV cache layouts -- a contiguous arena per sequence and a reference-counted paged cache over a shared block pool. `PrefillWithPrefix` reuses blocks whose token prefix has been seen before. `Prefill` and `DecodeStep` are bit-identical to the per-token path on either layout. Greedy decoding on a real checkpoint reproduces llama.cpp's text exactly.                                                                                                                                                                                   |
| Server         | `internal/server`   | `orbix serve`: OpenAI-compatible `/v1/chat/completions` and `/v1/completions` with SSE streaming, `/v1/models`, `/healthz`, and Prometheus-format `/metrics`. Streaming text is decoded incrementally so multi-byte characters split across tokens still render; concatenated stream deltas equal the non-streaming body exactly. A disconnected client cancels its sequence through the engine. Both endpoints accept an Orbix extension, `priority`, which ranks a request against others waiting for a slot: higher advances first and the admission policy orders requests of equal priority. It is omitted from OpenAI's schema, so a client that never sends it is unaffected. |

## What You Can Learn From This Codebase

Each layer maps to a concept in LLM inference. The code is organized to make that mapping explicit.

| Layer               | Concept                                       | What the code shows                                                                      |
| ------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **GGUF loader**     | Model formats, quantization types, tokenizers | How a model file is structured, how tensors are laid out, how tokens become IDs          |
| **Compute kernels** | Matmul, attention, normalization, activations | What a transformer forward pass actually computes, and why quantization changes the math |
| **KV cache**        | KV cache, paged allocation, prefix reuse      | Why inference is memory-bound, how attention state is stored and reused                  |
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
│                continuous batching · admission policy           │
└──────────────┬───────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────┐   ┌───────────────────────────┐
│           KV cache              │   │     Compute kernels       │
│   BlockPool: refcounted blocks  │   │     quantized matmul      │
│   PagedKVCache: per-sequence    │   │     attention · norms     │
│   PrefixIndex: hashed prefixes  │   │     silu · softmax        │
└─────────────────────────────────┘   └───────────────────────────┘
```

Solid boxes above are what the code does today. Two components that a project like
this usually has are deliberately *not* drawn, because drawing them would imply they
exist:

- There is no `Backend` abstraction. One compute path exists -- the in-process
  kernels -- and introducing an interface with a single implementation would be
  ceremony rather than a seam. A remote backend is the change that would earn it.
- There is no `KVCacheManager` interface, and no radix tree. The paged cache and its
  hashed prefix index cover the same job: a shared pool of reference-counted blocks,
  and an index that hands back the blocks a repeated prompt prefix already produced.
  The index matches whole blocks by hash and compares the tokens, so a collision
  costs a rejected hit rather than a wrong result. A radix tree would let a prefix be
  matched at sub-block granularity and would make evicting a subtree natural, which
  is a real difference -- but it would not change what the cache is for, so it is not
  claimed here.
- **Scheduler** owns admission, batching, and the prefill/decode lifecycle.
- **Compute kernels** are where the pure-Go work happens, with reference
  implementations first and assembly fast paths later.

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

// Event is the streaming unit.
type Event struct {
	Token        int32
	HasToken     bool
	Done         bool
	Err          error
	FinishReason string
}
```

Progress against those interfaces, stated plainly so the block above reads as a target
rather than as a description:

| Interface        | State                                                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| `Engine`         | Exists as `engine.Engine`, but takes a model path rather than a `LoadOptions` struct                                  |
| `Scheduler`      | Exists as `Engine.Submit` + `Engine.Pump`; there is no `Stats()` yet, and a round is driven by the caller, not a loop |
| `KVCacheManager` | Not a separate interface: `llama.BlockPool` owns allocation and `llama.PrefixIndex` owns prefix matching              |
| `Backend`        | Not implemented. There is one compute path (the in-process kernels); a remote backend would be the place to add it    |

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

`orbix run`, `orbix model inspect`, `orbix bench`, and `orbix serve` work today.
The embedding snippet below is the target API, kept here so the shape of the
engine stays visible while that layer is built.

### Serve a GGUF model

```bash
go install github.com/pplmx/orbix/cmd/orbix@latest

orbix serve \
  --model ./models/qwen2.5-7b-instruct-q4_0.gguf \
  --addr :8080
```

The server implements `/v1/chat/completions`, `/v1/completions` (both streaming
and not), `/v1/models`, `/healthz`, and `/metrics`. Two things about the chat
endpoint are worth knowing before pointing a client at it:

- **The chat template is neutral.** GGUF stores a model's real template as
  Jinja2, and this project takes no Jinja dependency. The server renders
  `Role: content` lines and leaves the assistant turn open. If your model needs
  its own header or marker tokens, render the prompt yourself and send it to
  `/v1/completions`, which passes text through verbatim.
- **`stop` matches on token ids.** A stop string is encoded and its first token
  is used, which covers `"\n"` and marker strings. A stop that spans several
  tokens is not matched.

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

Every command below marked with a version works today. `orbix chat` runs on the
engine with a paged cache and prefix reuse, so each turn re-attaches the previous
turn's KV blocks instead of prefilling the whole conversation again.

| Command                      | Description                                     | Status   |
| ---------------------------- | ----------------------------------------------- | -------- |
| `orbix model inspect <path>` | Print GGUF metadata, tensor types, quantization | v0.1     |
| `orbix run "<prompt>"`       | One-shot inference                              | v0.1     |
| `orbix version`              | Print version and build info                    | v0.1     |
| `orbix chat`                 | Interactive REPL with conversation history      | v0.3     |
| `orbix serve`                | Start the OpenAI-compatible inference server    | v0.3     |
| `orbix bench`                | Run a built-in throughput/latency benchmark     | v0.2     |

```bash
orbix run --model ./model.gguf "Once upon a time"
orbix run --model ./model.gguf --stats --max-tokens 64 "Explain KV caching."

orbix chat --model ./model.gguf

orbix bench --model ./model.gguf --concurrency 4 --max-tokens 32
orbix bench --model ./model.gguf --suite     # every scenario in one table

orbix model inspect ./model.gguf
orbix model inspect --json ./model.gguf   # machine-readable
```

`orbix model inspect` is also a learning tool: it prints the GGUF layout in a
readable form, which is useful when studying how a model file is structured.

`orbix bench --suite` runs a fixed set of scenarios and prints them as one table. A
single throughput number answers one question about one machine; the suite is what shows
which mechanisms pay for themselves. It pairs each mechanism with a control row -- paged
against contiguous caches, prefix reuse against none, shortest against FIFO -- so a
difference is attributable to that mechanism rather than to the scenario changing.

Measured on stories15M q4_0 with a 41-token prompt, 8 workers:

```text
SCENARIO         TOKENS  ELAPSED  TOK/S  FIRST  NOTE
single stream    8       77ms     104.3  40ms   baseline: one request, contiguous cache
batch x4         32      261ms    122.8  159ms  batched decode, contiguous caches
batch x16        128     993ms    128.9  644ms  where batched decode stops scaling
paged x16        128     983ms    130.2  643ms  same batch, block cache instead of whole-context reservations
prefix no reuse  32      261ms    122.8  159ms  four requests sharing one prompt, reuse off
prefix reuse x4  32      192ms    166.3  95ms   the same four, reuse on
policy shortest  63      396ms    159.0  41ms   MaxActive 1, a short request queued behind a long one
policy fifo      63      395ms    159.5  381ms  the same queue without reordering
```

Two of those rows need a sentence of explanation, because their numbers look flat for a
reason:

- **Paged vs contiguous** is a memory trade, not a throughput one. The paged cache serves
  the same batch at the same speed while letting 16 sequences share one pool instead of
  each reserving a whole context. The suite prints the pool shape so the trade is visible
  rather than implied.
- **The policy rows have the same total time on purpose.** A policy reorders a fixed
  amount of work; it cannot reduce it. What it changes is the short request's first-token
  latency, 381 ms under FIFO against 41 ms under shortest, and that is the `FIRST` column.

`orbix serve` also exports the same numbers over `/metrics` in Prometheus text format,
so the suite's baseline and a live deployment's behaviour are the same measurements
rather than two unrelated readings:

```text
# TYPE orbix_tokens_generated_total counter
orbix_tokens_generated_total 6
# TYPE orbix_requests_queued gauge
orbix_requests_queued 0
# TYPE orbix_kv_pool_used_fraction gauge
orbix_kv_pool_used_fraction 0
```

The exposition format is written by hand rather than through a client library, for the
same reason the GGUF parser is: it is four lines of syntax, and the hand-written encoder
is read by the code review rather than trusted as a dependency. The tests parse the
output back the way a scraper would, so a HELP line in the wrong place fails a test
instead of a dashboard.

Two of the metrics exist because their failure mode is invisible in latency until it is
already failing. `orbix_requests_queued` growing against a steady round rate is a
capacity problem rather than a slow model, and `orbix_kv_pool_free` reaching zero is a
paged deployment about to start refusing requests. Neither is recoverable from timing
alone.

## Supported Models and Quantization

Orbix targets the GGUF ecosystem. Initial support:

| Family                  | Architectures        | Status       |
| ----------------------- | -------------------- | ------------ |
| **Llama / Llama 2 / 3** | `llama`              | v0.1         |
| **Qwen 2 / 2.5**        | `qwen2`              | v0.2         |
| **Mistral**             | `mistral`            | v0.3         |
| **Mixtral**             | `mixtral`            | planned v0.4 |
| **Gemma**               | `gemma`              | planned v0.4 |

| Quantization                   | Status |
| ------------------------------ | ------ |
| `F32`, `F16`                   | v0.1   |
| `Q8_0`                         | v0.1   |
| `Q4_0`, `Q4_1`                 | v0.2   |
| `Q5_0`, `Q5_1`                 | v0.3   |
| `K-quants` (`Q2_K`-`Q6_K`)     | done   |

GGUF parsing and tokenization are implemented in pure Go from the start; the quantization table expands incrementally.

Both supported architectures are checked against real checkpoints, not only against
synthetic fixtures. So is the quantization: the official Qwen2.5-0.5B checkpoints in
Q4_0, Q4_K_M, and Q6_K form all produce the same continuation, and Q4_K_M exercises
four decoders at once. `scripts/hf_to_gguf.py` converts a Qwen2.5 safetensors checkpoint
to GGUF for that purpose; it is deliberately independent of the Go code, because a
converter that shared the loader's assumptions could not catch a wrong one. With
`QWEN2_MODEL` set it runs those tests:

```bash
uv run --with numpy python3 scripts/hf_to_gguf.py ./Qwen2.5-0.5B-Instruct /tmp/qwen2-f32.gguf
QWEN2_MODEL=/tmp/qwen2-f32.gguf go test ./internal/llama/ -run Qwen2
```

## Performance Expectations

Orbix is a learning project, and it is honest about where it stands:

- **CPU-only.** No GPU support. For GPU throughput, use vLLM, SGLang, or TensorRT-LLM.
- **Reference kernels are not optimized per instruction.** They are written to be readable, with no SIMD and no assembly. What they do use is more than one core: the row-partitioned products run on a bounded worker set, which took decoding on stories15M from 65 to 235 tokens per second on a 128-core EPYC. Assembly fast paths on `amd64` (AVX2) and `arm64` (NEON) are planned behind the same interfaces, after the reference implementations are correct.
- **Parallelism is over rows, never over a reduction.** Every output row is computed by one goroutine in the same order the serial kernel uses, so float32 addition is never reassociated and the parallel result is bit-identical. That is a deliberate constraint: a split reduction would be faster on paper and would silently change every number the project compares against llama.cpp.
- **Scheduling turns concurrency into throughput.** A round prefills every pending prompt and decodes every active sequence in one pass, so total throughput scales with concurrency: on stories15M-q4_0 it goes 237 tok/s at one stream to 571 tok/s at sixteen. Per-stream speed still drops as streams are added, because the model is one shared resource.
- **A shared prompt costs almost nothing the second time.** With `--prefix-cache`, a request whose prompt starts with blocks the pool has already computed attaches them instead of prefilling them again. On a 96-token shared prefix, one prefill drops from 56.7 ms to 4.1 ms (14x), and four requests with the same prompt finish in 154 ms instead of 246 ms. This is the case a chat application actually has: a system prompt and earlier turns are re-sent verbatim on every call.
- **Benchmarks are self-run.** `orbix bench` is part of the CLI so you can measure on your own hardware, with your own model. It reports the worker count it actually used.

The intended comparison is not "Orbix vs vLLM on an A100". It is "what does it take to make LLM inference work in pure Go, and what do I learn by building it?"

## Roadmap

The roadmap is a learning path as much as a feature list. Each version adds one conceptual layer.

- [x] `v0.1` — **GGUF and first kernels**: GGUF loader, F32/F16/BF16/Q4_0/Q8_0 decode, reference matmul/RMSNorm/softmax/SiLU kernels, SentencePiece tokenizer, Llama forward pass, `orbix run`, `orbix model inspect`
- [x] `v0.2` — **Scheduling and cache**: continuous batching, a paged KV cache, Q4_0/Q4_1, Qwen2 — batched prefill and decode with an admission policy, a reference-counted paged cache with prefix reuse, every 32-weight ggml block format, Llama and Qwen2
- [ ] `v0.3` — **Serving**: `orbix serve` with OpenAI-compatible SSE, priority queues, Q5_0/Q5_1, Mistral — *partial: `orbix serve` implements chat and text completions with SSE streaming, client-disconnect cancellation, and a per-request `priority` field; an admission policy picks which queued requests advance, Q5_0/Q5_1 decode and the `mistral` architecture with its sliding window work. A model's own Jinja template and Mixtral's mixture of experts are not done*
- [ ] `v0.4` — **Performance paths**: K-quants, prefix-aware radix cache, `amd64` AVX2 and `arm64` NEON assembly — *partial: every K-quant and IQ4_NL decode, and the paged cache with a prefix index covers the reuse case; a radix-based prefix tree and assembly are not done*
- [ ] `v0.5` — **Measurement and observability**: `orbix bench` suite, remote HTTP backend, Prometheus and OpenTelemetry — *partial: `orbix bench --suite` prints every scenario as one table, and `/metrics` exports counters and gauges in Prometheus text format; a remote HTTP backend and OpenTelemetry traces are not done*
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

Every setting is a flag today; there is no config file. `ORBIX_MODEL` is the one
environment variable, so a shell can point every command at the same checkpoint.

```bash
export ORBIX_MODEL=./models/llama-3-8b-q4_0.gguf
orbix serve --max-active 32 --policy shortest --blocks 2048 --block-size 16 --prefix-cache
```

| Flag                       | What it controls                                                   |
| -------------------------- | ------------------------------------------------------------------ |
| `--model`                  | Checkpoint path; falls back to `ORBIX_MODEL`                       |
| `--max-active`             | Sequences advanced per round, and therefore the batch size         |
| `--policy`                 | Which sequences fill a round: `fifo`, `shortest`, or `fair`        |
| `--workers`                | Kernel goroutines; 0 means one per CPU                             |
| `--blocks`, `--block-size` | Paged KV cache; without them each request reserves a whole context |
| `--prefix-cache`           | Reuse KV blocks across requests sharing a prompt prefix            |

A config file is on the roadmap, not in the code: the interface block at the top of
this file describes the target, and this table describes what exists.

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
