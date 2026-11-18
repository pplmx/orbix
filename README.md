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

| Layer          | Package             | State                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| -------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                                                                                                                                                                                                                               |
| GGUF container | `internal/gguf`     | Parses header, metadata, tensor directory, and streams tensor data. Every field and byte offset cross-checked against an independent parser.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Half precision | `internal/f16`      | binary16 to/from float32 with correct subnormals, infinities, NaNs, and round-to-nearest-even. All 65,536 bit patterns verified against a reference.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Dequantization | `internal/quant`    | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. The 32-weight formats are pinned by exact-value tests on constructed blocks; the 256-weight K-quants add a round-trip error bound and a check against a real checkpoint tensor, because a wrong sub-block scale still decodes to plausible magnitudes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Kernels        | `internal/kernel`   | MatMul, MatMulTransposed, MatVec, RMSNorm, Softmax, SiLU, and elementwise helpers. A `Runner` splits the row-partitioned products across goroutines with a measured size threshold; every row is still computed by one goroutine in serial order, so results are bit-identical to the serial kernel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Tokenizer      | `internal/tokenize` | SentencePiece and byte-level BPE segmentation, with the four merge, space, and pre-tokenizer conventions read from the file rather than assumed. Thirty-four pre-tokenizer families are implemented (`qwen2`, `llama-bpe`, `gpt-2`, `starcoder`, `deepseek-v3`, `tekken`, `gpt4o`, `whitespace`, `bloom`, `deepseek-coder`, `falcon`, `deepseek-llm`, `mmbert`, `viking`, `superbpe`, `soprano`), covering all 100 names llama.cpp maps to a family (`dbrx`, `smaug-bpe`, `chatglm-bpe`, `glm4` and `glm5` share `llama-bpe`'s expression verbatim, and `gemma4`, `granite-embed-multi-311m`, `domyn-small` and `sarvam-moe` share `mmbert`'s, and `grok-2`, `bailingmoe`, `bailingmoe2` and `llada-moe` share the new one, and `cohere2moe` and `tiny_aya` share another); a file naming another one tokenizes as a single chunk, and `orbix model inspect` says so rather than leaving it to be discovered as poor output. There are no unimplemented names left -- checked against llama.cpp's pre-type assignments, every name resolves to a rule that agrees with the reference on the chunk boundaries. Each family's chunk boundaries are pinned by tests built from its llama.cpp expression, and the rules were diffed against those expressions as sequential passes over 80,128 strings -- every disagreement found that way is fixed, and the 126 strings that exposed them are kept in the package as a regression corpus. `MODEL`-gated tests round-trip every Unicode code point and a set of convention-heavy strings through a real checkpoint.                                          |
| Chat templates | `internal/chat`     | Evaluates the subset of Jinja2 a checkpoint's `tokenizer.chat_template` needs -- conditionals, loops, set, concatenation, and the trim/tojson filters -- so a conversation is rendered in the format the model was trained on. Anything outside the subset is reported at parse time, and the caller falls back to a neutral transcript rather than emitting a half-rendered prompt.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Engine         | `internal/engine`   | Batches a scheduling round: every pending prompt prefills in one pass, then every active sequence decodes together in one pass, so concurrency raises total throughput. `MaxActive` bounds a round and an admission policy (`fifo`, `shortest`, `fair`) picks which sequences fill it. Each sequence keeps its own logits copy and KV cache; the cache is a contiguous arena or paged blocks from a shared pool, with optional prefix reuse. `RenderChatPrompt` is the one chat template the REPL and the HTTP server share.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Transformer    | `internal/llama`    | Llama and Qwen2 forward passes: adjacent-pair (Llama) and NEOX (Qwen2) RoPE, optional attention bias, grouped-query attention, and two KV cache layouts -- a contiguous arena per sequence and a reference-counted paged cache over a shared block pool. `PrefillWithPrefix` reuses blocks whose token prefix has been seen before. `Prefill` and `DecodeStep` are bit-identical to the per-token path on either layout. Greedy decoding on a real checkpoint agrees with llama.cpp's log-probabilities to within 0.12 nats, which is close enough that the continuation matches on the prompts tried -- and not a guarantee that it matches on every one, since a greedy choice is only as stable as the gap between the top two candidates.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Tracing        | `internal/otel`     | Exports spans in OpenTelemetry's OTLP/HTTP+JSON format without the SDK: the message it replaces is a handful of structs, which is cheaper to read here than to take as a dependency. Ids are base64 (which proto3 maps a bytes field to, and which most hand-written exporters get wrong by emitting hex), timestamps are nanosecond strings, and export is best-effort: a collector that is down drops spans and counts them rather than slowing a request.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Server         | `internal/server`   | `orbix serve`: OpenAI-compatible `/v1/chat/completions` and `/v1/completions` with SSE streaming, `/v1/models`, `/healthz`, and Prometheus-format `/metrics`. Streaming text is decoded incrementally so multi-byte characters split across tokens still render; concatenated stream deltas equal the non-streaming body exactly. A disconnected client cancels its sequence through the engine. Both endpoints accept an Orbix extension, `priority`, which ranks a request against others waiting for a slot: higher advances first and the admission policy orders requests of equal priority. It is omitted from OpenAI's schema, so a client that never sends it is unaffected.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

## What You Can Learn From This Codebase

Each layer maps to a concept in LLM inference. The code is organized to make that mapping explicit.

| Layer               | Concept                                       | What the code shows                                                                      |
| ------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **GGUF loader**     | Model formats, quantization types, tokenizers | How a model file is structured, how tensors are laid out, how tokens become IDs          |
| **Compute kernels** | Matmul, attention, normalization, activations | What a transformer forward pass actually computes, and why quantization changes the math |
| **KV cache**        | KV cache, paged allocation, prefix reuse      | Why inference is memory-bound, how attention state is stored and reused                  |
| **Scheduler**       | Continuous batching, prefill/decode, priority | Why serving is a scheduling problem, not just a compute problem                          |
| **Server**          | OpenAI-compatible API, SSE, request lifecycle | How a serving layer maps HTTP requests onto an inference engine                          |

If you are learning LLM inference, read in this order. Each entry links to that layer's
`doc.go`, which explains what the layer is responsible for and why it is built the way it
is; the code below it is written to be read in the same order.

1. **[`internal/gguf`](internal/gguf/doc.go)** — the container: how a model file is laid
   out, and why this package describes tensors by offset instead of loading them.
2. **[`internal/f16`](internal/f16/doc.go)** and
   **[`internal/quant`](internal/quant/doc.go)** — half precision and block quantization:
   how a weight becomes a float32, and why the K-quant layouts are checked against exact
   values rather than a round trip.
3. **[`internal/kernel`](internal/kernel/doc.go)** — the numerics: matmul, RMSNorm,
   softmax, SiLU, and the one rule everything else depends on, that parallelism is over
   output rows so a parallel result equals the serial one exactly.
4. **[`internal/llama`](internal/llama/doc.go)** — attention, RoPE, and the KV cache. The
   cache is in this package rather than its own because it is a data structure the
   architecture owns: the contiguous and paged implementations are here, along with the
   prefix index and the mixture-of-experts feed-forward.
5. **[`internal/tokenize`](internal/tokenize/doc.go)** — text to token ids, including the
   byte-level and SentencePiece conventions GGUF records.
6. **[`internal/engine`](internal/engine/doc.go)** — scheduling: continuous batching,
   admission policy, and why a round is the unit of progress.
7. **[`internal/chat`](internal/chat/doc.go)** — the chat template subset, and what a
   template outside it falls back to.
8. **[`internal/server`](internal/server/doc.go)** — the OpenAI-compatible API, SSE
   streaming, `/metrics`, and the access log.

Two layers are not on that path because nothing in the forward pass goes through them:
**[`internal/otel`](internal/otel/doc.go)** exports traces, and
**[`internal/gguftest`](internal/gguftest/doc.go)** builds the model fixtures the tests run
on.

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

- There is no `Backend` abstraction, and there will not be one. A remote backend was
  considered and rejected: everything the scheduler is worth -- continuous batching,
  prefix reuse, a paged cache -- works by driving one batched forward pass over local
  KV blocks, and a backend reachable over HTTP can only be asked for a prompt and
  answer with tokens. Adding the seam would not extend those mechanisms to a remote
  backend; it would mean the local path carries them and the remote path silently
  does not. Two implementations with different capabilities behind one interface is
  worse than one implementation, because the difference is invisible from the call
  site.
- There is no `KVCacheManager` interface, and no radix tree. The paged cache and its
  hashed prefix index cover the same job: a shared pool of reference-counted blocks,
  and an index that hands back the blocks a repeated prompt prefix already produced.
  The index matches whole blocks by hash and compares the tokens, so a collision
  costs a rejected hit rather than a wrong result. A radix tree would let a prefix be
  matched at sub-block granularity, which buys nothing here because the cache stores
  whole blocks; what it would also give is subtree eviction, and the index does that
  explicitly instead: each entry records the block it continues, so eviction takes
  the end of a chain and leaves the head reachable.
- **Scheduler** owns admission, batching, and the prefill/decode lifecycle.
- **Mixture of experts** is a property of the feed-forward layer, not a separate
  architecture: a model with `expert_count` set replaces its single gated FFN with one
  per expert plus a small router, and runs `expert_used_count` of them per token. The
  routing is top-k over the router logits with a softmax over the selected experts only,
  which renormalising a full softmax would give the same answer for. Both the prefill and
  the decode path group rows by chosen expert so one matmul serves every row that picked
  the same one.
- **Compute kernels** are where the pure-Go work happens, with reference
  implementations written to be read. There are no SIMD fast paths: see the note below.

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

| Interface        | State                                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| `Engine`         | Exists as `engine.Engine`, but takes a model path rather than a `LoadOptions` struct                              |
| `Scheduler`      | Exists as `Engine.Submit` + `Engine.Pump` + `Engine.Stats`; a round is driven by the caller, not a loop           |
| `KVCacheManager` | Not a separate interface: `llama.BlockPool` owns allocation and `llama.PrefixIndex` owns prefix matching          |
| `Backend`        | Not implemented, by decision. One compute path exists, and a remote one could not support what the scheduler does |

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

- **The chat template is the model's own, when it can be evaluated.** GGUF stores a
  checkpoint's format as a Jinja2 string under `tokenizer.chat_template`, and
  `internal/chat` evaluates a subset of it: `if`/`elif`/`else`, `for` with the `loop`
  variables, `set`, `+` concatenation, comparisons and `is defined`, and the `trim` and
  `tojson` filters. That covers the published templates for Qwen2.5 and Llama 3.1
  exactly, and both are tested against their real source.
  A template using anything outside the subset is reported at parse time and the server
  falls back to a neutral `Role: content` transcript, printing which of the two it is
  using at startup. The fallback matters: a half-rendered template would produce a
  prompt that looks plausible and is wrong, while a neutral transcript is merely less
  good than the model's own format.
  If you would rather control the prompt, send it to `/v1/completions`, which passes
  text through verbatim.
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

| Command                      | Description                                                                                                  | Status |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| `orbix model inspect <path>` | Print GGUF metadata, tensor types, quantization, and whether this build reproduces the model's pre-tokenizer | v0.1   |
| `orbix run "<prompt>"`       | One-shot inference                                                                                           | v0.1   |
| `orbix version`              | Print version and build info                                                                                 | v0.1   |
| `orbix chat`                 | Interactive REPL with conversation history                                                                   | v0.3   |
| `orbix serve`                | Start the OpenAI-compatible inference server                                                                 | v0.3   |
| `orbix bench`                | Run a built-in throughput/latency benchmark                                                                  | v0.2   |

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

One measurement, stated with everything needed to judge it: stories15M q4_0 (19 MB), a
41-token prompt, 8 generated tokens per request, 8 kernel workers, on a 128-core AMD EPYC
7742, Go 1.27.3, linux/amd64. `orbix bench --suite` prints these conditions itself, so a
table you produce can be compared against this one instead of taken on faith:

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

A row can also show fewer tokens than its budget. That is not a measurement failure: the
model emitted its end-of-sequence token, and the suite says so in the `NOTE` column
("N request(s) stopped early on EOS") rather than leaving you to work out why two rows
that should match do not.

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

| Family                  | Architectures        | Status        |
| ----------------------- | -------------------- | ------------- |
| **Llama / Llama 2 / 3** | `llama`              | v0.1          |
| **Qwen 2 / 2.5**        | `qwen2`              | v0.2          |
| **Mistral**             | `mistral`            | v0.3          |
| **Mixtral**             | `llama` with experts | v0.3          |
| **Gemma**               | `gemma`              | not supported |

| Quantization                   | Status |
| ------------------------------ | ------ |
| `F32`, `F16`                   | v0.1   |
| `Q8_0`                         | v0.1   |
| `Q4_0`, `Q4_1`                 | v0.2   |
| `Q5_0`, `Q5_1`                 | v0.3   |
| `BF16`                         | done   |
| `IQ4_NL`                       | done   |
| `K-quants` (`Q2_K`-`Q6_K`)     | done   |

GGUF parsing and tokenization are implemented in pure Go from the start; the quantization table expands incrementally.

Two of the three architectures are checked against real checkpoints, not only against
synthetic fixtures: `llama` by comparison with llama.cpp's own log-probabilities, and
`qwen2` by a converted official checkpoint whose continuation is on-topic. The
mixture-of-experts path has its own real file. `mistral` does not: its configuration and
its sliding window are covered against constructed files, and adding a real checkpoint
test would need a Mistral GGUF that this project has not had.

The quantization is checked at the block level rather than by comparing whole
generations across formats: each 32-weight format has exact-value tests on constructed
blocks, the 256-weight K-quants add a round-trip error bound, and a test reads a real
K-quant tensor from a checkpoint when one is available -- set `MODEL_KQUANT` to a
K-quantized file to run it. Comparing a single prompt's
continuation across quantizations would test the model more than the decoder -- a
4-bit rounding difference is expected to change a greedy continuation eventually.

A checkpoint that exercises that path, and more of the engine than the small stories model
does: TinyLlama 1.1B's Q2_K build is 483 MB, has 22 layers, and uses grouped-query attention
(32 heads over 4 KV heads), so it covers GQA and the K-quant decoders together. It also
carries a chat template, which the stories model does not:

```bash
curl -sSL -o /tmp/tinyllama-q2k.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
```

With it set, the Q6_K tensor in that file decodes to a mean of -0.0004 and a standard
deviation of 0.0247 over 65.5M values, which is the shape a weight matrix has; the count is
what would go wrong first if a sub-block layout were misread.

`scripts/hf_to_gguf.py` converts a Qwen2.5 safetensors checkpoint to GGUF for the
weight-level cross-check; it is deliberately independent of the Go code, because a
converter that shared the loader's assumptions could not catch a wrong one. With
`QWEN2_MODEL` set it runs those tests:

```bash
uv run --with numpy python3 scripts/hf_to_gguf.py ./Qwen2.5-0.5B-Instruct /tmp/qwen2-f32.gguf
QWEN2_MODEL=/tmp/qwen2-f32.gguf go test ./internal/llama/ -run Qwen2
```

`scripts/jinja_diff.py` checks the chat-template subset the same way: it renders a corpus of
checkpoint templates and one-construct cases through both this package and real Jinja2, and
reports where they differ. Jinja2 is the reference implementation a template was written
against, so a disagreement is either a bug here or a construct the package documents as
unsupported -- the tool separates the two, and exits non-zero if any template renders
differently rather than being refused.

```bash
uv run --with jinja2 python3 scripts/jinja_diff.py
```

The mixture-of-experts path has the same treatment, on a small public MoE checkpoint.
Set `MIXTRAL_MODEL` to run it:

```bash
MIXTRAL_MODEL=/tmp/tinymixtral.gguf go test ./internal/llama/ -run TestMoEReachesTheExperts
```

The stories model is a 19 MB public download, and it is the one of the four that takes no
conversion step:

```bash
curl -sSL -o /tmp/stories15M-q4_0.gguf \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
```

The gates take the suite from twelve skipped tests to none, so a machine holding the
checkpoints can check every claim the project makes. Both numbers are observable in the run
itself -- `go test ./... -v | grep -c '^--- SKIP'` reads 12 with no checkpoint on disk and 0 with
all four set -- which is why they are named here instead of a test total: the count of tests
changes with every commit and nothing can pin it, while the skip count is a property of the
gating.

```bash
MODEL=/tmp/stories15M-q4_0.gguf \
QWEN2_MODEL=/tmp/qwen2-f32.gguf \
MIXTRAL_MODEL=/tmp/tinymixtral.gguf \
MODEL_KQUANT=/tmp/tinymixtral.gguf \
go test ./...
```

Worth knowing before judging that run: a file whose weights were never trained produces
a flat next-token distribution -- one measured at p(argmax) = 0.0003 against 0.26 for a
trained checkpoint -- so its output is gibberish whatever the implementation does. The
test therefore asserts that zeroing the experts changes the logits, which is a statement
about the code, rather than that the text reads well, which would be a statement about
the file.

## Performance Expectations

Orbix is a learning project, and it is honest about where it stands:

- **CPU-only.** No GPU support. For GPU throughput, use vLLM, SGLang, or TensorRT-LLM.
- **Prefill is superlinear, and attention is why.** On Llama 3.2 1B Q2_K with 128
  workers, interleaved runs of two sizes gave 4x the tokens for 2.78x the time (501
  tokens in 8.7-14.5 s, 2001 in 32 s). Interleaving matters here: taking the two
  sizes in separate runs let load drift put a 2.5x spread on the smaller one, which
  is wider than the effect being measured. Attention is the term that grows: before
  it was parallelized across workers, 2001 tokens took 177 s and 16 layers of
  attention were 135 s of that.
- **Treat the numbers above as indicative, not as a benchmark.** They were taken on
  a shared machine whose load drifted between 40 and 90 over the course of a few
  minutes, and repeat runs of the same workload varied by more than the improvements
  being measured -- a 3721-token prompt once timed faster than a 3641-token one.
  The direction of each change is sound and the bit-identity tests are unaffected,
  but a timing claim from this machine needs a quiet one to confirm. `orbix bench`
  prints its own conditions so a reader can judge a table taken elsewhere.
- **Reference kernels are not optimized per instruction.** They are written to be
  readable, with no SIMD and no assembly. What they do use is more than one core: the
  row-partitioned products run on a bounded worker set, which took decoding on
  stories15M from 65 to 235 tokens per second on a 128-core EPYC.
- **SIMD and assembly are deliberately out of scope.** This project exists to make
  inference legible in one language, and a kernel that cannot be read in the
  repository's own language defeats that. Parallelism over independent rows is where
  the speed comes from instead, and it is what the numbers above measure.
- **A model runs in its trained context, and no larger.** The rope table is built for
  that window, so `--ctx` accepts a smaller value and refuses a larger one with a
  message naming both numbers. Going past it needs rope scaling -- the frequency
  adjustment a long-context fine-tune applies -- which this engine does not implement.
  A generation budget larger than the window is cut to fit, and `--stats` says when
  that happened, because a run that asked for 128 tokens and printed 123 otherwise
  looks like the model stopped early.
- **Parallelism is over rows, never over a reduction.** Every output row is computed by one goroutine in the same order the serial kernel uses, so float32 addition is never reassociated and the parallel result is bit-identical. That is a deliberate constraint: a split reduction would be faster on paper and would silently change every number the project compares against llama.cpp.
- **Scheduling turns concurrency into throughput.** A round prefills every pending prompt and decodes every active sequence in one pass, so total throughput scales with concurrency: on stories15M-q4_0 it goes 237 tok/s at one stream to 571 tok/s at sixteen. Per-stream speed still drops as streams are added, because the model is one shared resource.
- **A shared prompt costs almost nothing the second time.** With `--prefix-cache`, a request whose prompt starts with blocks the pool has already computed attaches them instead of prefilling them again. On a 96-token shared prefix, one prefill drops from 56.7 ms to 4.1 ms (14x), and four requests with the same prompt finish in 154 ms instead of 246 ms. This is the case a chat application actually has: a system prompt and earlier turns are re-sent verbatim on every call.
- **Benchmarks are self-run.** `orbix bench` is part of the CLI so you can measure on your own hardware, with your own model. It reports the worker count it actually used.

The intended comparison is not "Orbix vs vLLM on an A100". It is "what does it take to make LLM inference work in pure Go, and what do I learn by building it?"

## Roadmap

The roadmap is a learning path as much as a feature list. Each version adds one conceptual layer.

- [x] `v0.1` — **GGUF and first kernels**: GGUF loader, F32/F16/BF16/Q4_0/Q8_0 decode, reference matmul/RMSNorm/softmax/SiLU kernels, SentencePiece tokenizer, Llama forward pass, `orbix run`, `orbix model inspect`
- [x] `v0.2` — **Scheduling and cache**: continuous batching, a paged KV cache, Q4_0/Q4_1, Qwen2 — batched prefill and decode with an admission policy, a reference-counted paged cache with prefix reuse, every 32-weight ggml block format, Llama and Qwen2
- [x] `v0.3` — **Serving**: `orbix serve` with OpenAI-compatible SSE, priority queues, Q5_0/Q5_1, Mistral — SSE streaming, client-disconnect cancellation, a per-request `priority` field, an admission policy, every 32-weight and K-quant block format, the `mistral` architecture with its sliding window, chat templates rendered from `tokenizer.chat_template`, and mixture-of-experts layers
- [x] `v0.4` — **Performance paths**: K-quants, prefix-aware cache — every K-quant and IQ4_NL decode, and a paged cache whose prefix index records which block continues which so eviction gives up a chain's end rather than its middle. Assembly is deliberately out of scope: see below
- [x] `v0.5` — **Measurement and observability**: `orbix bench --suite` prints every scenario as one table with the conditions that produced it, `/metrics` exports Prometheus-format counters and gauges, `--otlp-endpoint` exports traces in OTLP/HTTP+JSON, and `serve --log-format` writes one structured access record per request. A remote HTTP backend was considered and dropped: see the note under Architecture
- [x] `v1.0` — **Stable interface**: a cross-platform CI matrix that builds five targets and runs the suite on two, benchmarks published with the machine and flags that produced them, and a documented reading path through every layer. "Stable" means the interfaces are settled enough to read as a whole, not that they will not change: the project is still pre-alpha

Every version on that list is complete, and that means less than a green checklist
suggests. It means each layer has an implementation that is correct against its reference
and small enough to read; it does not mean the project is production-ready. There is no
GPU path, no request queuing beyond one process, no model architecture beyond the
Llama-family, and no plan to add those. The roadmap was a reading order, and it is
finished because the reading order is finished.

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

| Flag                       | What it controls                                                    |
| -------------------------- | ------------------------------------------------------------------- |
| `--model`                  | Checkpoint path; falls back to `ORBIX_MODEL`                        |
| `--max-active`             | Sequences advanced per round, and therefore the batch size          |
| `--max-queued`             | Backlog accepted before new requests get 503; 0 means 8x max-active |
| `--policy`                 | Which sequences fill a round: `fifo`, `shortest`, or `fair`         |
| `--workers`                | Kernel goroutines per batched pass; 0 means one per CPU             |
| `--blocks`, `--block-size` | Paged KV cache; without them each request reserves a whole context  |
| `--prefix-cache`           | Reuse KV blocks across requests sharing a prompt prefix             |

A config file is on the roadmap, not in the code: the interface block at the top of
this file describes the target, and this table describes what exists.

The two cache layouts differ in what they reserve, and the difference is large enough to
decide the flag. A contiguous cache reserves the whole context for every accepted request
when it arrives, so the cost is `context x KV width x layers`, multiplied by the active
and queued slots. A paged cache takes blocks as a sequence generates, so memory follows
what is actually in flight. The server prints the figure it computed at startup, and warns
when a contiguous reservation passes a gibibyte.

`--workers` divides one batched pass across goroutines, so the useful setting is bounded by
what a pass contains: a round advances at most `--max-active` sequences, and a prefill has
one position row per prompt token in the batch. Above that the extra workers have nothing to
split. Measured on a 128-core host with eight concurrent requests, medians of five interleaved runs
gave 324.9 tokens per second at 8 workers and 327.6 at 128 -- the same within noise, so the
count is not a throughput knob once it passes the batch, and the default of one per CPU is
already more than an eight-way batch can use. A single run of either setting varies by more
than that difference, which is why the numbers here are medians of interleaved runs.

For scale: the 128k-window Llama 3.2 checkpoint in this repository's test set reserves
about 8 GB per request that way, while the small test fixture reserves 1.7 MB. Any model
with a long window should be run with `--blocks`, where the total is the pool size and
`--block-size` positions per block.

## Development

```bash
git clone https://github.com/pplmx/orbix
cd orbix

make test      # CGO_ENABLED=0 go test ./...
make lint      # golangci-lint run
make check     # the local gate: vet, test, lint, gofmt
make docs      # rumdl check: Markdown is code here too
make fuzz      # run every fuzz target; FUZZTIME sets the per-target budget
make build     # CGO_ENABLED=0 go build -o bin/orbix ./cmd/orbix
make bench     # run kernel and engine benchmarks
```

`make check` is what a contributor runs before pushing; it needs only the Go toolchain and
golangci-lint. `make docs` covers the README and every `doc.go` and needs rumdl, so it is a
separate target rather than part of `check` — the two are the gates CI runs, and running both
makes a green branch predictable rather than hoped for.

`make fuzz` searches rather than gates, which is why it is separate too. Six targets cover the
paths that read untrusted bytes: the GGUF container, the quantization decoders, the chat template
parser, and the tokenizer's pre-split and decode. Each keeps a seed corpus that `go test` replays
on every run, so a crash that was found once stays fixed without anyone remembering to fuzz; the
target itself is what finds the next one, and costs a few minutes.

Reading a file is the one thing here that runs on an arbitrary input — a model someone else
produced — and a panic or an allocator kill during it takes the process down rather than
returning an error. That is what the targets exist to rule out.

CI enforces `CGO_ENABLED=0` on every platform. A build that requires CGo is a build that fails.

Renovate keeps the workflow's pinned actions current: `.github/renovate.json` extends
`config:recommended`, so a new `actions/checkout`, `actions/setup-go`, or lint action release
arrives as a pull request rather than as a stale pin nobody notices. Nothing in the build
depends on it — the versions in `ci.yml` are the ones used.

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
