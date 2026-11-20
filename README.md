# Orbix

**A pure-Go, CGo-free LLM inference engine — built to learn how LLMs actually work.**

GGUF loading, quantized kernels, KV cache, continuous batching, and OpenAI-compatible serving — one layer at a time, in Go.

[![Go Version](https://img.shields.io/badge/go-1.27%2B-00ADD8?logo=go)](https://go.dev)
[![CGo](https://img.shields.io/badge/cgo-disabled-brightgreen)](https://github.com/pplmx/orbix)
[![Status](https://img.shields.io/badge/status-learning--in--public-orange)](https://github.com/pplmx/orbix)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> **Status: learning project, pre-alpha.** The API will change. The code prioritizes readability and correctness over performance.

Orbix reads GGUF models, runs quantized transformer inference on CPU, schedules requests with continuous batching, and serves an OpenAI-compatible API — from a single static binary with no CGo, no shared libraries, and no external runtime.

It is not competing with vLLM or llama.cpp. It is a **learning vehicle**: each layer of inference implemented explicitly in Go, small enough to read end to end, so that quantization, attention, KV cache and batching each have a concrete, inspectable implementation.

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

Most LLM inference code lives in Python + CUDA, wrapped in C++ kernels, and hidden behind
framework abstractions. That is the right choice for production and a hard place to *learn*.
Orbix asks whether inference can be understood by building it in Go, from the file format up. The
constraints follow from that:

- **Pure Go, no CGo.** `CGO_ENABLED=0` is a learning constraint here, not a deployment trick: if
  a C library does it, it has to be written out to be understood.
- **Readability over performance.** Kernels are written to be read. Assembly fast paths come
  later, behind the same interface.
- **One layer at a time**, so each can be studied independently.
- **Honest benchmarks.** No cherry-picked numbers; where Orbix is slower than llama.cpp, this
  file says so.

The cost, plainly: pure-Go CPU inference is slower than C++ with hand-tuned SIMD. That is not the
comparison this project is trying to win.

## Implementation Status

Every layer on the roadmap exists. The package's `doc.go` says how it was verified.

| Layer          | Package             | State                                                                     |
| -------------- | ------------------- | ------------------------------------------------------------------------- |
| GGUF container | `internal/gguf`     | Header, metadata, tensor directory, streamed tensor data.                 |
| Half precision | `internal/f16`      | binary16, including subnormals, infinities, NaNs. All 65,536 patterns.    |
| Dequantization | `internal/quant`    | F32/F16/BF16, Q4/Q5/Q8, IQ4_NL, Q2_K-Q6_K.                                |
| Kernels        | `internal/kernel`   | MatMul, MatMulTransposed, MatVec, RMSNorm, Softmax, SiLU.                 |
| Tokenizer      | `internal/tokenize` | SentencePiece and byte-level BPE; every pre-tokenizer name llama.cpp maps.|
| Chat templates | `internal/chat`     | The Jinja2 subset a checkpoint's `tokenizer.chat_template` needs.         |
| Engine         | `internal/engine`   | Continuous batching, admission policy, paged or contiguous KV cache.      |
| Transformer    | `internal/llama`    | Llama and Qwen2 forward passes, RoPE, GQA, prefix reuse, MoE.             |
| Tracing        | `internal/otel`     | OTLP/HTTP+JSON spans, hand-written, no SDK.                               |
| Server         | `internal/server`   | OpenAI-compatible API, SSE, `/metrics`, access log.                       |

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
   independent outputs -- never over a reduction -- so a parallel result equals the serial
   one exactly.
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

Solid boxes are what the code does today. Two components a project like this usually has are
deliberately *not* drawn:

- **No `Backend` abstraction.** Continuous batching, prefix reuse, and the paged cache all work
  by driving one batched forward pass over local KV blocks. A remote backend can only be asked
  for a prompt and answer with tokens, so the seam would leave the local path with the
  mechanisms and the remote path silently without them -- a difference invisible at the call
  site.
- **No `KVCacheManager` interface and no radix tree.** The paged cache and its hashed prefix
  index cover the job: a shared pool of refcounted blocks plus an index that hands back what a
  repeated prefix already produced. Matching is whole-block, so a radix tree's sub-block
  granularity buys nothing, and eviction is explicit instead: each entry records the block it
  continues, so eviction takes the end of a chain and leaves the head reachable.

**Mixture of experts** is a property of the feed-forward layer rather than a separate
architecture: `expert_count` replaces the single gated FFN with one per expert plus a router,
and `expert_used_count` run per token. **Compute kernels** are the pure-Go work, written to be
read, with no SIMD fast paths.

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

The compute kernels are real, exported, and callable from this repository's own packages. The
signatures are the ones in `internal/kernel` today:

```go
package kernel

func MatMul(dst, a, b []float32, m, n, k int) error
func MatMulTransposed(dst, a, b []float32, m, n, k int) error
func MatVec(dst, a, v []float32, n, k int) error
func RMSNorm(dst, src, weight []float32, eps float32) error

// NewRunner returns a Runner that may use up to workers goroutines.
func NewRunner(workers int) *Runner
func (r *Runner) MatVec(dst, a, v []float32, n, k int) error
func (r *Runner) MatMulTransposed(dst, a, b []float32, m, n, k int) error
```

Decoding a quantized tensor is not a kernel: it lives in `internal/quant`, so a Q4_0 matrix
arrives here already widened to float32. Attention is not a kernel either -- it is assembled in
`internal/llama` from `Softmax`, `RMSNorm`, `AddBias` and `AddInPlace`, because the mask and the
cache layout are model concerns rather than arithmetic ones.

## Quick Start

`orbix run`, `orbix model inspect`, `orbix bench`, and `orbix serve` work today. Embedding the
engine in another Go module does not, for a reason the section at the end of this chapter
explains: the packages are under `internal/`.

### Serve a GGUF model

```bash
CGO_ENABLED=0 go install github.com/pplmx/orbix/cmd/orbix@latest

orbix serve \
  --model ./models/qwen2.5-7b-instruct-q4_0.gguf \
  --addr :8080
```

The prefix is not decoration. `go install` uses whatever CGO_ENABLED the environment has, which
is 1 on a machine with a C compiler, and a binary built that way reports `cgo: true` in
`orbix version` — at odds with the only configuration this project supports. The code contains no
`import "C"`, so such a build works, but it is not the artifact the project claims to ship.

The server implements `/v1/chat/completions`, `/v1/completions` (both streaming
and not), `/v1/models` (and `/v1/models/{id}`, which the SDKs' `models.retrieve` calls),
`/healthz`, and `/metrics`. A path under `/v1` this server does not implement answers 404 in
the API's JSON error envelope rather than Go's plain-text one, because the SDKs decode the body
before they raise: a text 404 reaches the caller as a parse failure, not as a not-found.

Every error body carries a `code`, including the ones a client hits before any field is parsed:
`invalid_json`, `unsupported_encoding`, `body_too_large`, `invalid_value`,
`missing_required_parameter`, `unsupported_parameter`, `context_length_exceeded`,
`model_not_found`. OpenAI's guidance is to branch on `code` rather than message text. `param` is
present when one field is at fault and absent when the failure is about the body as a whole, and
`type` separates `invalid_request_error` from `server_error` and from
`service_unavailable_error` -- the one the SDKs retry.

Unsupported fields are handled by what a client does with the answer. One that only tunes
sampling -- `logit_bias`, `presence_penalty`, `response_format` -- is ignored, because the SDKs
send such fields whether or not the engine can use them. One that describes the *response* is
refused with a `400` naming it and `code: unsupported_parameter`: `n`, `logprobs`, `top_logprobs`,
`echo`, `suffix`, `best_of`. A client that sent `n: 3` indexes `choices[2]`, so one choice is not
a smaller response but a different one. Several at once are all named, so one round trip fixes
the request.

`max_completion_tokens` is refused for a third reason, and it is the only field here whose
silent omission a client cannot notice at all. OpenAI's newer name for `max_tokens` has to be
sent *instead of* it, so a client that has migrated sends only the new one; the server read
`max_tokens`, found zero, and generated its default 256 tokens for a request that asked for 3.
The reply parsed, the shape was right, and nothing said the length was wrong -- measured on
both completions and chat. Ignoring a limit is worse than ignoring a sampling knob, so it gets
the same 400 the response-shape fields do.

`stream_options.include_usage` is honoured rather than ignored, because it is the only way a
streaming caller learns what it spent: without it a streamed response reports no usage at all.
With it, one extra chunk arrives before `[DONE]` whose `choices` is an empty array and whose
`usage` holds the whole request's counts, and every ordinary chunk in the same stream carries
`"usage": null`. Both halves are the documented contract, and the counts equal what the
non-streaming body reports for the same request.

Two things about the chat
endpoint are worth knowing before pointing a client at it:

- **The chat template is the model's own, when it can be evaluated.** GGUF stores a
  checkpoint's format as a Jinja2 string under `tokenizer.chat_template`, and
  `internal/chat` evaluates a subset of it: `if`/`elif`/`else`, `for` with the `loop`
  variables, `set`, `+` concatenation, comparisons, the `is` tests `defined`, `none`,
  `mapping` and `iterable`, and the `trim`, `tojson` and `length` filters. That covers the
  published templates for Qwen2.5 and Llama 3.1 exactly, and both are tested against their
  real source.
  A template using syntax outside the subset is reported at parse time; one calling a
  function this package does not implement fails at render time instead, because the parser
  accepts any call. Either way the server falls back to a neutral `Role: content`
  transcript, printing which of the two it is using at startup. The fallback matters: a
  half-rendered template would produce a prompt that looks plausible and is wrong, while a
  neutral transcript is merely less good than the model's own format.
  If you would rather control the prompt, send it to `/v1/completions`, which passes
  text through verbatim.
- **`stop` is matched on text, and stops generation early only for single-token
  strings.** An earlier version of this note said a stop spanning several tokens
  was not matched, which understated it: the decoded text is checked against every
  stop string, so `"中国的首都"` -- four tokens -- truncates correctly. What a
  multi-token string does not do is stop generation sooner; the sequence runs to
  its token budget unread while the response is already cut. For `"\n"` and other
  markers, which encode to one token, the engine halts the sequence itself.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Explain continuous batching."}],
    "stream": true
  }'
```

### Behind a reverse proxy

The server sets `X-Accel-Buffering: no` on streaming responses, which is the one thing it can
do from its side of the connection. nginx buffers proxied responses by default
(`proxy_buffering on`), so without that header an SSE stream still arrives complete but all at
once, which for a token stream is the same as not streaming. nginx reads the header and turns
buffering off for that response only; every other endpoint keeps the buffering that a proxy
default is actually for. Proxies that do not understand the header ignore it.

Two things remain the deployment's job, because no application header can express them:

- **Raise `proxy_read_timeout`.** nginx defaults to 60 seconds bounding the gap between reads,
  which a long model load or a stalled generation can exceed.
- **Keep reading.** A stream nobody reads is abandoned, not paused, once the backpressure
  reaches the engine: 64 events buffer per sequence, then two seconds of grace, then the slot
  goes to someone else. The bound is on the engine's queue rather than on the network, and
  kernel socket buffers absorb a surprising amount, so the guarantee is "drain as you go" --
  which every SSE client does by default -- not "you have two seconds".
- **Use HTTP/1.1 upstream.** nginx defaults to 1.1 from 1.29.7; before that it is 1.0, where
  ending a stream means closing the connection.

`X-Accel-Buffering` is nginx-specific and consumed rather than forwarded (it is in the default
hidden-header set), so a two-tier nginx chain needs `proxy_pass_header X-Accel-Buffering` on the
inner tier. HAProxy and Envoy forward bytes as they arrive and need nothing; Caddy flushes
`text/event-stream` regardless. All of them still need timeouts sized for a response that can
outlive the default.

### No CORS and no authentication

The server sends no `Access-Control-Allow-*` headers, so a browser page on another origin
cannot read a response -- and because the request body is JSON decoded without a Content-Type
check, a page can still *send* one. `POST` with `Content-Type: text/plain` is a CORS "simple
request", which the browser delivers without a preflight, and the server parses the body
whatever the header says. The page cannot see the reply, but the request runs:

```bash
# Accepted from any origin, without a preflight, from a page the user merely visited.
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: text/plain' -H 'Origin: http://example.invalid' \
  -d '{"prompt": "...", "max_tokens": 256}'
```

This is a property of an unauthenticated API that also performs work, not a bug in the CORS
decision. Adding CORS would not close it, because CORS governs who may read a response, not who
may cause one; and checking the Content-Type would only narrow it, since the tolerance is what
makes the `curl -d` form above usable in the first place. Measured with `--max-active 2`: six
such requests from a page's origin were accepted in a row and generated 240 tokens between
them, while the preflighted equivalent is already refused -- `OPTIONS` on these routes answers
405, so a cross-origin `application/json` request never leaves the preflight.

What closes it is binding to loopback or authenticating in front. The default `--addr :8080`
listens on every interface; a host reachable from elsewhere is an inference API anyone can
spend.

Request bodies are bounded, which limits how much work one connection can cause: the limit is
`ContextLen` x 48 bytes with a 1 MiB floor, so 6 MiB for the 128k-context checkpoint. That
number is not a tuning knob so much as the same constraint the context already imposes, seen
from the request side -- a body larger than this cannot describe a prompt the model could
attend over. Anything over the limit is answered `413` and the connection is closed without
reading the rest. Without the bound the cost rose with the body: a 64 MiB request spent 82
seconds being read and tokenized before the server could say it was too long.

### Embedding it in a Go service

Not supported today, and the reason is structural rather than an omission. Every package that
would do the work lives under `internal/`, which Go forbids another module from importing:

```text
use of internal package github.com/pplmx/orbix/internal/engine not allowed
```

That is deliberate for now -- `internal/` is what makes the package boundaries a rule the
compiler enforces rather than a convention, and the layers have moved enough that a frozen
public API would have been a liability. The interfaces in [Core Interfaces](#core-interfaces)
are the shape such an API would take; the table there says which of them exist and which do
not, and nothing in this section should be read as a promise that they are reachable.

What works instead:

- **Run the server and talk HTTP.** `orbix serve` is the supported integration surface, and it is
  the OpenAI shape, so the client libraries already exist: see
  [Serve a GGUF model](#serve-a-gguf-model), and the `curl` calls there are the contract.
- **Fork or copy what you need.** The code is structured to be read, and `internal/engine` is the
  entry point a fork would call: `engine.New(path, engine.Options{...})` returns an engine,
  `Submit` queues, `Pump` advances a round, and `Stats` reports. That is what the CLI does.

### Cross-compile

```bash
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o orbix-arm64 ./cmd/orbix
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o orbix.exe ./cmd/orbix
GOOS=linux GOARCH=riscv64 CGO_ENABLED=0 go build -o orbix-riscv64 ./cmd/orbix
```

`CGO_ENABLED=0` is repeated on each line rather than exported once above them, because these are
lines a reader copies one at a time and a forgotten export is invisible until the binary reports
`cgo: true`.

No toolchain setup beyond Go itself. The version is not incidental: `go.mod` declares `go 1.27.0`,
and the code uses generic methods and `strings.CutLast`, both of which arrived in that release, so
an older toolchain fails to build rather than degrading quietly.

## CLI

Every command below marked with a version works today. `orbix chat` runs on the
engine with a paged cache and prefix reuse, so a turn's prompt starts with the previous
turn's and the blocks already computed are re-attached instead of prefilled again. Reuse is
whole-block only, so a conversation shorter than one block (the REPL's default is 16
positions) has nothing to attach: `/stats` reports the positions actually reused, which is
0 until the history outgrows a block and 32 once it does on a 30-word message.

`/stats` also reports the KV pool, `N of M blocks free (S positions each, T total)`, because
running out of it is one of two failures a session hits repeatedly -- the other being the
conversation outgrowing the model's window. Both errors say that retrying changes nothing and
which flag is the way out. The total is there because the free count alone misleads: a session
refused on every turn may show all blocks free, which means the request is larger than the pool
can ever hold rather than that anything is contended.

`/clear` forgets the conversation and nothing else. The blocks previous turns published stay in
the prefix index, so the next prompts still attach what they share with the old history -- which
is what makes it a way out of a full window rather than a way to start from scratch.

| Command                      | Description                                                                                                  | Status |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| `orbix model inspect <path>` | Print GGUF metadata, tensor types, quantization, and whether this build reproduces the model's pre-tokenizer | v0.1   |
| `orbix run "<prompt>"`       | One-shot inference                                                                                           | v0.1   |
| `orbix version`              | Print version and build info; `-json` for a machine-readable object                                          | v0.1   |
| `orbix chat`                 | Interactive REPL with conversation history                                                                   | v0.3   |
| `orbix serve`                | Start the OpenAI-compatible inference server                                                                 | v0.3   |
| `orbix bench`                | Run a built-in throughput/latency benchmark                                                                  | v0.2   |

```bash
orbix run --model ./model.gguf "Once upon a time"
orbix run --model ./model.gguf --stats --max-tokens 64 "Explain KV caching."
orbix run --model ./model.gguf --workers 32 "Explain KV caching."   # see below on the ratio
orbix run --model ./model.gguf Once upon a time    # the same prompt, unquoted

orbix chat --model ./model.gguf

orbix bench --model ./model.gguf --concurrency 4 --max-tokens 32
orbix bench --model ./model.gguf --suite     # every scenario in one table

orbix model inspect ./model.gguf
orbix model inspect --json ./model.gguf   # machine-readable; values keep their GGUF type
```

`orbix model inspect` is also a learning tool: it prints the GGUF layout in a readable form.

The human-readable tables cut a metadata value or a tensor name at 60 bytes with a rune-safe
`...`, because one long name otherwise pushes a table's later columns off screen. `--json`
carries the whole name, since a script looking a tensor up needs the one the file records.

`--json` keeps each value the type the file declares, with one exception the format forces: a
NaN or infinite float is emitted as the string `"NaN"`, `"+Inf"` or `"-Inf"`, because JSON has
no token for either. Such a value is legal in GGUF, and the alternative was one such key
failing the whole document.

`orbix bench --suite` runs a fixed set of scenarios and prints them as one table. It pairs each
mechanism with a control row -- paged against contiguous, prefix reuse against none, shortest
against FIFO -- so a difference is attributable to that mechanism.

How much prefix reuse is worth depends on the mix: the pair in the table shares 41 prompt tokens
against 8 generated per request. Raising `--max-tokens` dilutes the saving, since reuse does not
touch generation. A long prompt with a short answer is where it pays.

The suite fixes the settings its rows compare -- concurrency, request count, admission
policy, and cache shape -- so those flags are refused rather than ignored when `--suite` is
given, and the error names every one that was passed. `--model`, `--max-tokens`, `--prompt`,
and `--workers` still apply: they describe the machine and the workload, not the mechanism.

One measurement, stated with everything needed to judge it: stories15M q4_0 (19 MB), a
41-token prompt, 8 generated tokens per request, 8 kernel workers, on a 128-core AMD EPYC
7742, Go 1.27.3, linux/amd64. `orbix bench --suite` prints these conditions itself, so a
table you produce can be compared against this one instead of taken on faith:

```text
SCENARIO         TOKENS  ELAPSED  TOK/S  FIRST  NOTE
single stream    8       78ms     102.3  41ms   baseline: one request, contiguous cache
batch x4         32      266ms    120.4  159ms  batched decode, contiguous caches
batch x16        128     1001ms   127.8  637ms  where batched decode stops scaling
paged x16        128     1015ms   126.2  651ms  same batch, block cache instead of whole-context reservations
prefix no reuse  32      317ms    101.0  42ms   four requests sharing one prompt, one after another, reuse off
prefix reuse x4  32      229ms    139.5  41ms   the same four one after another, reuse on
policy shortest  63      401ms    156.9  41ms   MaxActive 1, a short request queued behind a long one
policy fifo      63      415ms    151.7  401ms  the same queue without reordering
```

Those are medians of nine interleaved runs: a single run moves by more than the difference
between neighbouring rows. The `NOTE` column in a live run is longer -- it also reports the
block pool's shape and the positions reused, trimmed here to fit.

`FIRST` is a latency of a different request depending on the row: on a throughput row it is
how long until *any* request produced a token, and on the two policy rows it is the **short**
request's own first token. Three rows need a word because their numbers look flat or backwards:

- **Paged vs contiguous** is a memory trade, not a throughput one: the same batch at the same
  speed, with 16 sequences sharing one pool instead of each reserving a whole context.
- **The policy rows do the same work and can exceed `--max-tokens`.** A policy reorders a fixed
  quantity of work, so `TOK/S` and `ELAPSED` there are machine load rather than a property of
  the scheduler; the round in `NOTE` is the property (2 under `shortest`, 63 under `fifo`).
  The rows raise any budget below 60 tokens so there is a queue to order.
- **The prefix rows run their four requests one after another.** A concurrent burst prefills
  them together, so no request arrives after another cached its prefix. Both rows repeat the
  prompt until it spans two blocks, because a block is published only once its last position
  is written.

A row can show fewer tokens than its budget without being a measurement failure: the model
emitted its end-of-sequence token, and the `NOTE` column says so. A single measurement reports
the same kind of shortfall as a `note:` line for the case the suite cannot hit -- a window that
leaves less room than `--max-tokens` asks for -- naming both counts, because `finish: length`
reads as "the budget was reached" either way.

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

The exposition is written by hand rather than through a client library, for the same reason
the GGUF parser is: a family is three lines, and the encoder is read by code review rather than
trusted as a dependency. Sixteen families come to 52 lines on a paged server, because a
labelled series costs one line per value.

The count depends on the configuration: a contiguous server exports twelve families (the four
`kv_pool_*` gauges are omitted without a pool, since a zero would read as "the pool is empty"),
and `--otlp-endpoint` adds one more. Families are omitted, never exported as zero.

Three exist because their failure mode is invisible in latency until it is already failing.
`orbix_requests_queued` growing against a steady round rate is capacity, not a slow model;
`orbix_kv_pool_free` reaching zero is a paged deployment about to refuse requests. The third,
`orbix_requests_finished_by_reason_total`, is what catches clients leaving: a disconnect looks
entirely healthy in every other number, and shows up here as `reason="cancel"` or
`reason="abandoned"` growing against `reason="stop"`. The parts always sum to
`orbix_requests_finished_total`, and the tests assert it.

`orbix_requests_refused_queue_full_total` and `orbix_requests_refused_pool_total` count the
requests answered `503`. They are the server's own counters rather than the engine's, since a
refused request never enters the engine and counting it as submitted would flatter the admission
rate. Two families rather than one, because the fixes differ: a full backlog wants a deeper
`--max-queued` or more replicas, an exhausted pool wants more `--blocks` or shorter prompts.

### The access log

`serve --log-format json` writes one record per request to **stderr**, not stdout -- the
startup banner is the thing on stdout, so a platform that captures only stdout gets the
banner and none of the requests. The banner names the stream for that reason. `text` is the
same record rendered for a terminal, and `off` disables it while leaving the banner alone.

```json
{"time":"2026-11-20T12:09:28.085356937+08:00","level":"INFO","msg":"request",
 "method":"POST","path":"/v1/completions","status":200,"duration":7033502,"bytes":232}
```

Five fields answer "what happened to that request": the method, the path, the status, the
duration and the body's byte count. Two details are worth knowing before charting it.

`duration` is **nanoseconds**, as a bare integer. That is `slog.Duration`'s JSON encoding,
not a choice made here, and it is the field most likely to be misread -- a dashboard that
assumes milliseconds is off by 1000x. The `text` format prints it with a unit (`643.921µs`)
precisely because a bare number has none.

`bytes` is what the handler actually wrote, which is what makes a streaming record useful:
505 bytes for a five-chunk SSE response is a statement about the response, where a status
alone would not distinguish it from an empty one. A 4xx is logged at `INFO` and only a 5xx
at `WARN`, on the reasoning that a client's mistake is not a server failure and logging
those as warnings teaches a reader to ignore the level. A `503` from a saturated queue
*is* logged at `WARN`: capacity running out is the operator's problem.

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

GGUF parsing and tokenization are implemented in pure Go from the start; the quantization table expands incrementally. Two ggml types are defined and sized but not decoded -- `Q8_1` and `Q8_K` -- and both are conversion-tool intermediates rather than formats a released checkpoint uses. A file carrying either is refused with `cannot decode <type>` at load, the same way an unrecognised type is.

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
QWEN2_MODEL=/tmp/qwen2-f32.gguf CGO_ENABLED=0 go test ./internal/llama/ -run Qwen2
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
MIXTRAL_MODEL=/tmp/tinymixtral.gguf CGO_ENABLED=0 go test ./internal/llama/ -run TestMoEReachesTheExperts
```

The stories model is a 19 MB public download, and it is the one of the four that takes no
conversion step:

```bash
curl -sSL -o /tmp/stories15M-q4_0.gguf \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
```

The four gates take the suite from fifteen skipped tests to none, so a machine holding the
checkpoints can check every claim the project makes: `go test ./... -v | grep -c '^--- SKIP'`
reads 15 without a checkpoint and 0 with all four set. Re-measure rather than trusting the
number -- adding a gated test raises it.

```bash
MODEL=/tmp/stories15M-q4_0.gguf \
QWEN2_MODEL=/tmp/qwen2-f32.gguf \
MIXTRAL_MODEL=/tmp/tinymixtral.gguf \
MODEL_KQUANT=/tmp/tinymixtral.gguf \
CGO_ENABLED=0 go test ./...
```

`just test` does not set these variables, so the gated tests skip under it exactly as they do
without them. Writing the block out here is what a reader holding the checkpoints needs; putting
the paths in the recipe would mean every contributor's `just test` ran a different suite.

An untrained file produces a flat next-token distribution, so its output is gibberish whatever
the implementation does. The MoE test therefore asserts that zeroing the experts changes the
logits -- a statement about the code, not about the file.

## Performance Expectations

Orbix is a learning project, and it is honest about where it stands:

- **CPU-only.** No GPU support. For GPU throughput, use vLLM, SGLang, or TensorRT-LLM.

- **Prefill throughput improves with prompt length, within the window.** On Llama 3.2
  1B Q2_K with 32 workers, the per-token cost fell from 52 ms at a 155-token prompt to
  40 ms at 614 tokens (medians of five interleaved runs). Attention is O(n^2) but
  parallelises over positions, while the projections are O(n) over columns that do not
  grow, so the growing term is the one that gains workers as it grows.

- **Treat the numbers above as indicative, not as a benchmark.** They were taken on a
  shared machine whose load drifted enough that repeat runs varied by more than the
  effects being measured. `orbix bench` prints its own conditions so a table taken
  elsewhere can be judged.

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

- **A shared prompt costs almost nothing the second time.** With `--prefix-cache`, a
  request whose prompt starts with blocks the pool has already computed attaches them
  instead of prefilling them again. The suite's reuse pair shows the effect end to end:
  101.0 tok/s against 139.5 when four sequential requests share one prompt, with the row
  reporting the 96 positions that came from the cache. Every `orbix bench` run with the
  flag on prints how many positions were reused, so a run that shares nothing says so
  instead of looking merely slow. This is the case a chat application actually has: a
  system prompt and earlier turns are re-sent verbatim on every call.

- **Benchmarks are self-run.** `orbix bench` is part of the CLI so you can measure on your own hardware, with your own model. It reports the worker count it actually used.

The intended comparison is not "Orbix vs vLLM on an A100". It is "what does it take to make LLM inference work in pure Go, and what do I learn by building it?"

## Roadmap

The roadmap is a learning path as much as a feature list. Each version adds one conceptual layer.

- [x] `v0.1` — **GGUF and first kernels**: GGUF loader, F32/F16/BF16/Q4_0/Q8_0 decode, reference matmul/RMSNorm/softmax/SiLU kernels, SentencePiece tokenizer, Llama forward pass, `orbix run`, `orbix model inspect`
- [x] `v0.2` — **Scheduling and cache**: continuous batching, a paged KV cache, Q4_0/Q4_1, Qwen2 — batched prefill and decode with an admission policy, a reference-counted paged cache with prefix reuse, the 32-weight block formats a checkpoint carries (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`), Llama and Qwen2
- [x] `v0.3` — **Serving**: `orbix serve` with OpenAI-compatible SSE, priority queues, Q5_0/Q5_1, Mistral — SSE streaming, client-disconnect cancellation, a per-request `priority` field, an admission policy, every 32-weight block format and five of the six K-quants, the `mistral` architecture with its sliding window, chat templates rendered from `tokenizer.chat_template`, and mixture-of-experts layers
- [x] `v0.4` — **Performance paths**: K-quants, prefix-aware cache — the five K-quant decodes a checkpoint actually carries (`Q2_K` through `Q6_K`) plus IQ4_NL, and a paged cache whose prefix index records which block continues which, so eviction gives up a chain's end rather than its middle. `Q8_K` is the one K-quant not decoded; it is an intermediate in conversion tools, and a file using it fails with `cannot decode Q8_K` rather than producing garbage. Assembly is deliberately out of scope: see below
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

`--block-size` cannot exceed the model's context: prefix reuse matches whole blocks, so a block
the context can never fill is never published and `--prefix-cache` would reuse nothing. `serve`,
`chat`, and `bench` refuse that combination rather than start it.

`--max-tokens` accepts 0 only on `run`, where "prefill and print nothing" is a result. The
server cannot express a zero-token request (`max_tokens` is an integer, so 0 is indistinguishable
from absent), a REPL turn generating nothing answers with an empty line, and a benchmark of zero
tokens is not a measurement. Negative values are refused everywhere.

A config file is on the roadmap, not in the code.

`--policy` needs a reason rather than a definition. `fifo`, the default, is submission order.
`shortest` takes the least remaining work first, which keeps a queue of mixed-size requests from
being dominated by its longest member, and reorders without bounding anything. `fair` takes
whichever sequence has gone longest without a turn, and is the one with a bound to quote: no
sequence waits more than about `len(active)/MaxActive` rounds in a row. That makes it the choice
for a long-running server, and the reason it is not the default -- a bound is only useful if the
operator asked for it.

What the three do to one queue -- `MaxActive 1`, four requests budgeted 16, 8, 4, 2 in submission
order, as the round each got its first token:

| Policy     | First-token rounds | What it did                                                          |
| ---------- | ------------------ | -------------------------------------------------------------------- |
| `fifo`     | 2, 19, 28, 33      | served in submission order, so each waited its turn                  |
| `shortest` | 19, 10, 5, 2       | exactly inverted: the 2-token request moved from round 33 to round 2 |
| `fair`     | 5, 6, 7, 8         | one turn apiece first, then round-robin from there                   |

The two cache layouts differ in what they reserve. A contiguous cache reserves the whole context
for every accepted request on arrival: `context x KV width x layers`, times the active and queued
slots. A paged cache takes blocks as a sequence generates, so memory follows what is in flight.
The server prints the figure it computed at startup and warns past a gibibyte.

A paged server answers `503` with `Retry-After` when the pool cannot hold a request *yet*,
which is a capacity signal a client or proxy should back off on. It does not answer that for
a request whose own prompt and budget need more blocks than the pool holds at all: nothing
another request finishes makes room for it, so retrying only repeats the answer. That case is
refused up front with a `400` whose message names both counts -- what the request needs and
what the pool has -- and suggests a shorter prompt, a lower `max_tokens`, or a larger
`--blocks`. Measured on the 256-position fixture with `--blocks 2 --block-size 16`: an
11-token prompt asking for 4 tokens returns `400` "needs 3 KV cache blocks ... but the pool
holds 2", while a shorter prompt against the same idle server returns `200`. Before that
check existed the request got `503 Retry-After: 1` on every retry, with
`orbix_kv_pool_free` sitting at 2 the whole time.

A streaming request cannot be told any of that. Its `200` and its headers are gone before the
pool runs out, so the failure arrives in the stream as a chunk with `finish_reason: "error"`
and the reason in a top-level `error` field -- no status code and no `Retry-After`, and a
client cannot tell a temporary capacity problem from a permanent one without reading the
message. This is the SSE format's constraint rather than a choice, which is why the refusal
counters still move on that path: `orbix_requests_refused_pool_total` is the only signal that
survives, and leaving it out made a streaming deployment report no capacity problem at all
while refusing every request.

`--workers` divides one batched pass across goroutines. What bounds its usefulness is the number
of independent outputs in a pass, not the number of sequences: a matvec splits by output row, so
a batch has one row per sequence, while the transposed products split by output *column*, and a
decode step has thousands of those whatever the batch size. On a 128-core host with eight
concurrent requests, 349.6 tok/s at 8 workers, 536.6 at 32, 543.9 at 128 -- a real knob below 32
and flat above it, so the default of one per CPU sits at the safe end.

On one stream instead of eight: stories15M q4_0, 128 tokens from one prompt, 64.2-65.1 tok/s at
one worker against 228.7-236.2 at 32 (3.52x-3.67x). The synthetic fixture reaches 3.9x because
its projection is large next to the per-round overhead; a 19 MB model does not.

For scale: the 128k-window Llama 3.2 checkpoint in this repository's test set reserves
8 GiB per request that way (`context_length x kv_heads x head_dim x 2 x 4 x layers`;
131072 x 8 x 64 x 2 x 4 x 16 = 8589934592 bytes). The synthetic fixtures are 32 KiB by
comparison, because their default window is 64 positions. Any model with a long window
should be run with `--blocks`, where the total is the pool size and `--block-size`
positions per block.

## Development

The commands below are [just](https://just.systems) recipes. `just` is a single binary and not
usually preinstalled, so install it first — `cargo install just`, `brew install just`,
`winget install --id Casey.Just --exact`, or a pre-built binary from its releases page. Every recipe in the block
below is a short shell command; the justfile is the readable list of what they are.

```bash
git clone https://github.com/pplmx/orbix
cd orbix

just test      # CGO_ENABLED=0 go test ./...
just lint      # golangci-lint run
just check     # the local gate: vet, test, lint, gofmt
just docs      # rumdl check: Markdown is code here too
just hooks     # prek: whitespace, final newlines, and YAML/TOML/JSON syntax
just commits   # cz check: commit subjects across a range; override RANGE=base..HEAD first
just release-preview # what the next release would be; changes nothing
just fuzz      # run every fuzz target; FUZZTIME sets the per-target budget
just build     # CGO_ENABLED=0 go build -o bin/orbix ./cmd/orbix
just bench     # run kernel and engine benchmarks
just cross     # build all five supported platforms with CGo disabled
just test-race # the suite under the race detector; needs cgo, so it is separate
```

`just check` is what a contributor runs before pushing; it needs the Go toolchain and
golangci-lint **v2**, because `.golangci.yml` is a v2 configuration and a v1 binary refuses to
read one. The recipe checks the version and says so, rather than letting the linter's own
message about the config stand in for it.

`just docs` runs `rumdl check .` and needs rumdl; `just hooks` runs the pre-commit hooks through
prek. Both are separate targets rather than part of `check` because neither tool is in the Go
toolchain, and `check` should stay runnable with Go and golangci-lint alone.

They differ in what they do on a problem: `just docs` reports and leaves the file alone, while
`just hooks` **rewrites** it (`rumdl` and `rumdl-fmt` pass `--fix`, and `end-of-file-fixer` and
`mixed-line-ending` edit in place). A hook that changed something exits non-zero; re-stage the
file and look at what it did rather than assuming the edit was cosmetic. `just check` sees Go
files only, which is why a stray whitespace line under `.github/workflows/` leaves it green and
`just hooks` catches it.

Run `prek install` once to make the hooks fire on every commit; it writes shims under
`.git/hooks/`, which git does not track.

`just fuzz` searches rather than gates. Six targets cover the paths that read untrusted bytes:
the GGUF container, the quantization decoders, the chat template parser, and the tokenizer's
pre-split and decode. Each keeps a seed corpus that `go test` replays on every run, so a crash
found once stays fixed; reading a file is the one thing here that runs on input someone else
produced, and a panic during it takes the process down rather than returning an error.

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
- Naming: brand **Orbix**, binary **`orbix`**, module path **`github.com/pplmx/orbix`**. There is no importable root package -- the layers are under `internal/` -- so a package named `orbix` would be a new decision rather than a rename.

## License

MIT © 2026 pplmx
