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

Every layer on the roadmap exists. Each row says what was verified rather than what was
intended:

| Layer          | Package             | State                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GGUF container | `internal/gguf`     | Parses header, metadata, tensor directory, and streams tensor data. Every field and byte offset cross-checked against an independent parser.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Half precision | `internal/f16`      | binary16 to/from float32 with correct subnormals, infinities, NaNs, and round-to-nearest-even. All 65,536 bit patterns verified against a reference.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Dequantization | `internal/quant`    | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. The 32-weight formats are pinned by exact-value tests on constructed blocks; the 256-weight K-quants add a round-trip error bound and a check against a real checkpoint tensor, because a wrong sub-block scale still decodes to plausible magnitudes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Kernels        | `internal/kernel`   | MatMul, MatMulTransposed, MatVec, RMSNorm, Softmax, SiLU, and elementwise helpers. A `Runner` splits the products across goroutines with a measured size threshold -- rows for `MatVec`, output columns for `MatMulTransposed`, since a decode step has one row and thousands of columns. Either way each output element is one serial dot product in the same order the reference kernel uses, so the parallel result is bit-identical to the serial one.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Tokenizer      | `internal/tokenize` | SentencePiece and byte-level BPE segmentation, with the four merge, space, and pre-tokenizer conventions read from the file rather than assumed. Thirty-four pre-tokenizer families are implemented (`qwen2`, `llama-bpe`, `gpt-2`, `starcoder`, `deepseek-v3`, `tekken`, `gpt4o`, `whitespace`, `bloom`, `deepseek-coder`, `falcon`, `deepseek-llm`, `mmbert`, `viking`, `superbpe`, `soprano`), covering all 100 names llama.cpp maps to a family (`dbrx`, `smaug-bpe`, `chatglm-bpe`, `glm4` and `glm5` share `llama-bpe`'s expression verbatim, and `gemma4`, `granite-embed-multi-311m`, `domyn-small` and `sarvam-moe` share `mmbert`'s, and `grok-2`, `bailingmoe`, `bailingmoe2` and `llada-moe` share the new one, and `cohere2moe` and `tiny_aya` share another); a file naming another one tokenizes as a single chunk, and `orbix model inspect` says so rather than leaving it to be discovered as poor output. There are no unimplemented names left -- checked against llama.cpp's pre-type assignments, every name resolves to a rule that agrees with the reference on the chunk boundaries. Each family's chunk boundaries are pinned by tests built from its llama.cpp expression, and the rules were diffed against those expressions as sequential passes over 80,128 strings -- every disagreement found that way is fixed, and the 126 strings that exposed them are kept in the package as a regression corpus. The corpus is the reproducible part: it is a Go file the tests walk, and its length can be counted. The 80,128 is the size of a one-off sweep whose generator was not kept, so it records what that run covered rather than offering a number a reader can recompute. `MODEL`-gated tests round-trip every Unicode code point and a set of convention-heavy strings through a real checkpoint. |
| Chat templates | `internal/chat`     | Evaluates the subset of Jinja2 a checkpoint's `tokenizer.chat_template` needs -- conditionals, loops, set, concatenation, and the trim/tojson filters -- so a conversation is rendered in the format the model was trained on. Syntax outside the subset is reported at parse time; a call to a function this package does not implement is reported when the template is rendered, since the parser accepts any call. Either way the caller falls back to a neutral transcript rather than emitting a half-rendered prompt.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Engine         | `internal/engine`   | Batches a scheduling round: every pending prompt prefills in one pass, then every active sequence decodes together in one pass, so concurrency raises total throughput. `MaxActive` bounds a round and an admission policy (`fifo`, `shortest`, `fair`) picks which sequences fill it. Each sequence keeps its own logits copy and KV cache; the cache is a contiguous arena or paged blocks from a shared pool, with optional prefix reuse. `RenderChatPrompt` is the one chat template the REPL and the HTTP server share.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Transformer    | `internal/llama`    | Llama and Qwen2 forward passes: adjacent-pair (Llama) and NEOX (Qwen2) RoPE, optional attention bias, grouped-query attention, and two KV cache layouts -- a contiguous arena per sequence and a reference-counted paged cache over a shared block pool. `PrefillWithPrefix` reuses blocks whose token prefix has been seen before. `Prefill` and `DecodeStep` are bit-identical to the per-token path on either layout. Greedy decoding on a real checkpoint agrees with llama.cpp's log-probabilities to within 0.12 nats, which is close enough that the continuation matches on the prompts tried -- and not a guarantee that it matches on every one, since a greedy choice is only as stable as the gap between the top two candidates.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Tracing        | `internal/otel`     | Exports spans in OpenTelemetry's OTLP/HTTP+JSON format without the SDK: the message it replaces is a handful of structs, which is cheaper to read here than to take as a dependency. Ids are lowercase hex, which is the one place OTLP/JSON overrides proto3's mapping for a bytes field, and timestamps are nanosecond strings, and export is best-effort: a collector that is down drops spans and counts them rather than slowing a request, and a collector that answers `200` while rejecting spans in `partialSuccess` has that rejection counted too, because a 2xx is not by itself evidence that anything was kept.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Server         | `internal/server`   | `orbix serve`: OpenAI-compatible `/v1/chat/completions` and `/v1/completions` with SSE streaming, `/v1/models`, `/healthz`, and Prometheus-format `/metrics`. Streaming text is decoded incrementally so multi-byte characters split across tokens still render; concatenated stream deltas equal the non-streaming body exactly. A disconnected client cancels its sequence through the engine. Both endpoints accept an Orbix extension, `priority`, which ranks a request against others waiting for a slot: higher advances first and the admission policy orders requests of equal priority. Priority preempts rather than only queue-jumping -- the ready set is re-ranked every round, so a running request yields its slot when higher-priority ones are waiting and resumes afterwards. On one slot a 120-token request takes about 0.55 s alone and 0.76 s when twelve short high-priority requests arrive during it, 37-39% longer over three runs on stories15M q4_0. It is omitted from OpenAI's schema, so a client that never sends it is unaffected.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

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

Every error body carries a `code`, including the ones a client hits before any field is
parsed: `invalid_json`, `unsupported_encoding` and `body_too_large` for a body the decoder
rejected, and `invalid_value`, `missing_required_parameter`, `unsupported_parameter`,
`context_length_exceeded` and `model_not_found` for a request it understood and refused.
OpenAI's guidance is to branch on `code` rather than on message text, and a client following
it could previously tell its prompt was too long from its temperature being negative, but not
either of those from a malformed body -- those three arrived with no code at all. `param` is
present when one field is at fault and absent when the failure is about the body as a whole,
and `type` separates a request error (`invalid_request_error`) from the server's own failure
(`server_error`) and from capacity (`service_unavailable_error`, which is the one the OpenAI
SDKs retry).

Unsupported request fields are handled by what a client does with the answer. A field that
only tunes sampling -- `logit_bias`, `presence_penalty`, `response_format` -- is ignored,
because the SDKs send such fields whether or not the engine can use them and refusing them
would make the server unusable from that SDK; the text is not what was asked for, but it is
the shape the client reads. A field that describes the *response* is refused with a 400 that
names it and `code: unsupported_parameter`: `n`, `logprobs`, `top_logprobs`, `echo`, `suffix`,
and `best_of`. A client that sent `n: 3` indexes `choices[2]`, and one that sent `logprobs`
reads `choice.logprobs`, so answering with one choice and no logprobs is not a smaller
response but a different one -- the failure would surface as an `IndexError` or `KeyError` in
the caller, far from the request that caused it. Several at once are all named, so one round
trip fixes the request.

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

- **Keep the connection open.** nginx's `proxy_read_timeout` defaults to 60 seconds and bounds
  the gap between reads, not the whole response, so a long model load or a stalled generation
  can exceed it. Raise it for the inference location.
- **Keep reading.** A stream nobody reads is abandoned, not paused, once the backpressure
  reaches the engine. The engine buffers 64 events per sequence and, when that fills, waits 2
  seconds for it to drain before retiring the sequence; its output is then discarded and its
  slot goes to someone else. A client that opens the stream, does something slow, and comes
  back finds a closed connection rather than the tokens it was owed.
  How soon this bites depends on a size the server does not control: kernel socket buffers
  absorb a surprising amount, so a short response finishes into them even with nobody reading
  (measured: 120 tokens, about 20 KB, completed into the buffer with the reader stopped for
  five seconds). The bound is on the *engine's* queue, not on the network, so a long generation
  or a small receive window reaches it and a short one may not. The guarantee is therefore
  "drain as you go", which every SSE client does by default and a hand-rolled one may not --
  not "you have 2 seconds".
- **The upstream connection has to be HTTP/1.1 for chunked transfer.** nginx proxies HTTP/1.1
  by default from 1.29.7; before that the default was 1.0, where ending a stream means closing
  the connection. Set `proxy_http_version 1.1` with an empty `Connection` header on older
  versions, or where a higher-level block set something else.

`X-Accel-Buffering` is nginx-specific, and nginx consumes it rather than forwarding it: it is
in nginx's default hidden-header set, so it never reaches the client. A two-tier nginx chain
therefore needs `proxy_pass_header X-Accel-Buffering` on the inner tier, or the outer tier
never sees it.

The other proxies do not need it, for different reasons. HAProxy and Envoy forward response
bytes as they arrive and have no whole-message buffering to turn off. Caddy buffers partially
by default, but its documentation lists `Content-Type: text/event-stream` among the cases where
it flushes immediately regardless. All three still need timeouts sized for a response that can
outlive the default -- which is the part no application header can fix.

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
running out of it is the one failure a session hits repeatedly: a conversation outgrows the
pool, every later turn fails the same way since a failed turn does not grow the history, and
the error names the pool's size without saying how close the session is to it. The total is
there because the free count alone misleads -- a session refused on every turn may show all
blocks free, which means the request is larger than the pool can ever hold rather than that
anything is contended. That is the case `/clear` cannot fix, and seeing the size is what tells
the two apart.

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

`orbix model inspect` is also a learning tool: it prints the GGUF layout in a
readable form, which is useful when studying how a model file is structured.

`--json` keeps each value the type the file declares, with one exception that the format
forces: a float that is NaN or infinite cannot be a JSON number, because JSON's grammar has no
token for either, so it is emitted as the string `"NaN"`, `"+Inf"` or `"-Inf"`. Such a value is
legal in GGUF -- a float is any 64-bit pattern -- and a checkpoint can carry one as a sentinel
or from a broken conversion. The alternative was letting one such key fail the whole document:
before this, `model inspect --json` on a file with a NaN printed `json: unsupported value: NaN`
and nothing else, so none of the file's other keys could be read.

`orbix bench --suite` runs a fixed set of scenarios and prints them as one table. A
single throughput number answers one question about one machine; the suite is what shows
which mechanisms pay for themselves. It pairs each mechanism with a control row -- paged
against contiguous caches, prefix reuse against none, shortest against FIFO -- so a
difference is attributable to that mechanism rather than to the scenario changing.

How much prefix reuse is worth depends on the mix, and the two rows above are read under the
conditions stated with the table: 41 prompt tokens against 8 generated per request, where the
prompt is most of the work and the pair differs by 38%. Raise `--max-tokens` and the shared
saving is diluted by generation, which reuse does not touch: at `--max-tokens 32` the same pair
measures 205.4 against 218.2 tok/s, about 6% instead of 38%, and the difference between those two
readings is the token budget rather than the mechanism. A long prompt with a short answer
(retrieval, classification, question answering) is where this pays; a short prompt with a long
generation is where it does not.

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

Those are medians of nine interleaved runs, not one run: on this machine a single run
moves by more than the difference between neighbouring rows, so one table taken once
cannot separate them. The `NOTE` column in a live run is longer than the text above --
it also reports the block pool's shape and the positions the prefix cache reused, which
are trimmed here to fit.

The `FIRST` column is a latency, but of a different request depending on the row, and reading it
without knowing which is the faster way to be wrong about this table. On a throughput row it is
how long until *any* request produced a token, which is essentially the prefill time. On the two
policy rows it is the **short** request's own first token -- not "any", because under MaxActive 1
the long request runs first and would supply that number under every policy, leaving the column
with nothing to say. The policies reorder a fixed amount of work, so where the short request lands
is the only thing they change: 41 ms under `shortest` against 401 ms under `fifo` is the whole
result, and the two rows' `TOK/S` figures are not.

Three of those rows need a sentence of explanation, because their numbers look flat or
backwards for a reason:

- **Paged vs contiguous** is a memory trade, not a throughput one. The paged cache serves
  the same batch at the same speed while letting 16 sequences share one pool instead of
  each reserving a whole context. The suite prints the pool shape so the trade is visible
  rather than implied.
- **The policy rows do the same amount of work, and they generate more than `--max-tokens`
  asks for.** A policy reorders a fixed quantity of work; it cannot reduce it, which is why
  the two rows' `ELAPSED` figures are close here. The 63 tokens in both rows are a 60-token
  long request plus a 3-token short one: the rows raise anything below 60, because two
  requests that both finish in three tokens measure the scheduler's startup rather than its
  ordering, and the note in a live run says so when it happens.
  Treat that as the design and not as a measurement of the scheduler, though: over six
  interleaved runs of this pair the `TOK/S` ratio ranged from 0.92 to 2.78, and `FIRST`
  ranged from 18 to 66 ms under shortest against 314 to 961 ms under FIFO -- all of it
  machine load, none of it the policy. The number that is a property of the scheduler is
  the round in the `NOTE` column, and it was 2 under shortest and 63 under FIFO in every
  one of those runs. That is why the engine counts rounds and the suite reports them.
- **The prefix rows run their four requests one after another, not concurrently.** A
  concurrent burst with `MaxActive` as large as the request count prefills all four
  together, so none arrives after another has cached its prefix and the reuse path is
  never taken. Sequence is what makes the pair a comparison at all; the row reports the
  positions reused so a reader can see it happened rather than infer it from throughput.
  Both rows also repeat the prompt until it spans two cache blocks, because a block is
  published to the prefix index only once its last position is written -- with the
  command's default five-token prompt there is never a whole block to reuse, and the two
  rows differ by nothing.

A row can also show fewer tokens than its budget. That is not a measurement failure: the
model emitted its end-of-sequence token, and the suite says so in the `NOTE` column
("N request(s) stopped early on EOS") rather than leaving you to work out why two rows
that should match do not.

A single measurement (without `--suite`) reports the same kind of shortfall as a `note:`
line, for the case the suite cannot hit: a prompt long enough that the model's window leaves
less room than `--max-tokens` asks for. The header states the budget and the total is smaller,
so the note names both — `the 4096-position window cut the budget: 6 of 100 requested tokens`.
Without it the report was ambiguous in the worst way for a measurement: `finish: length` reads
as "the budget was reached" whether the budget or the window ended the run.

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
same reason the GGUF parser is: a family is three lines -- `# HELP`, `# TYPE`, and one
sample -- and the hand-written encoder is read by the code review rather than trusted as
a dependency. Sixteen families come to 52 lines on a paged server, because a labelled
series costs one line per value. The tests parse the output back the way
a scraper would, so a HELP line in the wrong place fails a test instead of a dashboard.

The count depends on the configuration, which is worth knowing before comparing two
scrapes: a contiguous server exports twelve families (the four `kv_pool_*` gauges are
omitted without a pool, since a zero would read as "the pool is empty" rather than "there is
no pool"), and `--otlp-endpoint` adds one more. The families are only omitted, never
exported as zero, for that reason.

Three of the metrics exist because their failure mode is invisible in latency until it is
already failing. `orbix_requests_queued` growing against a steady round rate is a
capacity problem rather than a slow model, and `orbix_kv_pool_free` reaching zero is a
paged deployment about to start refusing requests. Neither is recoverable from timing
alone.

The third is `orbix_requests_finished_by_reason_total`, which splits the finished count by
why each request ended. A client that disconnects mid-generation produces a request that
looks entirely healthy in every other number: it counts as finished, it contributes an
ordinary latency, and it consumes exactly the tokens it was given. Measured against a live
server, an aborted stream and a completed one produced access-log lines differing only in
byte count. With the split, a server whose clients keep leaving shows it as
`reason="cancel"` or `reason="abandoned"` growing against `reason="stop"`, which is a
network or timeout problem rather than a model problem. The parts always sum to
`orbix_requests_finished_total`; the tests assert that, so a reason missed in the engine
breaks a test instead of quietly unbalancing a dashboard.

`orbix_requests_refused_queue_full_total` and `orbix_requests_refused_pool_total` count the
requests answered `503`. They are the server's own counters rather than the engine's, because
a refused request never enters the engine -- counting it as submitted would make the admission
rate look better than it is, and that is deliberate. The consequence was that refusals
appeared nowhere at all: measured with `--max-active 1 --max-queued 1` and 24 requests in
bursts of eight, 16 were refused with `503` while `orbix_requests_submitted_total` read 8 and
the queue gauge sat at 0, so a server shedding two thirds of its load reported nothing wrong.
Two families rather than one, because the fixes differ: a full backlog wants a deeper
`--max-queued` or more replicas, while an exhausted pool wants more `--blocks` or shorter
prompts.

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

The gates take the suite from fifteen skipped tests to none, so a machine holding the
checkpoints can check every claim the project makes. Both numbers are observable in the run
itself -- `go test ./... -v | grep -c '^--- SKIP'` reads 15 with no checkpoint on disk and 0 with
all four set -- which is why they are named here instead of a test total: the count of tests
changes with every commit and nothing can pin it, while the skip count is a property of the
gating. That does not make it immutable: adding a gated test raises the first number, removing a
gate that was never needed lowers it, and this sentence has now been wrong three times for
exactly those reasons. Re-measure it rather than trusting it; the last correction was two gates
that skipped without needing a checkpoint at all, because the tests drove the fixture.

```bash
MODEL=/tmp/stories15M-q4_0.gguf \
QWEN2_MODEL=/tmp/qwen2-f32.gguf \
MIXTRAL_MODEL=/tmp/tinymixtral.gguf \
MODEL_KQUANT=/tmp/tinymixtral.gguf \
CGO_ENABLED=0 go test ./...
```

`just test` does not set these variables. It runs `go test -count=1 ./...` with the justfile's
`CGO_ENABLED=0` export and nothing more, so the checkpoint-gated tests skip under it exactly as
they do without it. The block is written out here because a copyable command is what a reader
holding the checkpoints needs; putting the paths in the recipe would mean every contributor's
`just test` ran a different suite.

Worth knowing before judging that run: a file whose weights were never trained produces
a flat next-token distribution -- one measured at p(argmax) = 0.0003 against 0.26 for a
trained checkpoint -- so its output is gibberish whatever the implementation does. The
test therefore asserts that zeroing the experts changes the logits, which is a statement
about the code, rather than that the text reads well, which would be a statement about
the file.

## Performance Expectations

Orbix is a learning project, and it is honest about where it stands:

- **CPU-only.** No GPU support. For GPU throughput, use vLLM, SGLang, or TensorRT-LLM.

- **Prefill throughput improves with prompt length, within the window.** On Llama 3.2
  1B Q2_K with 32 workers, medians of five interleaved runs gave 3.96x the tokens for
  3.04x the time (155 tokens in 8.1 s, 614 in 24.5 s), so the per-token cost fell from
  52 ms to 40 ms as the prompt lengthened. With 128 workers the same interleave gave
  2.76x, so the effect is larger when there are fewer workers to go round. That is the
  direction the arithmetic predicts:
  attention is O(n^2) in total but parallelises over positions, one goroutine per
  position scoring against the cache, while the projections are O(n) and are spread
  over output columns that do not grow with the prompt. So the growing term is the
  one that gets more workers as it grows. [INFERENCE] the rest of the explanation --
  that a longer prompt also amortises a round's fixed cost -- is arithmetic rather
  than measurement; what was measured is the per-token cost falling. A prompt long
  enough to make attention dominate per-position work would stop the trend, and this
  engine has no measurement of where that turn is.

  Interleaving the two sizes matters more than the sizes themselves: taken in separate
  runs, load drift put a 2.5x spread on the smaller one, wider than the effect being
  measured.

  This bullet used to say prefill was *superlinear* and give "4x the tokens for 2.78x
  the time" as evidence, which says the opposite. That figure was also measured before
  attention was parallelised, when 2001 tokens took 177 s and attention was 135 s of
  it. Both halves are replaced rather than deleted, because the shape of the mistake
  -- an adjective that outlived the measurement under it -- is the reusable part.

- **Treat the numbers above as indicative, not as a benchmark.** They were taken on
  a shared machine whose load drifted between 40 and 90 over the course of a few
  minutes, and repeat runs of the same workload varied by more than the improvements
  being measured -- a 3721-token prompt once timed faster than a 3641-token one.
  The direction of each change is sound and the bit-identity tests are unaffected,
  but a timing claim from this machine needs a quiet one to confirm. `orbix bench`
  prints its own conditions so a reader can judge a table taken elsewhere.

- **Reference kernels are not optimized per instruction.** They are written to be
  readable, with no SIMD and no assembly. What they do use is more than one core: the
  partitionable products run on a bounded worker set, which took decoding on
  stories15M from 65 to 235 tokens per second on a 128-core EPYC.

- **SIMD and assembly are deliberately out of scope.** This project exists to make
  inference legible in one language, and a kernel that cannot be read in the
  repository's own language defeats that. Parallelism over independent outputs is where
  the speed comes from instead, and it is what the numbers above measure.

- **A model runs in its trained context, and no larger.** The rope table is built for
  that window, so `--ctx` accepts a smaller value and refuses a larger one with a
  message naming both numbers. Going past it needs rope scaling -- the frequency
  adjustment a long-context fine-tune applies -- which this engine does not implement.
  A generation budget larger than the window is cut to fit, and `--stats` says when
  that happened, because a run that asked for 128 tokens and printed 123 otherwise
  looks like the model stopped early.

- **Parallelism is over independent outputs, never over a reduction.** The split decides
  which goroutine computes which output element -- rows for a matvec, columns for a
  transposed product -- and never splits the dot product itself, so every element is summed
  in the same order the serial kernel uses. float32 addition is therefore never reassociated
  and the parallel result is bit-identical. That is a deliberate constraint: a split reduction
  would be faster on paper and would silently change every number the project compares against
  llama.cpp.

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

`--block-size` has one hard limit worth stating next to the flag: it cannot exceed the model's
context. Prefix reuse stores and matches whole blocks, so a block the context can never fill is
never published to the index, and `--prefix-cache` would reuse nothing while still reporting
itself on. `serve`, `chat`, and `bench` all refuse that combination rather than start; the
default of 16 is safe for any model with a context of 16 or more.

`--max-tokens` is the one flag whose accepted range differs by command, and the difference is
not arbitrary. `run` accepts 0, which prefills and prints nothing -- one command, and the
absence of output is the result. `serve`, `chat`, and `bench` refuse 0: the server cannot
express a request for zero tokens at all (a request's `max_tokens` is an integer, so 0 is
indistinguishable from the field being absent and takes the server's default), a REPL turn that
generates nothing answers with an empty line, and a benchmark of zero generated tokens is not a
measurement. None of them accepts a negative value, which every other numeric flag on these
commands refuses too.

A config file is on the roadmap, not in the code: the interface block at the top of
this file describes the target, and this table describes what exists.

`--policy` is the flag most likely to need a reason rather than a definition, so: `fifo` is the
default because submission order is the easiest thing to reason about and every other policy is a
visible departure from it. `shortest` serves the sequences with the least remaining work first,
which is what keeps a queue of mixed-size requests from being dominated by its longest member --
it reorders without bounding anything. `fair` serves whichever sequence has gone longest without a
turn, and that is the one with a bound to quote: no sequence waits more than about
`len(active)/MaxActive` rounds in a row, which makes it the choice for a long-running server where
starving a request is the failure to avoid. `fair` is not the default for the same reason `fifo`
is: a policy that bounds wait time is only useful if the operator asked for it.

What those three do to the same queue, measured on the fixture with `MaxActive 1` and four
requests whose budgets are 16, 8, 4 and 2 in submission order, as the round each got its
first token:

| Policy     | First-token rounds | What it did                                                          |
| ---------- | ------------------ | -------------------------------------------------------------------- |
| `fifo`     | 2, 19, 28, 33      | served in submission order, so each waited its turn                  |
| `shortest` | 19, 10, 5, 2       | exactly inverted: the 2-token request moved from round 33 to round 2 |
| `fair`     | 5, 6, 7, 8         | one turn apiece first, then round-robin from there                   |

`shortest` is the one that changes the order most, and the inversion is the point: it looks
purely at remaining work, so a request submitted last is served first when it is the smallest.
`fair` does not reorder past the first turn -- it hands everyone a turn before repeating, which
is why its first-token rounds are consecutive where `fifo`'s span the whole queue. Under `fair`
each sequence then returns every `len(active)/MaxActive` rounds: with four sequences on one slot
the measured round gap was exactly 4.

The two cache layouts differ in what they reserve, and the difference is large enough to
decide the flag. A contiguous cache reserves the whole context for every accepted request
when it arrives, so the cost is `context x KV width x layers`, multiplied by the active
and queued slots. A paged cache takes blocks as a sequence generates, so memory follows
what is actually in flight. The server prints the figure it computed at startup, and warns
when a contiguous reservation passes a gibibyte.

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

`--workers` divides one batched pass across goroutines. What bounds its usefulness is the
number of independent outputs in a pass, not the number of sequences: a matvec splits by
output row, so a batch has one row per sequence, but the transposed product behind the output
projection and the attention value/output projections splits by output *column*, and a decode
step has thousands of those whatever the batch size. Measured on a 128-core host with eight
concurrent requests, medians of seven interleaved runs gave 349.6 tokens per second at 8
workers, 536.6 at 32, and 543.9 at 128. The count is therefore a real knob below 32 and flat
above it -- 32 and 128 differ by about 1%, which is inside the run-to-run spread, while 8 is
half again slower. The default of one per CPU sits at the flat end, which is the safe place
for it.

This measurement was 324.9 and 327.6 (8 against 128, "the same within noise") before the
transposed matmul moved to a column split. The old figure was right for the old split: a
row-partitioned product could not use more workers than the batch had rows. The paragraph is
here rather than deleted because the way it went stale is worth knowing -- the split axis
decides which bound applies, and changing one silently invalidated the measurement of the
other.

The same knob on one stream instead of eight: stories15M q4_0 (19 MB, 6 layers, 288 hidden),
128 tokens from one prompt, five interleaved runs. One worker gave 64.2-65.1 tok/s and 32 gave
228.7-236.2, a ratio of 3.52x to 3.67x. The `run` example in the CLI section used to say 3.9x,
which is what the synthetic fixture produces -- its projection is large next to the per-round
overhead -- and a 19 MB model does not reach that.

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

`just check` is what a contributor runs before pushing; it needs only the Go toolchain and
golangci-lint **v2**, because `.golangci.yml` is a v2 configuration and a v1 binary refuses to
read one. The recipe checks the version and says so rather than letting the linter's own
message about the config stand in for it -- that message names neither the toolchain nor the
fix, and a mismatch turns the gate red on an otherwise clean tree.

`just docs` runs `rumdl check .` over the repository's Markdown -- which here is
the README and the pull-request template, since `.rumdl.toml` excludes `AGENTS.md` -- and needs
rumdl. `just hooks` runs the pre-commit hooks through prek. Both are separate targets rather
than part of `check` for the same reason: neither tool is in the Go toolchain, and `check`
should stay runnable with Go and golangci-lint alone. The three are the gates CI runs, and
running them makes a green branch predictable rather than hoped for.

The two differ in what they do when they find a problem, which is worth knowing before running
either: `just docs` reports and leaves the file alone, while `just hooks` **rewrites** it --
its `rumdl` and `rumdl-fmt` hooks pass `--fix`, and `end-of-file-fixer` and
`mixed-line-ending` edit in place too. A hook that changed something exits non-zero and says
`files were modified by this hook`; re-stage the file and look at what it did rather than
assuming the edit was cosmetic, since the Markdown fixer also reflows list spacing and table
alignment.

The `doc.go` files are Go, so rumdl does not see them; their prose is checked only by reading
it. That is why every package's design notes live in a `doc.go` that a reviewer sees in the
diff rather than behind a linter -- a broken Markdown table in the README fails `just docs`,
and a contradictory sentence in a package doc does not fail anything.

The split is not cosmetic. `just check` sees Go files only: a whitespace-only line appended to a
file under `.github/workflows/` leaves it green, exit 0, which is what `just hooks` catches.

Run `prek install` once to make the hooks fire on every commit. `just hooks` is the same checks on
demand and is what CI runs; installing is what makes them run without being asked. The hooks are
in `.pre-commit-config.yaml`, and nothing about them is copied into the repository when installed --
`prek install` writes shims under `.git/hooks/`, which git does not track.

`just fuzz` searches rather than gates, which is why it is separate too. Six targets cover the
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
- Naming: brand **Orbix**, binary **`orbix`**, module path **`github.com/pplmx/orbix`**. There is no importable root package -- the layers are under `internal/` -- so a package named `orbix` would be a new decision rather than a rename.

## License

MIT © 2026 pplmx
