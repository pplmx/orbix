# Architecture

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

## Core interfaces

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

The block above is a target, not a description:

| Interface        | State                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------- |
| `Engine`         | `engine.Engine`, taking a model path rather than a `LoadOptions` struct                      |
| `Scheduler`      | `Engine.Submit` + `Pump` + `Stats`; the caller drives the round, not a loop                  |
| `KVCacheManager` | Not separate: `llama.BlockPool` allocates, `llama.PrefixIndex` matches prefixes              |
| `Backend`        | Not implemented by decision: one compute path exists, and a remote one could not do this job |

The kernels are real and exported. Their signatures today:

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

Decoding a quantized tensor is not a kernel -- it lives in `internal/quant`, so a Q4_0 matrix
arrives already widened. Attention is not one either: it is assembled in `internal/llama` from
`Softmax`, `RMSNorm`, `AddBias` and `AddInPlace`, because the mask and cache layout are model
concerns rather than arithmetic ones.

## Where to read next

Each package has a `doc.go` explaining what the layer is responsible for. Read in this order and
the concepts arrive in the order a request travels through them.

Each package has a `doc.go` explaining what the layer is responsible for; read in this order and
the concepts arrive in the order a request travels through them.

| #   | Package                                                                                | Concept                                          |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 1   | [`internal/gguf`](../internal/gguf/doc.go)                                             | Model file layout; tensors by offset, not loaded |
| 2   | [`internal/f16`](../internal/f16/doc.go), [`internal/quant`](../internal/quant/doc.go) | Half precision and block quantization            |
| 3   | [`internal/kernel`](../internal/kernel/doc.go)                                         | Matmul, RMSNorm, softmax, SiLU                   |
| 4   | [`internal/tokenize`](../internal/tokenize/doc.go)                                     | Text to ids: byte-level and SentencePiece        |
| 5   | [`internal/llama`](../internal/llama/doc.go)                                           | Attention, RoPE, KV cache, paged blocks, MoE     |
| 6   | [`internal/engine`](../internal/engine/doc.go)                                         | Continuous batching, admission, a round          |
| 7   | [`internal/chat`](../internal/chat/doc.go)                                             | The Jinja2 subset a template needs               |
| 8   | [`internal/server`](../internal/server/doc.go)                                         | OpenAI API, SSE, `/metrics`, access log          |

Two layers are off that path because nothing in the forward pass goes through them:
[`internal/otel`](../internal/otel/doc.go) exports traces, and
[`internal/gguftest`](../internal/gguftest/doc.go) builds the test fixtures.
