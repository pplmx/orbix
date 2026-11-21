# Configuration

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

`--block-size` above the model's context is refused when reuse is on: prefix reuse matches whole
blocks, so a block the context can never fill is never published and `--prefix-cache` would reuse
nothing. `serve` and `bench` refuse it when `--prefix-cache` is given, and `chat` refuses it
always, because chat enables reuse by default rather than on request.

With reuse off the combination is accepted, and deliberately so: a block larger than the context
is then just a pool with one block in practice, which serves text correctly. Measured on a
64-position fixture with `--blocks 8 --block-size 128`: `serve` and `bench` start, while the same
invocation with `--prefix-cache` answers "use --block-size 64 or less".

`--prefix-cache` holds blocks, not just references to them. Every block the index remembers is
retained until its entry is evicted, and there are two eviction triggers, not one:

- when the index reaches its own bound of 1024 entries, it gives up the end of a chain;
- when the pool runs out of free blocks, `BlockPool.acquire` evicts remembered prefixes before
  reporting exhaustion, because a cached prefix is a speculation about the future while an
  allocation is not.

The second is what produces the floor measured below: a pool smaller than the index's bound
fills with remembered prefixes, and then every new allocation trades one of them away.

Measured on a 48-block pool with distinct prompts: 46 blocks free at the start, 26 after 100
requests, and **1 after 200, where it stays**. Adding the pool's whole size back as `--blocks`
moves that floor up but not away, and a prompt too short to fill a block contributes nothing to
it -- shares are only remembered once a block is full.

The practical reading: `--blocks` for a `--prefix-cache` deployment is the working set for
in-flight requests *plus* whatever the index retains, so sizing it for concurrency alone leaves
the floor near a handful of blocks. Requests still complete at that floor because eviction frees
what they need -- measured, a 202-position prompt with a 32-token budget still answered `200`
against a pool showing one free block -- so the failure is exhaustion on a request larger than
the pool can ever hold, which is refused with a 400 naming both counts.

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

For scale: the 128k-window Llama 3.2 checkpoint reserves 8 GiB per request that way
(131072 x 8 x 64 x 2 x 4 x 16 = 8589934592 bytes), against 32 KiB for the synthetic fixtures.
Any model with a long window should be run with `--blocks`.
