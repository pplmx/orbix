# Configuration

Every setting is a flag today; there is no config file. `ORBIX_MODEL` is the one
environment variable, so a shell can point every command at the same checkpoint.

```bash
export ORBIX_MODEL=./models/llama-3-8b-q4_0.gguf
orbix serve --max-active 32 --policy shortest --blocks 2048 --block-size 16 --prefix-cache
```

Flags shared by every command that generates:

| Flag                       | What it controls                                                    |
| -------------------------- | ------------------------------------------------------------------- |
| `--model`                  | Checkpoint path; falls back to `ORBIX_MODEL`                        |
| `--max-tokens`             | Cap on generated tokens, not counting the prompt                    |
| `--workers`                | Kernel goroutines per batched pass; 0 means one per CPU             |

Sampling, on `run` and `chat` only (the server takes these per request):

| Flag      | What it controls                                                       |
| --------- | ---------------------------------------------------------------------- |
| `--temp`  | Sampling temperature; 0 means greedy, which is the default             |
| `--top-k` | Top-k cutoff; 0 disables it. Applies only with `--temp` > 0            |
| `--top-p` | Nucleus cutoff in (0,1]; 0 disables it. Applies only with `--temp` > 0 |

Paged cache, on `serve`, `chat` and `bench`:

| Flag                       | What it controls                                                    |
| -------------------------- | ------------------------------------------------------------------- |
| `--blocks`, `--block-size` | Paged KV cache; without them each request reserves a whole context  |
| `--prefix-cache`           | Reuse KV blocks across requests sharing a prompt prefix             |
| `--no-prefix-cache`        | `chat` only: turn reuse off, which is otherwise on for a REPL       |

`serve` and `bench`:

| Flag                       | What it controls                                                    |
| -------------------------- | ------------------------------------------------------------------- |
| `--max-active`             | Sequences advanced per round, and therefore the batch size          |
| `--max-queued`             | Backlog accepted before new requests get 503; 0 means 8x max-active |
| `--policy`                 | Which sequences fill a round: `fifo`, `shortest`, or `fair`         |
| `--addr`                   | Listen address, `:8080` by default                                  |
| `--log-format`             | Access log to stderr: `text`, `json`, or `off`                      |
| `--otlp-endpoint`          | OTLP/HTTP collector for traces; unset means no tracing              |
| `--shutdown-grace`         | SIGTERM/SIGINT drain time for in-flight replies; 5s, 0 means none   |

`run` and `chat` conversation options:

| Flag       | What it controls                                                     |
| ---------- | -------------------------------------------------------------------- |
| `--ctx`    | Context length; 0 uses the model's trained window, larger is refused |
| `--seed`   | `run` only: sampling seed; no effect without `--temp`                |
| `--stats`  | `run` only: timings and token counts to stderr                       |
| `--echo`   | `run` only: print the prompt before the generated text               |
| `--system` | `chat` only: system prompt for the session                           |

`bench` workload:

| Flag                       | What it controls                                                    |
| -------------------------- | ------------------------------------------------------------------- |
| `--concurrency`            | Requests in flight at once                                          |
| `--requests`               | Requests per measurement                                            |
| `--prompt`                 | Prompt text for the measurement                                     |
| `--suite`                  | Run every scenario as one table; fixes the flags it would ignore    |

Every one of these is registered once in `cmd/orbix/flags.go` when several commands share it, so
the wording is the same wherever it appears. `orbix <command> -h` prints the same list with the
per-command defaults.

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

A config file is a decision this project has not made, not a gap waiting to be filled. The
roadmap in the performance page is a reading order and is complete; this sentence used to
say a config file was "on the roadmap", which pointed at a list that never contained one.
The reason there is no such file is recorded in `AGENTS.md`'s dependency table: one
environment variable and zero configuration sources is why viper and Koanf were not adopted,
and a config file would give those decisions a different answer rather than merely adding a
feature.

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
The server prints the figure it computed at startup and warns past a gibibyte. It prints a
`weights:` line beside it for the part that exists before any request arrives -- see
[the weights section](#the-weights-are-held-as-float32-which-is-the-memory-figure-to-plan-with),
which is usually the larger of the two.

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

Above 32 the plateau is flat measured four ways, not two: 64 greedy tokens from one prompt on the
same checkpoint took 245-256 ms at 32 workers, 240-255 ms at 64, 234-247 ms at 96 and 243-253 ms
at 128 over two runs each -- overlapping ranges, with no trend in the worker count. Raising the
default would buy nothing on a wide host, which is why it stays at one goroutine per CPU.

For scale: the 128k-window Llama 3.2 checkpoint reserves 8 GiB per request that way
(131072 x 8 x 64 x 2 x 4 x 16 = 8589934592 bytes), against 32 KiB for the synthetic fixtures.
Any model with a long window should be run with `--blocks`.

## The weights are held as float32, which is the memory figure to plan with

The KV cache is not the only resident cost, and on a small quantized model it is not the
largest one. `llama.LoadWeights` widens every tensor into a `[]float32` once at load, so the
forward pass works in plain floats and never decodes a block again. The consequence is that
resident memory follows the model's **parameter count**, not its file size: 4 bytes per
parameter, whatever the file was compressed to.

Measured peak RSS, one request, `orbix run --max-tokens 1`:

| Checkpoint                  | File    | Parameters | Peak RSS | File x |
| --------------------------- | ------- | ---------- | -------- | ------ |
| stories15M Q4_0             | 18 MiB  | 24.4 M     | 136 MiB  | 7.6    |
| Llama 3.2 1B Q2_K/Q3_K/Q4_K | 553 MiB | 1.24 B     | 5.3 GiB  | 9.6    |
| TinyLlama 1.1B Q2_K         | 461 MiB | 1.10 B     | 4.3 GiB  | 9.5    |
| Qwen2.5-0.5B Q8_0           | 644 MiB | 630 M      | 3.0 GiB  | 4.8    |

The exact resident figure is four bytes times the parameter count, which is the column to
read; the file-size column is there only to show how little it predicts. Do not try to derive
one from the other by a single ratio. Radically compressed files grow more (TinyLlama is 2.6
bits per weight and expands 9.5x) and lightly compressed ones less (Qwen2.5 Q8_0 is 8.5 bits
and expands 4.8x), but neither is `32 / bits_per_weight`, because every file also carries
tensors that were never quantized -- the per-layer norms and often the output projection are
F32 already and do not grow at all. Measured against that formula the four rows above come out
at 7.1x, 12.3x and 3.8x where 7.6x, 9.5x and 4.8x were observed.

A 7B Q4_0 checkpoint is roughly 3.7 GiB on disk and needs about 26 GiB resident, which is more
than most hosts have -- that is a property of this design, not a misconfiguration, and it is
why the project runs small checkpoints.

A mixture-of-experts file is worth calling out separately, because its file size is the least
predictive of all: every expert is widened and held even though only `expert_used_count` of them
run per token. Measured on TitanML's 101 MiB Tiny Mixtral test file, which has 8 experts at 2
per token, the startup banner reports `weights: 942.1 MiB` -- nine times the file, and all of it
resident for a model that computes with a quarter of it.

Planning a deployment means adding the two costs: the widened weights above, plus whatever
`--blocks` sets aside for KV. On Llama 3.2 1B the weights alone are 4.6 GiB
(1.24 G parameters x 4 bytes) before any cache exists, so the 8 GiB-per-request contiguous
figure further up is on top of that, not instead of it. The measured 5.3 GiB peak for that
model is those weights plus however many KV pages a single short request happened to touch --
the reservation is virtual, so a longer prompt makes the same server larger.

Keeping the model in its quantized form and decoding per matmul would remove the expansion at
the cost of speed, since each weight would be decoded once per use rather than once per load.
That trade is not taken here: a decoder in the inner loop is a different engine, and this one
is written to be read.

## Tracing covers requests that reach the engine

`--otlp-endpoint` posts OTLP/HTTP+JSON to a collector. What arrives is one span per request, and
the attributes are the OpenTelemetry `gen_ai` ones plus two of this engine's own:

| Attribute                    | Meaning                                        |
| ---------------------------- | ---------------------------------------------- |
| `gen_ai.request.model`       | The name the request asked for                 |
| `gen_ai.response.model`      | The name this server answers as                |
| `gen_ai.request.max_tokens`  | The budget the request asked for               |
| `gen_ai.usage.input_tokens`  | Prompt tokens, counted after tokenization      |
| `gen_ai.usage.output_tokens` | Tokens actually generated                      |
| `orbix.prefill_ms`           | Milliseconds from span start until prefill     |
| `orbix.decode_ms`            | Milliseconds from prefill until the span ended |

Measured against a live collector on Llama 3.2 1B with a 7-token prompt and `max_tokens: 8`:
`input_tokens` 7, `output_tokens` 8, `prefill_ms` 238, `decode_ms` 797. The two timing
attributes are what make a slow answer attributable to prefill or to decode instead of a guess,
and they are the whole attribution: there is one span per request, not one per scheduling round.

A request the server refuses before it reaches the engine produces **no span at all**. A pool
that cannot hold the request, a missing prompt, a bad sampler value: all are refused with a 400
and never become a sequence, so nothing is recorded. Measured: a request refused for needing more
KV blocks than the pool holds posted nothing to the collector. Tracing is not an audit log, then;
`/metrics` counts refusals, and the access log records the status, but a per-request trace of a
rejected call needs one of those rather than this.

The exporter never blocks the request path. Spans are queued in memory and dropped when the
queue is full or a send fails, with `orbix_trace_spans_dropped_total` as the record. Measured
against a collector that accepted connections and never replied: ten requests stayed at about
20 ms each and the server kept serving; a 300-request flood across 16 clients still answered
every request and dropped 139 of 300 spans in one run and 235 in another. The count is a race
between the sender and the flood rather than a fixed number, so plan for a large fraction to be
dropped under load and read the counter rather than assuming a rate.
