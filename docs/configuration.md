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
| `--workers`                | Kernel goroutines per batched pass; 0 means one per available CPU   |

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

| Flag       | What it controls                                                                 |
| ---------- | -------------------------------------------------------------------------------- |
| `--ctx`    | `run` only: context length; 0 uses the model's trained window, larger is refused |
| `--seed`   | `run` only: sampling seed; no effect without `--temp`                            |
| `--stats`  | `run` only: timings and token counts to stderr                                   |
| `--echo`   | `run` only: print the prompt before the generated text                           |
| `--system` | `chat` only: system prompt for the session                                       |

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

The same rule bites at the other end, and it is the one that surprises people: **a prompt shorter
than one block is never reused**, however many times it is sent. Measured with `--block-size 16`,
each prompt sent twice in a row, reading `orbix_prefix_positions_reused_total` before and after:

| Prompt (words) | Reused positions added by the two requests |
| -------------- | ------------------------------------------ |
| 8              | 0                                          |
| 20             | 16                                         |
| 32             | 48                                         |
| 48             | 64                                         |
| 64             | 64                                         |

Nothing is published until a whole block of the prompt has been consumed, so a short system
prompt costs a full prefill on every turn. `orbix_prefix_positions_reused_total` staying at 0 is
the symptom, and a smaller `--block-size` is the fix -- not a larger one, which raises the
threshold.

The rule holds on a real checkpoint as well as in that table, which was taken from a fixture.
Measured on gemma-3-270m-it-Q8_0 with a 30-token prompt sent twice, varying only `--block-size`:

| `--block-size` | Reused positions |
| -------------- | ---------------- |
| 8              | 16               |
| 16             | 16               |
| 32             | 0                |
| 64             | 0                |

The same 30 tokens reuse 16 positions below the threshold and none above it, and for the reason the
paragraph gives: at 32 the prompt never fills a block, so nothing is ever published.

How much is reused does grow with the prompt, in whole blocks. Measured on the same checkpoint with
`--block-size 16`, a fresh server per row, the same prompt sent twice, and the counter read across
the second request only:

| Prompt sent | Second request reused |
| ----------- | --------------------- |
| 1 sentence  | 0 positions           |
| 3 sentences | 32                    |
| 6 sentences | 48                    |
| 12 sentences| 112                   |

One sentence is about ten tokens, so it fills no block and reuses nothing -- the rule at the top of
this section. Four sentences' worth is where the count starts moving, and it keeps moving: the reused
positions are the whole blocks below the prompt's tail, so a longer prompt reuses more and the
ceiling is the prompt length rounded down to a block boundary.

Throughput barely moves across these values, measured on stories15M-q4_0 with eight requests at
concurrency four: 413.7 tok/s at a block size of 8 down to 406.4 at 64, a spread smaller than the
run-to-run noise. `--block-size` is a cache-granularity setting, not a throughput one.

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
order, as the round each got its first token. Three runs each, because the exact numbers are not
stable to the round: a token that arrives in the same round as the submit returns can be counted
either side of it.

Re-measured 2026-11 with `go test -run
TestShortestServesTheShortestFirstWhateverTheSubmissionOrder -v ./internal/engine/`, three runs, all
three identical:

| Policy     | First-token rounds |
| ---------- | ------------------ |
| `fifo`     | `2, 19, 28, 33`    |
| `shortest` | `19, 10, 5, 2`     |
| `fair`     | `5, 6, 7, 8`       |

Those are one round earlier than the table above at every position except the third, which matches,
and the same three runs on this host give the same numbers each time rather than drifting. What
changed between the two measurements is not recorded -- a different host, or an engine change since
-- so treat the offset as a reminder that these are counts of rounds rather than a figure to match.

What the table is for holds either way. `fifo` increases, `shortest` reverses that order so the
two-token request moves from round 33 to round 2, and `fair` gives each request exactly one turn
before anyone gets a second.

| Policy     | First-token rounds                | What it did                                           |
| ---------- | --------------------------------- | ----------------------------------------------------- |
| `fifo`     | `3, 20, 28, 34` / `2, 20, 29, 34` | served in submission order                            |
| `shortest` | `20, 11, 6, 2` / `20, 10, 6, 3`   | inverted: the 2-token request goes from last to first |
| `fair`     | `6, 7, 9, 9` / `6, 7, 8, 9`       | one turn apiece first, then round-robin               |

The ordering is what the table is for, and it is identical in every run: `shortest` inverts
`fifo` exactly, and `fair` gives all four a turn by round 9 where `fifo` reaches the fourth at
round 34. Two entries share a round under `fair` because a round admits more than one sequence
once each has had a turn.

These were reproduced from the engine directly rather than through `orbix bench --suite`: that
suite's policy rows submit two requests (one 60-token, one 3-token) and report a first-token
latency in milliseconds, which answers a different question. The numbers here are rounds, on the
four-request queue above.

Rounds are the unit the engine schedules in; what an operator waits is wall-clock, and the gap
there is larger than the round numbers suggest. Measured on Llama 3.2 1B with `--max-active 1`
and a 60-token request already running, a second request budgeted 4 tokens waited:

| Policy     | Wait for the short request |
| ---------- | -------------------------- |
| `fifo`     | 5230, 5105, 5066 ms        |
| `shortest` | 544, 422, 761 ms           |

A median of 5105 ms against 544 ms is 9.4 times, and no run of one policy came within six times
of any run of the other. The mechanism is the same as the round table: under `fifo` the short
request sits behind the full 60 tokens, while `shortest` picks it at the next scheduling
decision and finishes it in a handful of rounds.

These numbers were re-taken on an idle host after an earlier attempt on a loaded one, and the
comparison is worth recording because of how much the environment moved. The same measurement
under a load average near 100 gave 5820-8365 ms against 776-942 ms: 9.5 times by median where
the idle host gave 9.4, while the individual figures ran 14% to 73% high. The ratio is the
property; the milliseconds are the machine. Take the ordering from this page and re-measure the
timings on the host that will run them.

The rounds in the table above are the load-independent statement. The timings are here to show
that a round is not a fixed amount of time, and that a queue reordered by remaining work is felt
in seconds rather than in scheduling bookkeeping.

The two cache layouts differ in what they reserve. A contiguous cache reserves the whole context
for every accepted request on arrival: `context x KV width x layers`, times the active and queued
slots. A paged cache takes blocks as a sequence generates, so memory follows what is in flight.
The server prints the figure it computed at startup and warns past a gibibyte. It prints a
`weights:` line beside it for the part that exists before any request arrives -- see
[the weights section](#the-weights-are-held-as-float32-which-is-the-memory-figure-to-plan-with),
which is usually the larger of the two.

Three capacity failures look alike from the client and are not the same, so they are answered
differently:

| What happened                                          | Answer                         | What a client does |
| ------------------------------------------------------ | ------------------------------ | ------------------ |
| The backlog is full (`--max-queued` reached)           | `503` + `Retry-After`, at once | back off and retry |
| A running sequence cannot get a block it needs         | `503` + `Retry-After`          | back off and retry |
| The request needs more blocks than the pool ever holds | `400`                          | shorten it         |

The first is answered immediately rather than after a wait -- measured, 15 ms for a request
arriving against a one-slot server whose single queue place was taken. Queueing it behind work
that is already queued would only lengthen every client's timeout for an answer that is known
now. Measured with `--max-active 1 --max-queued 2`: three requests accepted (one running, two
waiting) and the next two refused with `engine: too many requests waiting (2 queued for 1 active
slots)`.

A paged server answers `503` with `Retry-After` when a *running* sequence cannot get the block
it needs, which is a capacity signal a client or proxy should back off on. A request that
merely arrives while the pool is empty is not refused: it is admitted and waits for blocks to
come back, because every active sequence advances one token at a time and releases the blocks
of a finished one. Measured with four blocks and four concurrent requests: `orbix_kv_pool_free`
read 0 while a fifth request needing two blocks answered 200.

That wait is latency, and it is the thing to size `--blocks` for. On the same saturated server
the fifth request took 3.62 seconds to answer, against 1.79-2.06 seconds for the same request
on an idle one -- so about 1.7 of those seconds were spent waiting for a block, queueing behind
four 20-token requests that each held one to completion. A deployment whose latency budget
cannot absorb that needs more blocks, not a different policy: no scheduling rule makes a block
available sooner.

The base figure is worth seeing too. Two seconds for 20 tokens on Llama 3.2 1B is the engine's
own speed on this host -- 9.6 tok/s measured with `run --stats` on the same checkpoint -- not
overhead the pool introduced.

The 503 is not the answer for a request whose own prompt and budget need more blocks than the
pool holds at all: nothing another request finishes makes room for it, so retrying only repeats
the answer. That case is
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

"One per CPU" is `runtime.GOMAXPROCS(0)`, not the host's core count, and the two differ wherever
a CPU quota is set. Measured in a scope limited to four CPUs of a 128-core host: `NumCPU` still
reported 128 while `GOMAXPROCS` reported 4. So a container gets four workers by default rather
than a hundred and twenty-eight competing for four cores -- which is the useful behaviour, and
worth knowing before reading the worker figures above as if they applied to a capped pod.

The default is also the best value there, not merely a safe one. In the same four-CPU scope,
stories15M q4_0 with 64 greedy tokens:

| `--workers` | tok/s (two runs) |
| ----------- | ---------------- |
| 1           | 66.6, 66.4       |
| 2           | 120.7, 121.2     |
| 4 (default) | 192.3, 193.2     |
| 8           | 196.2, 194.7     |

Scaling rises to the quota and flattens past it -- 1.82 times at two workers and 2.90 at four,
not the 2x and 4x a linear reading would predict, because the kernels have a serial part and
because four jobs on four CPUs still contend. Eight workers, twice the quota, land 1-2% above
four: inside the spread between runs, so the plateau is flat rather than falling. The automatic
value sits on that plateau rather than inside the climb, which is the property that matters.
`--workers 0` and an explicit `--workers 4` measured 189.4 and 192.3 -- the same within
run-to-run noise, which is what "0 means GOMAXPROCS" should look like.

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

How virtual is worth a number, because the banner's arithmetic is alarming by design. On Llama
3.2 1B with the default slots it prints:

```text
kv cache:   contiguous, about 8.0 GiB per request, 576.0 GiB for 8 active + 64 queued
```

Measured on that server: resident memory was 5355 MiB with nothing in flight and 5445 MiB after
eight concurrent requests finished -- a 90 MiB delta against a 576 GiB figure, because Linux
commits a page when it is written and a four-token reply touches four positions of a 128k
window. The figure is what the address space would hold if every accepted request ran to the end
of its context at once; it is the right thing to size `--max-active` by and the wrong thing to
compare against `free`.

`--blocks` does not change that. The pool is sized ahead but committed lazily too: a server
reporting `paged, 4096 blocks x 16 positions, 4.0 GiB shared` sat at 5422 MiB with nothing in
flight and 5458 MiB after a 200-token request -- 36 MiB against a 4 GiB pool, because only the
blocks a sequence fills are ever written. What `--blocks` buys is a bound: the process cannot
exceed that pool however many requests arrive, where a contiguous cache reserves a whole context
per accepted request and the sum is the banner's figure.

Keeping the model in its quantized form and decoding per matmul would remove the expansion at
the cost of speed, since each weight would be decoded once per use rather than once per load.
That trade is not taken here: a decoder in the inner loop is a different engine, and this one
is written to be read.

## Tracing covers requests that reach the engine

`--otlp-endpoint` posts OTLP/HTTP+JSON to a collector. What arrives is one span per **sequence**,
and the attributes are the OpenTelemetry `gen_ai` ones plus a few of this engine's own, which split
the duration into the parts an operator has to tell apart.

One span per sequence rather than per request, because a request with `n: 3` is three sequences:
the server submits one per choice so they share a batch. `best_of` adds more, since its candidates
are sequences too. The spans of one HTTP request share `gen_ai.response.id`, and
`orbix.choice_index` is what tells them apart -- without it a trace reader could count three spans
and learn nothing from any of them. The attribute is absent when the request had a single sequence,
where an index of 0 would say nothing.

| Attribute                    | Meaning                                             |
| ---------------------------- | --------------------------------------------------- |
| `gen_ai.request.model`       | The name the request asked for                      |
| `gen_ai.response.model`      | The name this server answers as                     |
| `gen_ai.request.max_tokens`  | The budget the request asked for                    |
| `gen_ai.usage.input_tokens`  | Prompt tokens, counted after tokenization           |
| `gen_ai.usage.output_tokens` | Tokens actually generated                           |
| `orbix.queue_ms`             | Milliseconds from span start until work began       |
| `orbix.prefill_ms`           | Milliseconds from work starting until prefill ended |
| `orbix.decode_ms`            | Milliseconds from prefill until the span ended      |
| `orbix.choice_index`         | Which of the request's sequences this span is       |
| `orbix.choice_count`         | How many sequences the request generated            |

Measured against a live collector on Llama 3.2 1B with a 7-token prompt and `max_tokens: 8`:
`input_tokens` 7, `output_tokens` 8, `prefill_ms` 238, `decode_ms` 797.

The three timing attributes are what make a slow answer attributable instead of a guess, and they
have to be three: the span starts when the request is submitted, so a request that waits for a slot
spends part of its span doing nothing, and measuring prefill from the span's start would make a
saturated server look like a slow model. `queue_ms` is that wait; `prefill_ms` and `decode_ms` are
measured from the moment the first forward pass began. Measured on the test fixture with one active
slot, a request queued behind a 2000-token prefill reports `queue_ms` 317, `prefill_ms` 0, while the
request that went first reports `queue_ms` 1, `prefill_ms` 316.

A request that never began — cancelled while queued, or failed before its first pass — reports its
whole lifetime as `queue_ms` and omits the other two, because those are the requests whose wait
matters most and reporting nothing would hide them.

There is one span per sequence, not one per scheduling round, so this is where the attribution
ends. The two views of the token counts agree: `usage.completion_tokens` in the response body
counts every sequence generated, and the sum of `gen_ai.usage.output_tokens` over a request's
spans is the same number. Measured with `n: 3` and `max_tokens: 4`, the usage block said 12 and
three spans said 4 each.

A request the server refuses before it reaches the engine produces **no span at all**. A pool
that cannot hold the request, a missing prompt, a bad sampler value: all are refused with a 400
and never become a sequence, so nothing is recorded. Measured on one server with one collector:
a request refused for needing 126 KV blocks from a 32-block pool posted nothing, while an
accepted request beside it posted its one span. Tracing is not an audit log, then;
`/metrics` counts refusals, and the access log records the status, but a per-request trace of a
rejected call needs one of those rather than this.

The exporter never blocks the request path. Spans are queued in memory and dropped when the
queue is full or a send fails, with `orbix_trace_spans_dropped_total` as the record. Measured
against a collector that accepted connections and never replied: ten requests stayed at about
20 ms each and the server kept serving; a 300-request flood across 16 clients still answered
every request and dropped 139 of 300 spans in one run and 235 in another. The count is a race
between the sender and the flood rather than a fixed number, so plan for a large fraction to be
dropped under load and read the counter rather than assuming a rate.
