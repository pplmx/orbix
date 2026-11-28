# CLI reference

`orbix chat` runs on a paged cache with prefix reuse, so each turn re-attaches the blocks the
previous one computed instead of prefilling them. Reuse is whole-block only, so a conversation
shorter than one block has nothing to attach -- `/stats` reports the positions actually reused,
which is 0 until the history outgrows a block.

`/stats` also reports the KV pool, `N of M blocks free (S positions each, T total)`, because
running out is one of two failures a session hits repeatedly, the other being the conversation
outgrowing the window. Both errors say retrying changes nothing and which flag is the way out.
The overflow advice separates the two ways to reach it: `/clear` helps when the history is
what outgrew the window, and does not when a single message -- usually `--system` -- never fit
it. Measured on the 128-position fixture with a 120-word system prompt: the first turn
overflowed at 131 tokens and `/clear` followed by the same input overflowed again, because the
system prompt is re-sent every turn.

Each of those is explained in full once per conversation; every later attempt in the same
conversation prints the numbers and `(unchanged; /clear starts over)` instead of repeating the
paragraph, and `/clear` restores the full version for the next one. The rule covers both
failures, since both repeat until the conversation is cleared. Measured before it: 40 inputs
against the 128-position fixture produced 34 copies of the overflow paragraph, and eight inputs
against `--blocks 4` produced six copies of the pool one.
The total is there because the free count alone misleads: a session refused every turn may show
all blocks free, meaning the request is larger than the pool can ever hold.

`/clear` forgets the conversation and nothing else. Published blocks stay in the prefix index,
so the next prompts still attach what they share with the old history -- which is what makes it a
way out of a full window rather than a fresh start.

A turn the window cut short says so, once per session: `(the 128-token window is full, so the
128-token budget was cut to 104; /clear or lower --max-tokens)`. Without it a cut reply is
indistinguishable from one the model ended itself, since both stop mid-sentence. The note is
printed once rather than under every later turn, because the limit is a property of the
conversation and every turn after the first cut runs into it. `run --stats` prints the same
figures as a `note:` line, and `bench` reports them in its `NOTE` column.

The Status column names the release each command ships in. Three have been tagged (`v0.1.0`,
`v0.2.0`, `v0.3.0`), so every marker in the table below names a release that exists. Every command
in it works today, and the marker says when it arrived rather than whether it is usable.

| Command                      | Description                                                                                                  | Status |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| `orbix model inspect <path>` | Print GGUF metadata, tensor types, quantization, and whether this build reproduces the model's pre-tokenizer | v0.1   |
| `orbix run "<prompt>"`       | One-shot inference                                                                                           | v0.1   |
| `orbix version`              | Print version and build info; `-json` for a machine-readable object                                          | v0.1   |
| `orbix chat`                 | Interactive REPL with conversation history                                                                   | v0.3   |
| `orbix serve`                | Start the OpenAI-compatible inference server                                                                 | v0.3   |
| `orbix healthcheck`          | Probe a running server's `/healthz`; the probe for a container that has no curl                              | v0.3   |
| `orbix bench`                | Run a built-in throughput/latency benchmark                                                                  | v0.2   |

`version` reports what `-ldflags` baked in at build time, which only `just build` and the release
workflow do. A binary from the README's `go install ...@latest` prints `orbix dev` with
`commit: none` and `built: unknown` -- those are the zero values of the injected variables, not an
error. The `go`, `platform` and `cgo` lines come from the runtime and are correct either way.

```bash
orbix run --model ./model.gguf "Once upon a time"
orbix run --model ./model.gguf --stats --max-tokens 64 "Explain KV caching."
orbix run --model ./model.gguf --workers 32 "Explain KV caching."   # see below on the ratio
orbix run --model ./model.gguf Once upon a time    # the same prompt, unquoted
orbix run --model ./model.gguf -- "-5 degrees outside"   # a prompt that starts with a dash
echo "Once upon a time" | orbix run --model ./model.gguf --stdin   # prompt from a pipe

orbix chat --model ./model.gguf

orbix bench --model ./model.gguf --concurrency 4 --max-tokens 32
orbix bench --model ./model.gguf --suite     # every scenario in one table

orbix model inspect ./model.gguf
orbix model inspect --json ./model.gguf   # machine-readable; see the note on arrays below
```

An empty prompt is accepted by `run` and refused by `serve`, and the difference is deliberate.
`run ""` encodes to the model's BOS token alone -- `--stats` reports `prompt: 1 tokens` -- and
generates from there, which is a completion from scratch and a thing this command can do.
`serve` answers `400` with `missing_required_parameter` on `prompt`, because an HTTP request
whose prompt is empty or whitespace-only is a client that sent nothing, and generating a story
from it is work nobody asked for. `chat` sidesteps the question: a line that is empty or only
whitespace is skipped before it becomes a turn, so the model is never asked to answer nothing.

In the `--json` output a scalar keeps its GGUF type: integers stay numbers, strings stay
strings, booleans stay booleans, and a non-finite float becomes the string `"NaN"` or `"+Inf"`
because JSON has no token for either. An **array** does not: it is rendered as the summary string
the human table shows, such as `"[151387 uint32]"` for a 151387-element `uint32` array. That is a
count and an element type, not the elements, which is deliberate for vocabularies whose
token list runs to six figures -- but it means a script that wants the entries cannot read them
from this output, and `--json` is not a full fidelity dump of the file.

`orbix model inspect` prints the GGUF layout readably. Its tables cut a metadata value or tensor
name at 60 bytes with a rune-safe `...`, since one long name otherwise pushes the later columns
off screen; `--json` carries the whole name. `--json` also keeps each value the type the file
declares, with two exceptions: a NaN or infinite float becomes the string `"NaN"`, `"+Inf"` or
`"-Inf"` because JSON has no token for either and one such key would otherwise fail the whole
document, and an array becomes a summary string like `"[151387 uint32]"` rather than a JSON array
-- see the note above on what that costs a script.

`orbix bench --suite` prints every scenario as one table, pairing each mechanism with a control
row -- paged against contiguous, prefix reuse against none, shortest against FIFO -- so a
difference is attributable to that mechanism. How much reuse is worth depends on the mix: the
pair shares 41 prompt tokens against 8 generated, and raising `--max-tokens` dilutes the saving
because reuse does not touch generation.

The suite fixes the settings its rows compare -- concurrency, request count, admission policy,
cache shape -- so those flags are refused when `--suite` is given. `--model`, `--max-tokens`,
`--prompt` and `--workers` still apply: they describe the machine and the workload.

One measurement, with everything needed to judge it -- stories15M q4_0, 41-token prompt, 8
generated per request, 8 workers, 128-core EPYC 7742, Go 1.27.3. A live run prints its own
conditions, so a table you produce can be compared against this one:

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
  the scheduler. The round in `NOTE` is the property, and what it has to show is the ordering:
  under `shortest` the short request's first token arrives within the first handful of rounds,
  under `fifo` it waits behind the long one for a number of rounds that scales with that
  request's budget. The exact round is not worth comparing against a table -- measured
  repeatedly at `--max-tokens 60` on the 256-position fixture, `shortest` landed on round 3 or 4
  and `fifo` on 64 or 65, so the run-to-run noise is a round or two even on one machine. Read
  the two figures as "a few rounds" against "one round per token of the long request". The rows
  raise any budget below 60 tokens so there is a queue to order, which the `NOTE` says when it
  happens.
- **The prefix rows run their four requests one after another.** A concurrent burst prefills
  them together, so no request arrives after another cached its prefix. Both rows repeat the
  prompt until it spans two blocks, because a block is published only once its last position
  is written.

A row can show fewer tokens than its budget without being a measurement failure: the model
emitted its end-of-sequence token, and the `NOTE` column says so. A single measurement reports
the same kind of shortfall as a `note:` line for the case the suite cannot hit -- a window that
leaves less room than `--max-tokens` asks for -- naming both counts, because `finish: length`
reads as "the budget was reached" either way.

## Metrics

`orbix serve` exports these numbers over `/metrics` in Prometheus text format, so a suite's
baseline and a live deployment's behaviour are the same measurements:

```text
# TYPE orbix_tokens_generated_total counter
orbix_tokens_generated_total 6
# TYPE orbix_requests_queued gauge
orbix_requests_queued 0
# TYPE orbix_kv_pool_used_fraction gauge
orbix_kv_pool_used_fraction 0
```

The full set, in the order the handler writes them:

| Family                                    | Type    | Present when      | What it says                                                  |
| ----------------------------------------- | ------- | ----------------- | ------------------------------------------------------------- |
| `orbix_requests_submitted_total`          | counter | always            | Requests accepted into the engine                             |
| `orbix_requests_finished_total`           | counter | always            | Requests that reached their end, whatever the reason          |
| `orbix_requests_finished_by_reason_total` | counter | always            | The same, split by `reason` label                             |
| `orbix_tokens_generated_total`            | counter | always            | Tokens delivered to callers                                   |
| `orbix_rounds_total`                      | counter | always            | Scheduling rounds that advanced at least one sequence         |
| `orbix_prefix_positions_reused_total`     | counter | always            | Prompt positions served from a shared KV prefix               |
| `orbix_requests_active`                   | gauge   | always            | Sequences being advanced right now                            |
| `orbix_requests_queued`                   | gauge   | always            | Accepted requests waiting for a slot                          |
| `orbix_round_max_active`                  | gauge   | always            | Sequences a round can advance                                 |
| `orbix_requests_refused_queue_full_total` | counter | always            | 503s because the backlog was full                             |
| `orbix_requests_refused_pool_total`       | counter | always            | 503s because the KV pool had no room                          |
| `orbix_kv_pool_blocks`                    | gauge   | paged             | Blocks in the shared pool                                     |
| `orbix_kv_pool_free`                      | gauge   | paged             | Unused blocks                                                 |
| `orbix_kv_pool_block_size`                | gauge   | paged             | Positions one block holds                                     |
| `orbix_kv_pool_used_fraction`             | gauge   | paged             | Share of blocks in use                                        |
| `orbix_trace_spans_dropped_total`         | counter | `--otlp-endpoint` | Spans discarded because the export queue was full or failed   |
| `orbix_model_info`                        | gauge   | always            | Constant 1, labelled `model` with the checkpoint being served |

The exposition is written by hand rather than through a client library, for the same reason
the GGUF parser is: a family is three lines, and the encoder is read by code review rather than
trusted as a dependency. Sixteen families come to 52 lines on a paged server, because a
labelled series costs one line per value -- measured, and the table above lists seventeen rows
because it also covers the one family that only a traced deployment has.

The count depends on the configuration: a contiguous server exports twelve families (the four
`kv_pool_*` gauges are omitted without a pool, since a zero would read as "the pool is empty"),
and `--otlp-endpoint` adds one more. Families are omitted, never exported as zero.

Three exist because their failure mode is invisible in latency until it is already failing.
`orbix_requests_queued` growing against a steady round rate is capacity, not a slow model;
`orbix_kv_pool_free` reaching zero is a paged deployment about to refuse requests. The third,
`orbix_requests_finished_by_reason_total`, is what catches clients leaving: a disconnect looks
entirely healthy in every other number, and shows up here as `reason="cancel"` or
`reason="abandoned"` growing against `reason="stop"`. The label's value comes from
`engine.FinishReason.String`, the same method that fills the `finish_reason` field, so the two
cannot drift: `stop`, `length`, `cancel`, `abandoned` and `error`. A `cancel` or `abandoned`
reason is one the client never sees in its own response -- it disconnected first -- which is why
this counter is where that shows up. The parts always sum to `orbix_requests_finished_total`, and
the tests assert it: the engine increments the total and one reason's slot in the same critical
section, and a reason it does not recognise is left out of the split rather than guessed at, so a
part smaller than the total is the signal that something produced an unknown reason.

`orbix_requests_refused_queue_full_total` and `orbix_requests_refused_pool_total` count the
requests answered `503`. They are the server's own counters rather than the engine's, since a
refused request never enters the engine and counting it as submitted would flatter the admission
rate. Two families rather than one, because the fixes differ: a full backlog wants a deeper
`--max-queued` or more replicas, an exhausted pool wants more `--blocks` or shorter prompts.

## The access log

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
