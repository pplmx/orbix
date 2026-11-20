# CLI reference

`orbix chat` runs on a paged cache with prefix reuse, so each turn re-attaches the blocks the
previous one computed instead of prefilling them. Reuse is whole-block only, so a conversation
shorter than one block has nothing to attach -- `/stats` reports the positions actually reused,
which is 0 until the history outgrows a block.

`/stats` also reports the KV pool, `N of M blocks free (S positions each, T total)`, because
running out is one of two failures a session hits repeatedly, the other being the conversation
outgrowing the window. Both errors say retrying changes nothing and which flag is the way out.
The total is there because the free count alone misleads: a session refused every turn may show
all blocks free, meaning the request is larger than the pool can ever hold.

`/clear` forgets the conversation and nothing else. Published blocks stay in the prefix index,
so the next prompts still attach what they share with the old history -- which is what makes it a
way out of a full window rather than a fresh start.

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
orbix model inspect --json ./model.gguf   # machine-readable; see the note on arrays below
```

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
declares, except that a NaN or infinite float becomes the string `"NaN"`, `"+Inf"` or `"-Inf"` --
JSON has no token for either, and one such key would otherwise fail the whole document.

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
