package engine

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/otel"
)

// ErrQueueFull is returned by Submit when the engine already has as many waiting requests
// as it will hold.
//
// It is a distinct error because the caller's reaction differs from a malformed request:
// the request was well formed, and retrying later or shedding load is the right answer.
var ErrQueueFull = errors.New("engine: too many requests waiting")

// ErrPoolExhausted reports that a sequence needed a KV block and the pool had none left, even
// after giving up every cached prefix it was holding on speculation.
//
// It is the scheduler's answer to "is this transient?", which is why it is declared here rather
// than left to the storage layer: a request that arrives on an exhausted pool is well formed and
// the process is healthy, so a server should say 503 and suggest a retry, not 500. Callers used
// to decide that by importing internal/llama and testing its sentinel, which put a scheduling
// judgement in the HTTP layer and made internal/server depend on the cache's package for it.
//
// poolAwareErr translates the storage layer's error into a shape that matches this sentinel on
// the way out, so a caller can ask errors.Is(err, ErrPoolExhausted) without the import. The text
// is still the storage layer's own -- it carries the block count and the block size, which is
// what an operator acts on.
var ErrPoolExhausted = errors.New("engine: block pool exhausted")

// errCallerGone reports that the caller stopped reading events before the sequence finished.
// It is a sentinel rather than an inline fmt.Errorf so the span can classify it: error.type must
// be a stable, low-cardinality string, which an error that carries the sequence id could not be.
var errCallerGone = errors.New("engine: caller stopped reading events")

// Event is one step of a sequence's output.
type Event struct {
	// Token is the generated token. HasToken says whether it is meaningful:
	// token 0 is a valid id (usually <unk>), so a zero value cannot mark the
	// absence of one.
	Token    int32
	HasToken bool
	// Done marks the last event for a sequence.
	Done bool
	// Err is set when the sequence failed; it is the last event.
	Err error
	// FinishReason says why generation stopped: "length", "stop", or "error".
	FinishReason string
	// Generated is how many tokens the sequence produced in total, set on the terminal
	// event. It is the honest figure for usage reporting, because a caller can stop reading
	// early -- at a stop string in the stream, say -- and still owe the count of what the
	// model actually generated. The alternative, counting the events a consumer happened to
	// receive, under-reports by however many it skipped.
	//
	// An abandoned sequence never delivers its terminal event: emit refuses everything once
	// the caller is judged gone, which is what abandonment means. Measured with an unread
	// channel and a 900-token budget, the caller received 67 events and then a closed channel
	// with no Done at all, while the engine had generated 68. A consumer that needs a total
	// for a request it stopped reading has to take it from somewhere other than this event.
	Generated int
	// Logprob is the natural logarithm of the probability the model gave Token, and
	// HasLogprob says whether it is meaningful. It is set only when the request asked for
	// log probabilities, because computing one costs two passes over the vocabulary.
	//
	// Zero is a real log probability (a token the model considered certain), which is why the
	// absence needs its own flag rather than a sentinel value.
	Logprob    float32
	HasLogprob bool
	// Top holds the most likely alternatives at this position, descending, when the request
	// asked for them. It is nil when only the chosen token's log probability was wanted.
	Top []Alternative

	// Round is the engine round that produced this event.
	//
	// It is stamped here rather than read from Engine.Round() by the consumer because a
	// consumer runs on its own goroutine: by the time it receives an event, the engine
	// may have advanced several rounds, so a counter read would report when the event
	// was noticed rather than when it was produced. A test that observes scheduling
	// needs the latter, and so does anyone reasoning about which round generated a
	// token.
	Round int
}

// Request is one generation job.
type Request struct {
	// Prompt is the already-tokenized input, including any BOS token.
	Prompt []int32
	// MaxTokens caps the generated tokens, not counting the prompt.
	MaxTokens int
	// Stop adds token ids that end generation beyond the model's EOS.
	Stop []int32
	// Sampler picks each next token. Nil means greedy.
	Sampler Sampler
	// PresencePenalty and FrequencyPenalty are OpenAI's repetition controls, in [-2, 2].
	// Positive values discourage tokens that have already appeared: presence adds a flat
	// penalty per distinct token seen, frequency adds one proportionally to how often.
	//
	// They are applied by the engine rather than by the sampler because they need the
	// generated tokens, which Sampler.Next does not receive. A sampler that silently ignored
	// them would be indistinguishable from one that honoured them until the output repeated
	// itself, which is exactly when a caller sets them.
	PresencePenalty  float32
	FrequencyPenalty float32
	// Operation names the GenAI operation for the trace span: "chat" for a chat
	// completion, "text_completion" for the legacy completions endpoint. Empty means
	// "chat". It is separate from the prompt because the two endpoints differ only in how
	// the prompt was rendered, and a trace reader distinguishes them by this attribute.
	Operation string
	// ResponseModel is the model name the caller will see in the response, when it differs
	// from what the convention puts in gen_ai.request.model. Empty omits the attribute.
	ResponseModel string
	// Stream reports whether the caller is consuming this as a stream. The convention says
	// gen_ai.request.stream is set if and only if the request is streaming, and that an
	// unset attribute means non-streaming -- so leaving a streamed request unmarked is a
	// statement that it was not streamed.
	Stream bool
	// ResponseID is the completion id the caller will return, e.g. "chatcmpl-123". The
	// convention requires it on the span, and it is how a trace is joined to the response
	// a user is looking at.
	ResponseID string

	// Cancel, when set, stops the sequence at the start of its next round. The
	// sequence then reports Done with the finish reason "cancel". A caller that
	// owns a connection sets this to its request context so a client that goes
	// away does not keep the model busy: without it, a disconnected stream would
	// generate its whole token budget into a channel nobody reads.
	Cancel <-chan struct{}

	// ChoiceIndex and ChoiceCount say which of a request's several sequences this one is. The
	// server submits one sequence per choice, and each gets its own span, so without an index a
	// trace reader sees n spans that differ only by span id. ChoiceCount is 0 when the request
	// had a single sequence, which is the common case and needs no index.
	ChoiceIndex int
	ChoiceCount int

	// Trace, when non-nil, records one span for this request, carrying orbix.queue_ms,
	// orbix.prefill_ms and orbix.decode_ms so a slow answer can be attributed to the wait for
	// a slot, the prompt pass, or generation rather than
	// guessed at.
	//
	// One span, not one per round: those two attributes are the whole attribution, and they
	// come from a timestamp taken when prefill returns. This comment said "a span for each
	// round that advances it" while nothing in the package created a round span -- measured
	// with a live collector, a request that decoded three tokens produced exactly one span,
	// named for the operation.
	//
	// It is per-request rather than per-engine because a caller may trace some traffic
	// and not other traffic, and because an engine shared by an untraced caller must not
	// pay for a tracer it never asked for. The exporter's own lifetime belongs to the
	// caller.
	Trace *otel.Exporter

	// Logprobs asks for the distribution each generated token was drawn from. False -- the
	// default -- computes nothing, which keeps the sampling path exactly as it was for a
	// caller that did not ask; the cost when it is set is two passes over the vocabulary row
	// per token.
	//
	// It is a separate switch from TopLogprobs because OpenAI's completions endpoint spells
	// "the chosen token's log probability and no alternatives" as logprobs: 0, so zero
	// alternatives is a request rather than an absence.
	//
	// The rows are described from the full-vocabulary log-softmax at the request's
	// temperature, not from the truncated set a top-k or top-p sampler draws from; see
	// llama.TokenLogprobs for why that is the honest number to report.
	Logprobs bool
	// TopLogprobs is how many of the most likely alternatives to describe alongside the
	// chosen token. It has no effect unless Logprobs is set.
	TopLogprobs int
	// Temperature is the sampling temperature, needed only to describe the distribution that
	// TopLogprobs asks for. It is not otherwise consulted here -- the sampler already has it.
	Temperature float32

	// Priority ranks this request against others waiting for a slot: a higher value
	// advances first. Zero is the default and means "with the rest".
	//
	// This is the caller's ranking, which the engine cannot infer. The admission
	// policy decides how to order requests of *equal* priority; it cannot know that
	// one of them is an interactive request a person is watching and another is a
	// batch job that will not be read for an hour. Negative values are lower than
	// default, which is what a background prefill wants.
	Priority int
}

// sequence is the scheduler's per-request state.
type sequence struct {
	id      int
	req     Request
	cache   llama.Cache
	pos     int     // positions written; also the prompt index while prefilling
	out     []int32 // generated tokens
	events  chan Event
	sampler Sampler

	// logits is this sequence's own copy of the last forward pass output. The
	// model owns a single scratch buffer, so interleaving sequences would
	// otherwise read whichever sequence ran most recently, which produced
	// fluent-looking garbage when two requests shared the engine.
	logits []float32

	// lastServed is the round in which this sequence last advanced, or -1 before its
	// first turn. PriorityFair uses it to give the long-waiting sequence a slot.
	lastServed int

	// round is the round the sequence is running in, stamped on every event it emits so
	// an event carries its own provenance instead of relying on when it was read.
	round int

	// abandoned is set when an event could not be delivered because the caller is no
	// longer reading. The engine stops advancing the sequence and releases its blocks
	// instead of blocking on a channel nobody drains.
	abandoned bool

	// reqSpan is the request's trace span, or the zero value when tracing is off. It is
	// finished when the sequence ends, which is why it travels with the sequence rather
	// than being recorded at Submit: a span has to cover the whole request, not just its
	// admission.
	reqSpan otel.Span
	// traceOn reports whether reqSpan is live, keeping the check a single bool load.
	traceOn bool
	// started is when the first forward pass over this request's prompt began. It is zero
	// until the scheduler picks the sequence up, which is what makes the wait before it
	// measurable: the span starts at Submit, so the time between the two is queue time.
	started time.Time
	// prefilled is when the prompt finished prefilling, so the request's span can
	// separate the cost of reading the prompt from the cost of generating. Both phases
	// are recorded as attributes on one span rather than as child spans, because a
	// child span per phase would triple the trace volume for information that fits in
	// two numbers.
	prefilled time.Time
}

// remaining is how many tokens the sequence still has to emit.
//
// It counts the budget left rather than the model's work: a sequence that has emitted
// most of its tokens is close to finishing, which is the property PriorityShortest
// sorts on.
func (s *sequence) remaining() int {
	left := s.req.MaxTokens - len(s.out)
	if left < 0 {
		return 0
	}
	return left
}

// Engine generates for several sequences at once.
//
// Concurrency here is cooperative, not parallel: llama.Model keeps scratch
// buffers per instance, so a single loaded model runs one forward pass at a
// time. What the engine buys is batching -- a round prefills every pending
// prompt and then decodes every active sequence in one pass -- so concurrency
// raises total throughput rather than only dividing one stream's speed. Each
// sequence still advances by one token per round, which is what keeps a short
// request from waiting behind a long one.
type Engine struct {
	model     *llama.Model
	maxActive int
	maxQueued int
	policy    Priority

	// round counts Pump calls. It exists so PriorityFair can record when a sequence
	// last ran without a wall clock: a round number is deterministic, and a test that
	// asserted on wall-clock times would be flaky on a loaded machine.
	round int

	// pool, when set, is the shared block store paged caches draw from. It is nil
	// for the contiguous layout, which gives every sequence a whole-context arena
	// up front.
	pool *llama.BlockPool

	mu     sync.Mutex
	active map[int]*sequence
	nextID int

	// counters are cumulative and guarded by mu, so a Stats reading is a consistent
	// snapshot rather than a mix of moments.
	requestsSubmitted uint64
	requestsFinished  uint64

	// finishedByReason splits requestsFinished by why each request ended. Written and read
	// under mu, like the total it refines.
	finishedByReason [FinishReasonCount]uint64
	tokensGenerated  uint64
	rounds           uint64
	// prefixReused counts prompt positions served from a shared prefix instead of a
	// forward pass. It stays zero for a contiguous cache, which cannot share.
	prefixReused uint64
}

// Options configures an Engine.
type Options struct {
	// MaxActive caps how many sequences are advanced per round. Zero means 8.
	MaxActive int
	// MaxQueued caps how many sequences may be accepted but not yet active. Zero means
	// 8*MaxActive.
	//
	// A contiguous cache reserves its whole context at Submit, so without a cap the
	// memory a client can make the process reserve is bounded only by how fast it can
	// open connections: every accepted request holds an arena whether or not a round
	// will reach it. Rejecting is the honest answer once the backlog is this deep --
	// the requests would otherwise wait behind each other for a slot that is not
	// coming, and the caller would rather know now.
	MaxQueued int
	// Workers is the model's kernel fan-out; zero means one per available CPU.
	// The engine drives one forward pass at a time, so this is the only
	// parallelism in play and defaulting it to the CPU count is what makes a
	// single stream fast.
	Workers int
	// BlockSize and Blocks switch the engine to a paged KV cache drawn from one
	// shared pool of Blocks blocks of BlockSize positions. Blocks == 0 keeps the
	// contiguous layout, where every sequence reserves the whole context as soon
	// as it is submitted.
	//
	// The trade is explicit. Contiguous is simpler and makes a sequence's memory
	// independent of its neighbours, but it charges a request for the context it
	// might use rather than the positions it does use: at a 4096-token window a
	// queued 4-token request still reserves 4096 positions. Paging sizes the
	// cache to the position count instead, at the cost of indirection in
	// attention.
	BlockSize int
	Blocks    int
	// Policy decides which requests advance when more are ready than MaxActive
	// allows. Zero is PriorityFIFO, which is submission order.
	Policy Priority
	// PrefixCache turns on prefix reuse for a paged cache. Requests whose prompts
	// start with the same whole blocks of tokens attach the already-computed keys
	// and values instead of prefilling them again, which is what makes a shared
	// system prompt or a multi-turn conversation cheap: the part that is re-sent
	// verbatim every turn costs one block lookup instead of a forward pass.
	//
	// It has no effect without Blocks, because a contiguous cache has no blocks to
	// share.
	PrefixCache bool
}

// New loads a model and returns an engine for it.
func New(path string, opts Options) (*Engine, error) {
	m, err := llama.NewWithOptions(path, llama.Options{Workers: opts.Workers})
	if err != nil {
		return nil, err
	}
	maxActive := opts.MaxActive
	if maxActive <= 0 {
		maxActive = 8
	}
	maxQueued := opts.MaxQueued
	if maxQueued <= 0 {
		// Enough slack that a burst is absorbed rather than refused, small enough that
		// the reserved memory stays proportional to what the engine can actually run.
		maxQueued = 8 * maxActive
	}
	e := &Engine{
		model:     m,
		maxActive: maxActive,
		maxQueued: maxQueued,
		policy:    opts.Policy,
		active:    make(map[int]*sequence),
	}
	if opts.Blocks > 0 {
		pool, err := llama.NewBlockPool(m.Config(), llama.BlockPoolOptions{
			BlockSize: opts.BlockSize,
			Blocks:    opts.Blocks,
		})
		if err != nil {
			return nil, err
		}
		if opts.PrefixCache {
			pool.EnablePrefixCache()
		}
		e.pool = pool
	}
	return e, nil
}

// Pool returns the shared block pool, or nil when the engine uses the contiguous
// layout. It exists so a caller can report the memory budget and how much of it a
// workload is using.
//
// Like Model and MaxActive, it reads a field that is assigned in New and never written
// again, so no lock is taken here. The pool's own state is concurrent-safe; what is not
// protected is the field, and it does not change.
func (e *Engine) Pool() *llama.BlockPool { return e.pool }

// Model exposes the loaded model, for configuration and statistics.
//
// The pointer is set in New and never replaced, so reading the field needs no lock. The
// value it points at is not read-only, though: a forward pass writes the model's scratch
// buffers, so a caller that runs inference through the returned pointer would be corrupting
// a pass the engine is running. The call sites read Config(), W.Params() and Runner() --
// configuration and counters, never a buffer -- which is the use this is for.
func (e *Engine) Model() *llama.Model { return e.model }

// Submit queues a request and returns the channel its events arrive on.
//
// The channel is closed after the final event, so a caller can range over it.
// Generation happens when Pump is called; Submit alone does no work.
//
// A caller that stops pumping is left holding an open channel that will never carry anything:
// Submit does not start a goroutine, so there is nothing else to close it. A caller who submits
// and then waits for an event without pumping waits forever, and the goroutine doing it leaks.
// Measured with no pump: nothing arrives on the channel and it stays open.
//
// Every caller here pumps: engine.Drive as `for eng.Pump() > 0`, the server's run loop as
// `if n := s.eng.Pump(); n > 0`, and the REPL as `if eng.Pump() == 0 { ... }`, which is why the
// requirement never shows in practice.
func (e *Engine) Submit(req Request) (<-chan Event, error) {
	cfg := e.model.Config()
	if len(req.Prompt) == 0 {
		return nil, fmt.Errorf("engine: empty prompt")
	}
	if len(req.Prompt) > cfg.ContextLen {
		return nil, fmt.Errorf("engine: prompt is %d tokens, context is %d", len(req.Prompt), cfg.ContextLen)
	}
	if req.MaxTokens < 0 {
		return nil, fmt.Errorf("engine: negative max tokens")
	}

	// Refuse before reserving anything, so a rejected request costs no memory.
	//
	// The limit protects the arena the contiguous layout takes here as much as it
	// protects the queue: a paged sequence takes blocks lazily, but its turn still
	// only comes after the slots ahead of it, so accepting it would not help either.
	// The queue check lives in the same critical section as the insert below, not in a
	// snapshot taken before it. It used to read len(active) under the lock, release, build the
	// cache, and re-acquire to insert -- so every goroutine that arrived during the gap saw the
	// same stale count and all of them were admitted. Measured with MaxActive 1 and MaxQueued
	// 2: twelve concurrent submits were all accepted and twelve sequences ended up active.
	//
	// That is a TOCTOU rather than a data race, which is why the race detector stays quiet:
	// every access to len(active) is guarded, and the defect is that the value outlives the
	// guard that read it.
	//
	// The cache is therefore built before the count is consulted, which is what a rejected
	// request used to avoid. It is the honest trade: a pool block is recycled when the sequence
	// is discarded, and a contiguous arena is freed with it, so a refusal costs the allocation
	// of one cache -- the alternative was admitting unboundedly many requests to save it.

	// Paged sequences take blocks as they generate; contiguous ones reserve the
	// whole context now. Both are created here so a request's admission cost is
	// the same call either way.
	var cache llama.Cache
	if e.pool != nil {
		cache = e.pool.New()
	} else {
		arena, err := llama.NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			return nil, err
		}
		cache = arena
	}

	e.mu.Lock()
	id := e.nextID
	e.nextID++
	s := &sequence{
		id:         id,
		req:        req,
		cache:      cache,
		pos:        0,
		events:     make(chan Event, 64),
		sampler:    req.Sampler,
		logits:     make([]float32, cfg.VocabSize),
		lastServed: -1,
	}
	if s.sampler == nil {
		s.sampler = llama.Greedy{}
	}
	if req.Trace != nil {
		tracer := req.Trace.Tracer()
		op := req.Operation
		if op == "" {
			op = "chat"
		}
		attrs := []otel.Attribute{
			// Required by the GenAI conventions, and the discriminator a backend uses to
			// pick the provider-specific attribute set. "orbix" is this engine's own name,
			// which the convention allows: the value is the provider as the instrumentation
			// knows it, not an upstream registry entry.
			otel.StringAttr("gen_ai.operation.name", op),
			otel.StringAttr("gen_ai.provider.name", "orbix"),
			otel.StringAttr("gen_ai.request.model", cfg.Name),
			otel.IntAttr("gen_ai.usage.input_tokens", int64(len(req.Prompt))),
			otel.IntAttr("gen_ai.request.max_tokens", int64(req.MaxTokens)),
		}
		if req.ResponseID != "" {
			attrs = append(attrs, otel.StringAttr("gen_ai.response.id", req.ResponseID))
		}
		// The convention separates the requested model from the one that answered: a
		// provider resolves an alias to a dated version, and the two differ. This server
		// serves exactly the one checkpoint it loaded, so both attributes hold the same
		// value today. It is set anyway because the attribute is Recommended and its
		// meaning is accurate -- what generated the response -- rather than because a
		// reader can learn something new from it.
		if req.ResponseModel != "" {
			attrs = append(attrs, otel.StringAttr("gen_ai.response.model", req.ResponseModel))
		}
		if req.Stream {
			attrs = append(attrs, otel.BoolAttr("gen_ai.request.stream", true))
		}
		// Only when there is a choice to make. A single-sequence request would otherwise carry
		// an index of 0 that says nothing, and every span in the common case would have it.
		if req.ChoiceCount > 1 {
			attrs = append(attrs,
				otel.IntAttr("orbix.choice_index", int64(req.ChoiceIndex)),
				otel.IntAttr("orbix.choice_count", int64(req.ChoiceCount)))
		}
		// The span name is the operation name, which is what the convention says to use.
		if span, ok := tracer.StartTrace(op, 2, attrs...); ok {
			s.reqSpan, s.traceOn = span, true
		}
	}
	// Checked here, under the same lock that publishes the sequence: see the note above on
	// why a snapshot taken earlier cannot be trusted.
	if q := len(e.active) - e.maxActive; q >= e.maxQueued {
		e.mu.Unlock()
		// The sequence was never published, so its channel and cache go with it. The caller
		// gets an error and a nil channel, which is the contract a rejected Submit has always
		// had.
		return nil, fmt.Errorf("%w (%d queued for %d active slots)",
			ErrQueueFull, q, e.maxActive)
	}
	e.active[id] = s
	e.requestsSubmitted++
	e.mu.Unlock()

	return s.events, nil
}

// Pump advances the active sequences by one round.
//
// It returns the number of sequences still running after the round. A caller
// loops until Pump reports zero; each call is one scheduling round, which is what
// makes the interleaving observable to tests.
//
// A round has two phases. Sequences still holding prompt tokens prefill, one
// batched pass each. Sequences that already hold their prompt sample their next
// token and then decode together in a single batched pass, so the projections run
// once for the whole round instead of once per sequence. A sequence prefills in
// one round and decodes from the next, which keeps every round a single step per
// sequence and leaves the interleaving property intact.
//
// At most MaxActive sequences advance per round; the rest wait in submission
// order. That bound is what keeps a burst of requests from allocating a batch
// whose scratch dwarfs the model.
//
// Only one goroutine may be inside Pump at a time. The engine does not serialise that
// for the caller: the model it drives owns one set of scratch buffers, so two concurrent
// rounds would write the same slices and produce output that is neither round's. The
// same constraint is why Model() returns a pointer whose value is not safe to run
// inference through while the pump is running -- the server and the REPL both keep Pump
// on a single goroutine and call Submit from wherever.
//
// Breaking the constraint panics rather than returning wrong numbers. Measured: two
// goroutines pumping one sequence give "panic: send on closed channel", because each round
// advances the sequence and the second closes the channel the first is still emitting into.
// That is the intended trade and the reason there is no test for it -- a lock would hide a
// caller bug behind slower rounds, while the panic names it, and a test whose whole content
// is a process-killing panic would take the package's other tests down with it. The cost is a
// crashed process rather than a per-request error, so every caller keeps Pump on one
// goroutine: see Server.run and the CLI's sequential loops.
//
// Panics inside the model propagate: a corrupt forward pass should not be turned
// into a per-request error that hides a bug in the engine.
func (e *Engine) Pump() int {
	e.mu.Lock()
	seqs := make([]*sequence, 0, len(e.active))
	for _, s := range e.active {
		seqs = append(seqs, s)
	}
	e.mu.Unlock()

	// Deterministic order keeps runs reproducible, which matters when comparing
	// against a reference implementation.
	sortByID(seqs)
	// The round number is taken under the lock and stamped on the admitted sequences
	// here, before any of them runs: PolicyFair compares it later, and reading it
	// without the lock would race with Round().
	e.mu.Lock()
	e.round++
	if len(seqs) > 0 {
		// round advances every call so PriorityFair's ordering stays monotonic, but the
		// exported counter only advances when a round actually had work: a caller that
		// pumps an idle engine must not make the throughput reading look busy.
		e.rounds++
	}
	round := e.round
	e.mu.Unlock()
	// Admit a round's worth. When everything fits, no policy applies and ready is
	// returned unchanged; when it does not, the policy decides who waits.
	seqs = admit(seqs, e.maxActive, e.policy)
	for _, s := range seqs {
		s.lastServed = round
		// The round is recorded on the sequence so the events it emits this round can
		// carry it without every emit site taking a lock to read the engine's counter.
		s.round = round
	}

	eos := int32(-1)
	if v := e.model.Config().EOSID; v >= 0 {
		eos = int32(v)
	}

	// Phase 1: prefill. Each sequence is independent here, and Prefill batches
	// internally, so there is nothing to share between sequences.
	var decoding []*sequence
	for _, s := range seqs {
		if s.pos < len(s.req.Prompt) {
			e.prefill(s)
			continue
		}
		decoding = append(decoding, s)
	}

	// Phase 2: sample, then decode the survivors together.
	batch := make([]pending, 0, len(decoding))
	for _, s := range decoding {
		tok, ok := e.sample(s, eos)
		if !ok {
			continue
		}
		batch = append(batch, pending{s: s, tok: tok})
	}
	if len(batch) > 0 {
		e.decode(batch)
	}

	// Phase 3: retire anything abandoned during the round. A sequence whose caller stopped
	// reading was skipped above, and finishing it here is what releases its blocks and
	// drops it from active -- without this it would stay counted forever and the caller's
	// loop would never see Pump reach zero.
	e.reapAbandoned()

	e.mu.Lock()
	n := len(e.active)
	e.mu.Unlock()
	return n
}

// reapAbandoned finishes every sequence whose events could not be delivered.
//
// The finish is marked with an error rather than a normal completion: the caller is gone,
// so nobody reads it, but a trace still records why the request ended and a test that
// inspects the channel sees a definite outcome instead of a truncated stream.
func (e *Engine) reapAbandoned() {
	e.mu.Lock()
	var gone []*sequence
	for _, s := range e.active {
		if s.abandoned {
			gone = append(gone, s)
		}
	}
	e.mu.Unlock()
	for _, s := range gone {
		e.finish(s, Event{Done: true, FinishReason: "abandoned", Err: errCallerGone})
	}
}

// pending is one sequence's participation in a batched decode.
type pending struct {
	s   *sequence
	tok int32
}

// poolAwareErr translates a model-side failure into the vocabulary this package promises.
//
// The block pool's exhaustion is the one cause a caller must react to differently -- 503 rather
// than 500, because retrying after another sequence releases its blocks succeeds -- so it is the
// one cause given a sentinel of this package's own. Everything else passes through unchanged: it
// is a model-side failure whose message is the whole answer.
//
// The result matches errors.Is against both sentinels, and that is not decoration. A caller of
// this package should be able to answer "is this transient?" from ErrPoolExhausted alone, without
// importing internal/llama; a caller that already knows the storage layer keeps working too.
// Wrapping with two %w verbs would give both, and it was the first attempt -- but it also puts
// both messages in the text, so the client read "engine: block pool exhausted: llama: block pool
// exhausted (4 blocks of 4 positions)". The sentence a user sees is the storage layer's, with the
// geometry an operator needs; the engine's sentinel is a way to recognise it, not more prose.
type poolExhaustedError struct{ cause error }

func (e *poolExhaustedError) Error() string { return e.cause.Error() }
func (e *poolExhaustedError) Unwrap() []error {
	return []error{ErrPoolExhausted, e.cause}
}

// poolAwareErr maps a model failure onto that shape.
//
// It is unexported and stays that way. A version of this round exported the wrapper's constructor
// so a test in cmd/orbix could build the translated shape by hand; that is API grown for a test,
// and the assertion it enabled -- "the cause survives the explanation" -- belongs in this package,
// where the real error is produced, not in a caller that only ever sees the result.
//
// TestPoolExhaustionReachesTheCallerAsTheSentinelsShape covers the branch that translates. The
// other, `return err`, has no test and is not worth constructing one for: every reachable failure
// out of PrefillWithPrefix besides exhaustion is a cache or shape error the caller cannot provoke,
// and the startPos refusal needs a caller that prefills from a position other than zero, which
// this engine never does.
func poolAwareErr(err error) error {
	if errors.Is(err, llama.ErrPoolExhausted) {
		return &poolExhaustedError{cause: err}
	}
	return err
}

// prefill processes a sequence's remaining prompt in one batched pass.
//
// pos doubles as the prompt index here because a sequence prefills before it
// decodes anything: while pos < len(Prompt) the cache holds exactly the prompt
// prefix, which is the invariant Prefill requires. The slices are still taken from
// pos so a later, resumable prefill keeps working, but prefix reuse itself only
// applies to a whole prompt from position zero.
func (e *Engine) prefill(s *sequence) {
	if s.traceOn && s.started.IsZero() {
		s.started = time.Now()
	}
	out, err := e.model.PrefillWithPrefix(s.cache, s.req.Prompt[s.pos:], s.pos)
	if err != nil {
		err = poolAwareErr(err)
		// Done is what closes the sequence for the caller: every other finish in this file sets
		// it, and this one did not, so a prefill failure -- a block pool with too few blocks is
		// the reachable case -- closed the channel without ever saying the sequence had ended.
		// A caller looping over the channel saw the tokens that followed and no finish_reason,
		// which a server turns into a response that looks truncated rather than failed.
		e.finish(s, Event{Done: true, FinishReason: "error", Err: err})
		return
	}
	if s.traceOn {
		s.prefilled = time.Now()
	}
	// Record how much of this prompt came from a shared prefix rather than a forward
	// pass. Only the paged cache can reuse, so the contiguous layout reports nothing --
	// which is why the counter lives on the engine rather than on the Cache interface.
	if paged, ok := s.cache.(*llama.PagedKVCache); ok {
		e.mu.Lock()
		e.prefixReused += uint64(paged.ReusedPositions())
		e.mu.Unlock()
	}
	// Prefill returns the logits of the last position: decoding continues from
	// exactly there.
	copy(s.logits, out)
	s.pos = len(s.req.Prompt)
}

// sample picks a sequence's next token and emits it. It reports false when the
// sequence finished this round and must not be decoded.
//
// Every decision here is taken from this sequence's own logits copy: the model
// holds one scratch buffer, so a sequence's logits must be captured before the
// batch runs.
func (e *Engine) sample(s *sequence, eos int32) (int32, bool) {
	cfg := e.model.Config()

	if s.cancelled() {
		e.finish(s, Event{Done: true, FinishReason: "cancel"})
		return 0, false
	}
	if len(s.out) >= s.req.MaxTokens {
		e.finish(s, Event{Done: true, FinishReason: "length"})
		return 0, false
	}

	// The repetition penalties are applied here rather than inside the sampler because they need
	// the tokens already generated, and Sampler.Next sees only a row of logits. It is the same
	// reason logit_bias is a wrapper and these are not.
	applyPenalties(s.logits, &s.req, s.out)
	tok := s.sampler.Next(s.logits)
	if tok == eos || contains(s.req.Stop, tok) {
		e.finish(s, Event{Done: true, FinishReason: "stop"})
		return 0, false
	}
	// The window is checked before the token leaves, not after. A token sampled at
	// position ContextLen would be one the model cannot then decode: appendKV refuses that
	// position, so sample returns false and the token is never fed back -- it is a
	// prediction about a position outside the window, delivered to the caller and counted
	// in TokensGenerated as though it were part of the sequence. Measured on a 16-position
	// fixture, the boundary used to produce room+1 tokens at every prompt length.
	//
	// Checking here also keeps the reported budget honest: cmd/orbix prints "the window cut
	// the budget to N", and N has to be the number of tokens the window actually allows.
	if s.pos >= cfg.ContextLen {
		e.finish(s, Event{Done: true, FinishReason: "length"})
		return 0, false
	}
	s.out = append(s.out, tok)
	// Count the token only once the caller has it. Stats documents TokensGenerated as "the
	// tokens a caller was told about", and emit refuses when the buffer is full and nobody
	// reads -- so counting before the send reported tokens that never left the engine.
	// Measured against a run pumped to exhaustion without a reader: 64 delivered and 65
	// counted, and the gap grew with concurrency (4 sequences gave 256 and 260) because each
	// abandoned sequence loses exactly the one token that filled its buffer.
	//
	// Counting at the forward pass instead would over-count a batched round, which is the
	// other direction and equally wrong; counting after the send is the only point where the
	// number means what the field says.
	// The distribution is described from the same row the sampler just drew on, before the
	// event goes out. It is computed here rather than inside the sampler so that a request
	// without logprobs costs nothing: the sampling path is untouched, and this branch is the
	// only thing that changes.
	ev := Event{Token: tok, HasToken: true, Round: s.round}
	if s.req.Logprobs {
		// Exactly the alternatives asked for, not one more: OpenAI describes top_logprobs as
		// the most likely tokens at the position, so a sampler that drew a surprising token
		// correctly leaves it out of that list. The chosen token's own value is reported
		// separately, which is why the two can cover different sets.
		lp, alts := TokenLogprob(s.logits, tok, s.req.Temperature, s.req.TopLogprobs)
		ev.Logprob, ev.HasLogprob, ev.Top = lp, true, alts
	}
	if !s.emit(ev) {
		return 0, false
	}
	e.mu.Lock()
	e.tokensGenerated++
	e.mu.Unlock()

	return tok, true
}

// decode advances every pending sequence by one token in a single batched pass.
func (e *Engine) decode(batch []pending) {
	caches := make([]llama.Cache, len(batch))
	tokens := make([]int32, len(batch))
	for i, p := range batch {
		caches[i] = p.s.cache
		tokens[i] = p.tok
	}

	out, err := e.model.DecodeStep(caches, tokens)
	if err != nil {
		err = poolAwareErr(err)
		// The whole round failed. Each sequence's cache may or may not have
		// advanced, so decoding cannot continue for any of them: finish them all
		// with the error rather than leaving some in the active set with a cache
		// that no longer matches its position.
		for _, p := range batch {
			// Done for the same reason as the prefill failure above: without it the channel
			// closes without a finish_reason, and a caller cannot tell a truncated answer from
			// a failed one.
			e.finish(p.s, Event{Done: true, FinishReason: "error", Err: err})
		}
		return
	}

	cfg := e.model.Config()
	for i, p := range batch {
		copy(p.s.logits, out[i*cfg.VocabSize:(i+1)*cfg.VocabSize])
		p.s.pos++
		if len(p.s.out) >= p.s.req.MaxTokens {
			e.finish(p.s, Event{Done: true, FinishReason: "length"})
		}
	}
}

// emit delivers one event without blocking.
//
// A blocking send looks correct in a test that always drains the channel and stalls the
// whole engine in production, where a caller can stop reading: the HTTP handler returns as
// soon as the response fails to write, and a disconnected client means nothing drains the
// rest. The buffer absorbs a burst, and once it is full the sequence is abandoned rather
// than waited on. Pump is a shared loop -- one stalled sequence stops every other request
// from advancing -- so it must never block on a per-request channel.
//
// Returning false means the events were not delivered and the sequence is finished with.
//
// Only the goroutine driving Pump may call this. Its abandoned flag is written here and
// read by reapAbandoned in the same round with no lock between them, which is safe exactly
// because both happen on that one goroutine -- the same single-goroutine rule Pump
// documents. A caller that emitted from an HTTP handler would introduce a data race on one
// bool and a much worse bug: two goroutines could both decide a sequence is finished and
// close its channel twice.
func (s *sequence) emit(ev Event) bool {
	if s.abandoned {
		return false
	}
	select {
	case s.events <- ev:
		return true
	default:
	}
	// The buffer is full, which is not proof the caller is gone: a pump running flat out
	// can fill 64 slots before a reading goroutine is scheduled even once.
	//
	// The wait is bounded whether or not there is a cancel channel. A cancel channel says the
	// caller can announce that it left; it does not say it will, and a caller that keeps the
	// connection open while reading nothing never fires it. That case reached this select and
	// blocked the shared pump forever: the buffer could not drain because the reader was not
	// reading, and the cancel could not fire because the socket was still open. Measured with
	// a 1000-position fixture, MaxTokens 900 against an undrained channel and an un-fired
	// cancel: Pump was still inside the round after five seconds.
	//
	// A socket whose peer stops reading is exactly that shape -- TCP holds the connection and
	// r.Context() stays live -- so the branch that trusted the cancel signal was the one a slow
	// client takes. Both paths now share the same deadline, and the cancel is an early exit
	// rather than the only exit.
	//
	// The cancel arm of that select is the one branch here without a test. Measured at four delay
	// lengths against the full-buffer fixture: at 200ms and 500ms the buffer never fills, and at 3s
	// the grace deadline fires first; at 1s the select is entered but the arm that wins is the
	// deadline, because a cancel closed between two sends is only seen by the next one. Getting the
	// two to coincide reliably means racing a scheduler, and a test that sometimes covers a branch is
	// worse than one that says it does not.
	timer := time.NewTimer(abandonGrace)
	defer timer.Stop()
	if s.req.Cancel != nil {
		select {
		case s.events <- ev:
			return true
		case <-s.req.Cancel:
			s.abandoned = true
			return false
		case <-timer.C:
			s.abandoned = true
			return false
		}
	}
	select {
	case s.events <- ev:
		return true
	case <-timer.C:
		s.abandoned = true
		return false
	}
}

// abandonGrace is how long a send waits for a full event buffer to drain before deciding
// the caller is gone. It applies to every request, with or without a cancel channel: the
// channel lets a caller leave early, it does not guarantee the caller leaves at all.
//
// Large enough that a reading caller always wins the race, small enough that a pump with
// several stalled sequences does not look hung.
//
// The comment here used to say the bound was "for callers of the engine API directly",
// because the HTTP server always supplies a cancel channel. That reasoning is what hid the
// bug: a server request whose peer stops reading keeps its context alive, so the cancel never
// fires and the send had nothing to wait on. The server is the caller that needed the bound.
const abandonGrace = 2 * time.Second

// cancelled reports whether the sequence's cancel channel is closed.
//
// A nil channel is never ready, so a request without Cancel never cancels.
func (s *sequence) cancelled() bool {
	if s.req.Cancel == nil {
		return false
	}
	select {
	case <-s.req.Cancel:
		return true
	default:
		return false
	}
}

// classifyError maps an error to the low-cardinality class that goes in error.type.
//
// The conventions require error.type to be predictable and low cardinality, so this returns one
// of a small fixed set rather than the error's message. Unwrapping is deliberate: a wrapped
// error's own type is usually the wrapper (*fmt.wrapError), which classifies nothing.
func classifyError(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, llama.ErrPoolExhausted):
		return "pool_exhausted"
	case errors.Is(err, errCallerGone):
		return "caller_gone"
	}
	// Anything else is a model-side failure: a prefill or decode that could not run. The
	// distinct causes are many and all mean the same thing to a reader of the trace.
	return "inference_failed"
}

// finish closes a sequence's channel and drops it from the active set.
//
// A paged sequence returns its blocks here. Released blocks go back to the pool
// the moment the sequence stops advancing, not when the caller drains its channel,
// because holding them would let a finished request starve one that is waiting.
func (e *Engine) finish(s *sequence, last Event) {
	e.mu.Lock()
	delete(e.active, s.id)
	e.requestsFinished++
	// Split by reason in the same critical section as the total, so the parts always sum to
	// it. reasonIndex only misses a reason finish is never called with, which would be a bug
	// here: leaving it out of the split is honest, and guessing a bucket would not be.
	if idx, ok := reasonIndex(last.FinishReason); ok {
		e.finishedByReason[idx]++
	}
	e.mu.Unlock()

	// Finish the request's span before the terminal event, so a caller that observes
	// Done and immediately scrapes the collector finds the span rather than racing it.
	if s.traceOn && s.req.Trace != nil {
		span := s.reqSpan
		span.Attributes = append(span.Attributes,
			otel.IntAttr("gen_ai.usage.output_tokens", int64(len(s.out))),
			// The convention names this in the plural and types it string[], one entry per
			// choice. A scalar under the singular name is invisible to a backend querying
			// the documented attribute.
			otel.StringsAttr("gen_ai.response.finish_reasons", last.FinishReason),
		)
		// A cancelled request is an error on the span even though it is not an error
		// event: the caller went away before the model finished, and OTel's GenAI
		// conventions map a cancelled generation to error. The event stream keeps
		// reporting it as a plain Done with finish_reason "cancel", because the caller
		// that would receive an error is the one that left.
		spanErr := last.Err
		if spanErr == nil && last.FinishReason == "cancel" {
			spanErr = context.Canceled
		}
		// error.type is Conditionally Required: present exactly when the operation failed, and
		// absent when it did not, so a backend can filter failures by its presence. The value
		// is a low-cardinality class rather than the error message, which would be unbounded
		// -- one series per distinct prompt length.
		if spanErr != nil {
			span.Attributes = append(span.Attributes,
				otel.StringAttr("error.type", classifyError(spanErr)))
		}
		// Finish first, so the span has an end time to measure the decode phase against.
		span = span.Finish(spanErr)
		// Three phases, because the span starts at Submit and the scheduler may not pick the
		// sequence up for a while. Folding that wait into prefill would make a saturated
		// server look like a slow model, which is the opposite of what these numbers are for.
		//
		// A request that never began -- cancelled while queued, or failed before its first
		// pass -- reports its whole lifetime as queue time rather than reporting nothing:
		// those are the requests whose wait matters most.
		began := s.started
		if began.IsZero() {
			began = span.End
		}
		// Milliseconds as integers: a trace viewer can plot them and a reader can compare
		// them against the bench suite's numbers.
		span.Attributes = append(span.Attributes,
			otel.IntAttr("orbix.queue_ms", began.Sub(span.Start).Milliseconds()))
		if !s.prefilled.IsZero() {
			span.Attributes = append(span.Attributes,
				otel.IntAttr("orbix.prefill_ms", s.prefilled.Sub(began).Milliseconds()),
				otel.IntAttr("orbix.decode_ms", span.End.Sub(s.prefilled).Milliseconds()),
			)
		}
		s.req.Trace.Record(span)
		s.traceOn = false
	}

	if err := s.cache.Reset(); err != nil {
		// A cache that cannot release its blocks is a pool bug, not a request
		// failure: report it rather than leaving the memory unrecoverable.
		if last.Err == nil {
			last.Err = err
			last.FinishReason = "error"
		}
	}
	if last.Err != nil || last.Done {
		// Every path out of the sequence comes through here, so this is where the total
		// belongs: filling it at the six finish call sites would be six chances to forget.
		last.Generated = len(s.out)
		// Best effort, like every other event: the caller may have stopped reading, and
		// this call must not be the thing that stalls the shared pump.
		s.emit(last)
	}
	close(s.events)
}

// Active returns how many sequences have been accepted and have not finished, queued ones
// included.
//
// Not "currently running": a sequence joins the map when Submit accepts it and leaves when it
// finishes, so with MaxActive 1 and three requests submitted before any Pump, Active is 3.
// Measured. What is running at this instant is not exposed -- a round advances at most
// MaxActive of them, and which ones is a scheduling decision no caller acts on.
// TestMaxActiveQueuesExcessRequests asserts the counting rule with five requests against
// MaxActive 2.
func (e *Engine) Active() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.active)
}

// MaxActive returns the configured concurrency cap.
//
// No lock: the field is set in New and never written again, which is also why Pump and
// Submit read it without one. The neighbouring accessors that do lock (Round, Stats)
// read counters that change every round. Keeping the distinction visible is what stops
// a later setter from turning this into a race -- a mutable field read here would need
// the lock, and nothing in the call sites would look wrong.
func (e *Engine) MaxActive() int { return e.maxActive }

// Round returns how many Pump calls have run.
//
// It is a scheduling clock rather than a metric: a test that measured request latency
// in wall-clock time would be flaky on a loaded machine, while a round count is exact.
func (e *Engine) Round() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.round
}

func contains(ids []int32, v int32) bool {
	for _, x := range ids {
		if x == v {
			return true
		}
	}
	return false
}

func sortByID(seqs []*sequence) {
	// Insertion sort: the slice is at most MaxActive long, so this is simpler
	// than pulling in slices.SortFunc for an unexported type.
	for i := 1; i < len(seqs); i++ {
		for j := i; j > 0 && seqs[j].id < seqs[j-1].id; j-- {
			seqs[j], seqs[j-1] = seqs[j-1], seqs[j]
		}
	}
}

// applyPenalties applies OpenAI's presence and frequency penalties to a logits row in place.
//
// Presence subtracts a flat amount from every token that has already appeared; frequency subtracts
// an amount proportional to how many times it appeared. Both count the prompt as well as the
// generated tokens, which is what makes a request that repeats a word already in the prompt push
// away from it.
//
// The signs are OpenAI's: a positive value discourages repetition. Negative values encourage it,
// and are legal -- the documented range is [-2, 2] at both ends.
//
// Counting is done per call rather than incrementally. The alternative is a map on the sequence
// that every sample updates, which is faster and one more piece of state to keep right across a
// cancel and a resume; a row of 32k logits is already being walked by the sampler, and this adds
// one pass over the generated run, which is bounded by the token budget.
func applyPenalties(logits []float32, req *Request, out []int32) {
	if req.PresencePenalty == 0 && req.FrequencyPenalty == 0 {
		return
	}
	// The counts of generated tokens, then the prompt's, because every prompt token is seen once.
	// A token counts once per occurrence for the frequency penalty and once in total for the
	// presence penalty.
	seen := make(map[int32]int32, len(out)+16)
	for _, tok := range req.Prompt {
		seen[tok] = 1
	}
	for _, tok := range out {
		seen[tok]++
	}
	for tok, n := range seen {
		if tok < 0 || int(tok) >= len(logits) {
			continue
		}
		var penalty float32
		if req.PresencePenalty != 0 {
			penalty += req.PresencePenalty
		}
		if req.FrequencyPenalty != 0 {
			// n is occurrences including the prompt's single one, and the frequency penalty
			// scales with that count rather than with the count after the first.
			penalty += req.FrequencyPenalty * float32(n)
		}
		logits[tok] -= penalty
	}
}
