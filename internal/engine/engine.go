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
	Sampler llama.Sampler
	// Operation names the GenAI operation for the trace span: "chat" for a chat
	// completion, "text_completion" for the legacy completions endpoint. Empty means
	// "chat". It is separate from the prompt because the two endpoints differ only in how
	// the prompt was rendered, and a trace reader distinguishes them by this attribute.
	Operation string
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

	// Trace, when non-nil, records a span for this request and for each round that
	// advances it, so a slow answer can be attributed to prefill, queueing, or decode
	// rather than guessed at.
	//
	// It is per-request rather than per-engine because a caller may trace some traffic
	// and not other traffic, and because an engine shared by an untraced caller must not
	// pay for a tracer it never asked for. The exporter's own lifetime belongs to the
	// caller.
	Trace *otel.Exporter

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
	sampler llama.Sampler

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
	tokensGenerated   uint64
	rounds            uint64
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
func (e *Engine) Pool() *llama.BlockPool { return e.pool }

// Model exposes the loaded model, for configuration and statistics.
func (e *Engine) Model() *llama.Model { return e.model }

// Submit queues a request and returns the channel its events arrive on.
//
// The channel is closed after the final event, so a caller can range over it.
// Generation happens when Pump is called; Submit alone does no work.
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
	e.mu.Lock()
	queued := len(e.active) - e.maxActive
	maxQueued := e.maxQueued
	e.mu.Unlock()
	if queued >= maxQueued {
		return nil, fmt.Errorf("engine: %d requests already waiting for %d slots: %w",
			queued, e.maxActive, ErrQueueFull)
	}

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
		// The span name is the operation name, which is what the convention says to use.
		if span, ok := tracer.StartTrace(op, 2, attrs...); ok {
			s.reqSpan, s.traceOn = span, true
		}
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

// prefill processes a sequence's remaining prompt in one batched pass.
//
// pos doubles as the prompt index here because a sequence prefills before it
// decodes anything: while pos < len(Prompt) the cache holds exactly the prompt
// prefix, which is the invariant Prefill requires. The slices are still taken from
// pos so a later, resumable prefill keeps working, but prefix reuse itself only
// applies to a whole prompt from position zero.
func (e *Engine) prefill(s *sequence) {
	out, err := e.model.PrefillWithPrefix(s.cache, s.req.Prompt[s.pos:], s.pos)
	if err != nil {
		e.finish(s, Event{Err: err, FinishReason: "error"})
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

	tok := s.sampler.Next(s.logits)
	if tok == eos || contains(s.req.Stop, tok) {
		e.finish(s, Event{Done: true, FinishReason: "stop"})
		return 0, false
	}
	s.out = append(s.out, tok)
	// Count the token as soon as it is queued for the caller, which is the moment its
	// throughput is committed to; counting at the forward pass would over-count a
	// batched round.
	e.mu.Lock()
	e.tokensGenerated++
	e.mu.Unlock()
	if !s.emit(Event{Token: tok, HasToken: true, Round: s.round}) {
		return 0, false
	}

	if s.pos >= cfg.ContextLen {
		e.finish(s, Event{Done: true, FinishReason: "length"})
		return 0, false
	}
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
		// The whole round failed. Each sequence's cache may or may not have
		// advanced, so decoding cannot continue for any of them: finish them all
		// with the error rather than leaving some in the active set with a cache
		// that no longer matches its position.
		for _, p := range batch {
			e.finish(p.s, Event{Err: err, FinishReason: "error"})
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
	// A cancelled request is a definite goodbye and lets the send give up at once. Without
	// a cancel channel there is no signal to wait on, so the wait is bounded instead: a
	// caller that is merely behind drains within that window, and one that has left for
	// good is abandoned rather than stalling the shared pump.
	wait := abandonGrace
	if s.req.Cancel != nil {
		select {
		case s.events <- ev:
			return true
		case <-s.req.Cancel:
			s.abandoned = true
			return false
		}
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case s.events <- ev:
		return true
	case <-timer.C:
		s.abandoned = true
		return false
	}
}

// abandonGrace is how long a send waits for a full event buffer to drain before deciding
// the caller is gone. It only applies to requests with no cancel channel.
//
// Large enough that a reading caller always wins the race, small enough that a pump with
// several stalled sequences does not look hung. The HTTP server always supplies a cancel
// channel, so this bound is for callers of the engine API directly.
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
		if !s.prefilled.IsZero() {
			// Milliseconds as integers: a trace viewer can plot them and a reader can
			// compare them against the bench suite's numbers.
			span.Attributes = append(span.Attributes,
				otel.IntAttr("orbix.prefill_ms", s.prefilled.Sub(span.Start).Milliseconds()),
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
		// Best effort, like every other event: the caller may have stopped reading, and
		// this call must not be the thing that stalls the shared pump.
		s.emit(last)
	}
	close(s.events)
}

// Active returns the number of sequences still being advanced.
func (e *Engine) Active() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.active)
}

// MaxActive returns the configured concurrency cap.
func (e *Engine) MaxActive() int { return e.maxActive }

// Policy returns the admission policy.
func (e *Engine) Policy() Priority { return e.policy }

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
