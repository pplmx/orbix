package engine

import (
	"fmt"
	"sync"

	"github.com/pplmx/orbix/internal/llama"
)

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
	// Cancel, when set, stops the sequence at the start of its next round. The
	// sequence then reports Done with the finish reason "cancel". A caller that
	// owns a connection sets this to its request context so a client that goes
	// away does not keep the model busy: without it, a disconnected stream would
	// generate its whole token budget into a channel nobody reads.
	Cancel <-chan struct{}
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
}

// Options configures an Engine.
type Options struct {
	// MaxActive caps how many sequences are advanced per round. Zero means 8.
	MaxActive int
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
	e := &Engine{
		model:     m,
		maxActive: maxActive,
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
	e.active[id] = s
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
	round := e.round
	e.mu.Unlock()
	// Admit a round's worth. When everything fits, no policy applies and ready is
	// returned unchanged; when it does not, the policy decides who waits.
	seqs = admit(seqs, e.maxActive, e.policy)
	for _, s := range seqs {
		s.lastServed = round
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

	e.mu.Lock()
	n := len(e.active)
	e.mu.Unlock()
	return n
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
	s.events <- Event{Token: tok, HasToken: true}

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

// finish closes a sequence's channel and drops it from the active set.
//
// A paged sequence returns its blocks here. Released blocks go back to the pool
// the moment the sequence stops advancing, not when the caller drains its channel,
// because holding them would let a finished request starve one that is waiting.
func (e *Engine) finish(s *sequence, last Event) {
	e.mu.Lock()
	delete(e.active, s.id)
	e.mu.Unlock()

	if err := s.cache.Reset(); err != nil {
		// A cache that cannot release its blocks is a pool bug, not a request
		// failure: report it rather than leaving the memory unrecoverable.
		if last.Err == nil {
			last.Err = err
			last.FinishReason = "error"
		}
	}
	if last.Err != nil || last.Done {
		s.events <- last
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
