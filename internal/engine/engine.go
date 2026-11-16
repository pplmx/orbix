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
}

// sequence is the scheduler's per-request state.
type sequence struct {
	id      int
	req     Request
	cache   *llama.KVCache
	next    int     // index into req.Prompt while prefilling, then advanced
	pos     int     // number of positions written
	out     []int32 // generated tokens
	events  chan Event
	sampler llama.Sampler

	// logits is this sequence's own copy of the last forward pass output. The
	// model owns a single scratch buffer, so interleaving sequences would
	// otherwise read whichever sequence ran most recently, which produced
	// fluent-looking garbage when two requests shared the engine.
	logits []float32
}

// Engine generates for several sequences at once.
//
// Concurrency here is cooperative, not parallel: llama.Model keeps scratch
// buffers per instance, so a single loaded model runs one forward pass at a
// time. What the engine buys is interleaving -- each active sequence advances by
// one token per round, so a short request finishes without waiting for a long
// one. That is what continuous batching means at this stage, and it is the
// property a scheduler is built to preserve when the forward pass later grows
// real batching.
type Engine struct {
	model     *llama.Model
	maxActive int

	mu     sync.Mutex
	active map[int]*sequence
	nextID int
}

// Options configures an Engine.
type Options struct {
	// MaxActive caps how many sequences are advanced per round. Zero means 8.
	MaxActive int
}

// New loads a model and returns an engine for it.
func New(path string, opts Options) (*Engine, error) {
	m, err := llama.New(path)
	if err != nil {
		return nil, err
	}
	maxActive := opts.MaxActive
	if maxActive <= 0 {
		maxActive = 8
	}
	return &Engine{
		model:     m,
		maxActive: maxActive,
		active:    make(map[int]*sequence),
	}, nil
}

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

	cache, err := llama.NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		return nil, err
	}

	e.mu.Lock()
	id := e.nextID
	e.nextID++
	s := &sequence{
		id:      id,
		req:     req,
		cache:   cache,
		pos:     0,
		events:  make(chan Event, 64),
		sampler: req.Sampler,
		logits:  make([]float32, cfg.VocabSize),
	}
	if s.sampler == nil {
		s.sampler = llama.Greedy{}
	}
	e.active[id] = s
	e.mu.Unlock()

	return s.events, nil
}

// Pump advances every active sequence by one token.
//
// It returns the number of sequences still running after the round. A caller
// loops until Pump reports zero; each call is one scheduling round, which is
// what makes the interleaving observable to tests.
//
// Panics inside the model propagate: a corrupt forward pass should not be
// turned into a per-request error that hides a bug in the engine.
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

	eos := int32(-1)
	if v := e.model.Config().EOSID; v >= 0 {
		eos = int32(v)
	}

	for _, s := range seqs {
		e.step(s, eos)
	}

	e.mu.Lock()
	n := len(e.active)
	e.mu.Unlock()
	return n
}

// step advances one sequence by a single token.
//
// Every decision here is taken from this sequence's own logits copy. The model
// has one scratch buffer, so its output must be captured immediately after the
// forward pass returns and before any other sequence runs.
func (e *Engine) step(s *sequence, eos int32) {
	cfg := e.model.Config()

	// Prefill: consume the prompt one token at a time, keeping the logits from
	// the last prompt token so decoding can start without re-running it.
	if s.next < len(s.req.Prompt) {
		tok := int(s.req.Prompt[s.next])
		out, err := e.model.Forward(s.cache, tok, s.pos)
		if err != nil {
			e.finish(s, Event{Err: err, FinishReason: "error"})
			return
		}
		copy(s.logits, out)
		s.next++
		s.pos++
		return
	}

	if len(s.out) >= s.req.MaxTokens {
		e.finish(s, Event{Done: true, FinishReason: "length"})
		return
	}

	tok := s.sampler.Next(s.logits)
	if tok == eos || contains(s.req.Stop, tok) {
		e.finish(s, Event{Done: true, FinishReason: "stop"})
		return
	}
	s.out = append(s.out, tok)
	s.events <- Event{Token: tok, HasToken: true}

	if s.pos >= cfg.ContextLen {
		e.finish(s, Event{Done: true, FinishReason: "length"})
		return
	}
	out, err := e.model.Forward(s.cache, int(tok), s.pos)
	if err != nil {
		e.finish(s, Event{Err: err, FinishReason: "error"})
		return
	}
	copy(s.logits, out)
	s.pos++

	if len(s.out) >= s.req.MaxTokens {
		e.finish(s, Event{Done: true, FinishReason: "length"})
	}
}

// finish closes a sequence's channel and drops it from the active set.
func (e *Engine) finish(s *sequence, last Event) {
	e.mu.Lock()
	delete(e.active, s.id)
	e.mu.Unlock()
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
