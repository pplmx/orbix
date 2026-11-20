package engine

// Stats is a point-in-time reading of an engine's counters.
//
// The counters are cumulative so a scraper can turn them into rates by differencing two
// samples; the gauges describe the current instant, which cannot be recovered from a
// difference. Both are here rather than in the server because only the engine knows
// what a round is, how much a sequence owes, or when a pool block is free -- a server
// that inferred them would be reporting a guess.
type Stats struct {
	// RequestsSubmitted counts every accepted request. It is incremented in Submit, so a
	// request rejected for an empty prompt or an oversized prompt does not appear: it
	// never entered the queue, and counting it would make the admission rate wrong.
	RequestsSubmitted uint64

	// RequestsFinished counts every request that reached its end, whatever the reason.
	// It is incremented in finish, so it includes cancellations and errors.
	RequestsFinished uint64

	// FinishedByReason splits RequestsFinished by why each request ended, so an operator
	// can tell a healthy server from one whose clients keep leaving. A cancel and an
	// abandoned sequence both look like a finished request in the total, and both are
	// invisible in latency until they dominate: a client that disconnects after its first
	// token still counts as a served request and still contributes a healthy-looking
	// latency, while the server computes a generation nobody reads.
	//
	// Indexed by FinishReason rather than keyed by string: the set is closed, known at
	// compile time, and a map would put an allocation on the finish path.
	FinishedByReason [FinishReasonCount]uint64

	// TokensGenerated counts the tokens a caller was told about, which is what the
	// throughput they experience is measured in. It is not the model's forward passes:
	// a batched round advances several sequences in one pass, so this number stays
	// larger than the pass count.
	TokensGenerated uint64

	// Rounds counts Pump calls that had a sequence to advance at the moment the round
	// began. A call that finds the queue empty and the active set empty is not counted, so
	// polling an idle engine does not inflate the throughput reading.
	//
	// The boundary is not the same as "produced a token": a sequence that finished inside a
	// round is still active at the top of the next one, which counts itself and then finds
	// nothing to do. So this number can be one higher than the count of rounds that emitted
	// output, and it is a round rate rather than a completion rate.
	Rounds uint64

	// Active is how many sequences are being advanced right now.
	Active int

	// Queued is how many accepted requests are waiting for a slot, which is Active minus
	// what a round can hold. It is the number to watch when latency rises: a growing
	// queue with a steady round rate is a capacity problem, not a slow model.
	Queued int

	// MaxActive echoes the round limit, so a dashboard can plot the queue against the
	// capacity that creates it.
	MaxActive int

	// PrefixReused counts prompt positions that came from a shared prefix instead of a
	// forward pass. It is the measure of what prefix caching bought: throughput alone
	// cannot show it, because a scenario that submits its requests together has nothing
	// to reuse.
	PrefixReused uint64

	// PoolBlocks and PoolFree describe the shared block pool, or zero when the engine
	// uses the contiguous layout.
	//
	// Free blocks trending to zero is the signal that a paged deployment is about to
	// start refusing requests, and it is invisible in latency until it is already
	// failing -- which is the whole reason to export it.
	PoolBlocks int
	PoolFree   int
	// PoolBlockSize is how many positions one block holds.
	PoolBlockSize int
}

// FinishReason identifies why a sequence ended. The values are the ones finish is called
// with; anything else is a bug, and ReasonIndex keeps it out of the array rather than
// letting it panic.
type FinishReason int

// The reasons a request can end, in the order they are array-indexed.
const (
	FinishStop FinishReason = iota
	FinishLength
	FinishCancel
	FinishAbandoned
	FinishError

	// FinishReasonCount is one past the last reason. An array of this size has a slot for
	// every value above.
	FinishReasonCount
)

// FinishReasons is every reason a request can end, in array-index order.
//
// Exported so a consumer that wants to report all of them does not restate the list: a
// hardcoded set elsewhere would keep compiling when a new reason was added, and the failure
// would be a metric label that no traffic ever reaches rather than a build error.
//
// Each call returns a fresh slice, so a caller may sort or modify it. The allocation is on the
// stack in practice -- measured at 0 B/op and 0 allocs/op for a 5-element result, and the
// function inlines -- which is why there is no cached package-level slice to hand out instead:
// sharing one would make the isolation a documentation problem rather than a property.
func FinishReasons() []FinishReason {
	out := make([]FinishReason, 0, FinishReasonCount)
	for r := FinishStop; r < FinishReasonCount; r++ {
		out = append(out, r)
	}
	return out
}

// String returns the name the event stream and the HTTP response already use, so a metric
// label and a finish_reason field cannot drift apart.
func (r FinishReason) String() string {
	switch r {
	case FinishStop:
		return "stop"
	case FinishLength:
		return "length"
	case FinishCancel:
		return "cancel"
	case FinishAbandoned:
		return "abandoned"
	case FinishError:
		return "error"
	}
	return "unknown"
}

// reasonIndex maps the string finish is given to its array index.
//
// finish receives the reason as the string the event carries rather than as the enum,
// because those strings are the contract with the caller: they appear in the event stream
// and, for the ones OpenAI defines, in the response body. Converting once here keeps the
// two representations tied together.
func reasonIndex(reason string) (FinishReason, bool) {
	switch reason {
	case "stop":
		return FinishStop, true
	case "length":
		return FinishLength, true
	case "cancel":
		return FinishCancel, true
	case "abandoned":
		return FinishAbandoned, true
	case "error":
		return FinishError, true
	}
	return 0, false
}

// FractionFull is what share of the pool is in use, or zero without a pool.
func (s Stats) FractionFull() float64 {
	if s.PoolBlocks == 0 {
		return 0
	}
	return float64(s.PoolBlocks-s.PoolFree) / float64(s.PoolBlocks)
}

// Stats returns a snapshot of the engine's counters and gauges.
//
// The engine's own numbers are taken under one lock, so Active and RequestsFinished agree
// about every request: reading them piecemeal would let the two disagree about one that
// finished in between, which is the kind of inconsistency that makes a dashboard show a
// queue that never existed.
//
// The pool fields are not in that snapshot. They come from BlockPool.Free, which takes the
// pool's own mutex, and the pump changes the pool after releasing e.mu -- so PoolFree can
// describe a slightly later instant than Active does. Both are consistent within themselves
// and neither can tear, and the pair is not read as a single fact: a deployment watching
// PoolFree is watching it trend, not comparing it to the active count. Making the two
// atomic would mean teaching the engine to hold the pool's lock, which is the coupling the
// two types exist to avoid.
func (e *Engine) Stats() Stats {
	e.mu.Lock()
	defer e.mu.Unlock()

	// A sequence is "queued" when it is accepted but the round cannot hold it: with
	// MaxActive slots and more sequences than that, the excess waits. This is derived
	// rather than tracked because the admission policy re-ranks every round, so there is
	// no stable per-sequence queue position to store.
	queued := max(0, len(e.active)-e.maxActive)

	st := Stats{
		RequestsSubmitted: e.requestsSubmitted,
		RequestsFinished:  e.requestsFinished,
		FinishedByReason:  e.finishedByReason,
		TokensGenerated:   e.tokensGenerated,
		Rounds:            e.rounds,
		Active:            len(e.active),
		Queued:            queued,
		MaxActive:         e.maxActive,
		PrefixReused:      e.prefixReused,
	}
	if e.pool != nil {
		st.PoolBlocks = e.pool.Blocks()
		st.PoolFree = e.pool.Free()
		st.PoolBlockSize = e.pool.BlockSize()
	}
	return st
}
