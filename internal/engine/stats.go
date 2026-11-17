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

	// TokensGenerated counts the tokens a caller was told about, which is what the
	// throughput they experience is measured in. It is not the model's forward passes:
	// a batched round advances several sequences in one pass, so this number stays
	// larger than the pass count.
	TokensGenerated uint64

	// Rounds counts Pump calls that did work.
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

// FractionFull is what share of the pool is in use, or zero without a pool.
func (s Stats) FractionFull() float64 {
	if s.PoolBlocks == 0 {
		return 0
	}
	return float64(s.PoolBlocks-s.PoolFree) / float64(s.PoolBlocks)
}

// Stats returns a snapshot of the engine's counters and gauges.
//
// The whole snapshot is taken under one lock, so the numbers describe a single instant.
// Reading them piecemeal would let Active and RequestsFinished disagree about a request
// that finished in between, which is the kind of inconsistency that makes a dashboard
// show a queue that never existed.
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
