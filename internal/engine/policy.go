package engine

import (
	"fmt"
	"slices"
)

// Priority orders requests when more are ready than a round can advance.
//
// The engine admits at most MaxActive sequences per round, so something has to decide
// which ones. Without a policy that decision is "lowest id first", which is submission
// order -- and submission order is the worst choice available for latency: a
// three-token request queued behind a three-hundred-token one waits for the long one
// to finish, because the short one is never selected while the long one holds a slot.
//
// Measured on stories15M with MaxActive=1: a 3-token request submitted second began
// generating on round 62 of 64.
type Priority int

// The admission policies.
const (
	// PriorityFIFO advances sequences in submission order. It is the default because
	// it is the easiest to reason about and the easiest to compare against: every
	// other policy is a deliberate departure from it.
	PriorityFIFO Priority = iota

	// PriorityShortest advances the sequences with the least remaining work first.
	//
	// "Remaining work" is the tokens the sequence still has to produce, which is known
	// once the prompt is prefilled and is a good proxy for how long the request will
	// hold its slot. Serving the smallest first is what keeps a queue of mixed-size
	// requests from being dominated by its longest member.
	//
	// TestAdmitShortestPicksLeastRemaining pins this on a queue FIFO would have ordered
	// the other way.
	PriorityShortest

	// PriorityFair advances the sequence that has gone longest without a turn.
	//
	// Unlike the other two, this bounds wait time for every request rather than
	// reordering the queue: each round takes the MaxActive sequences with the oldest
	// last-served round, so no sequence can be passed over more than about
	// len(active)/MaxActive rounds in a row. That is the property "fair" claims, and it
	// is what makes it the better choice for a long-running server where starvation of
	// any request is the failure to avoid. FIFO remains the default here, so that every
	// other policy is a visible departure from submission order rather than a property a
	// reader has to infer; a deployment that cares about worst-case wait sets this one.
	PriorityFair
)

// String names the policy, for the CLI and for error messages.
func (p Priority) String() string {
	switch p {
	case PriorityFIFO:
		return "fifo"
	case PriorityShortest:
		return "shortest"
	case PriorityFair:
		return "fair"
	default:
		return fmt.Sprintf("Priority(%d)", int(p))
	}
}

// ParsePriority reads a policy name.
func ParsePriority(s string) (Priority, error) {
	switch s {
	case "fifo", "":
		return PriorityFIFO, nil
	case "shortest":
		return PriorityShortest, nil
	case "fair":
		return PriorityFair, nil
	default:
		return 0, fmt.Errorf("engine: unknown scheduling policy %q (want fifo, shortest, or fair)", s)
	}
}

// admit chooses which of the ready sequences advance this round.
//
// ready must already be sorted by id, which is what makes a policy's behaviour
// reproducible: a tie under any policy falls back to submission order, so two runs of the
// same workload select the same sequences.
//
// Per-request priority outranks the policy: a higher Request.Priority advances first, and
// the policy only orders requests of equal priority. The two answer different questions --
// priority is the caller's ranking, which the engine cannot infer, while the policy is a
// heuristic over work remaining or wait time, which the caller cannot see.
//
// Every round re-ranks the whole ready set, so this **preempts**. A sequence that has been
// running is not pinned to its slot: if enough higher-priority requests are ready when the
// next round starts, it is simply not admitted, and it resumes when room returns. That is
// the intended reading of "higher advances first" here, and it is worth stating because the
// alternative -- finish what you started, then take the queue in priority order -- is what
// most callers assume a priority field means.
//
// Measured on one slot with stories15M q4_0: a 120-token request takes 553-556 ms alone and
// 756-769 ms when a stream of twelve short high-priority requests arrives during it, 37-39%
// longer over three interleaved runs. Nothing starves as
// long as the higher-priority arrivals are finite, because each round still admits
// maxActive sequences and the preempted one is first among its own priority class -- but a
// high-priority stream that never ends will hold a low-priority request indefinitely.
//
// The result is a subset of ready and never longer than limit.
func admit(ready []*sequence, limit int, policy Priority) []*sequence {
	if limit <= 0 || len(ready) == 0 {
		return nil
	}
	if len(ready) <= limit {
		// Nobody has to wait, so no policy applies. Returning the input unchanged
		// keeps the common case allocation-free and free of reordering.
		return ready
	}

	// Score every ready sequence, then take the winners. Sorting a copy rather than
	// using a heap: the queue is bounded by the number of submitted requests, and the
	// comparison is cheap.
	ranked := slices.Clone(ready)
	slices.SortStableFunc(ranked, func(a, b *sequence) int {
		return compareForPolicy(a, b, policy)
	})
	return ranked[:limit]
}

// compareForPolicy returns a negative number when a should advance before b.
//
// The outermost key is the caller's priority, so the policy only ever reorders requests
// the caller ranked equally. Every policy breaks its own ties by id, so selection is
// deterministic: without that, two requests with the same score would be ordered by
// insertion into a map, and the engine's output would depend on Go's map iteration order.
func compareForPolicy(a, b *sequence, policy Priority) int {
	// Higher priority first, so a larger value sorts earlier.
	if ap, bp := a.req.Priority, b.req.Priority; ap != bp {
		if ap > bp {
			return -1
		}
		return 1
	}
	switch policy {
	case PriorityShortest:
		// Remaining work is measured from the sequence's own budget: a sequence that
		// has already emitted most of its tokens has less left to do.
		ar := a.remaining()
		br := b.remaining()
		if ar != br {
			return ar - br
		}
	case PriorityFair:
		// The sequence served least recently goes first. lastServed starts at -1, so a
		// sequence that has never run outranks one that has.
		if a.lastServed != b.lastServed {
			return a.lastServed - b.lastServed
		}
	}
	// FIFO, and the tie-break for the other two.
	switch {
	case a.id < b.id:
		return -1
	case a.id > b.id:
		return 1
	default:
		return 0
	}
}
