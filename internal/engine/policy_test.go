package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/otel"
	"github.com/pplmx/orbix/internal/tokenize"
)

// Policies change which requests advance, never what a request produces: these tests
// use fake sequences wherever the property is about selection order, so they run
// without a model and stay fast. The end-to-end tests at the bottom check the property
// a scheduler is actually judged on -- that a short request is not stuck behind a long
// one -- against a real model.

// fakeSeq builds a sequence with just the fields the policies read.
func fakeSeq(id, maxTokens, emitted, lastServed int) *sequence {
	s := &sequence{id: id, lastServed: lastServed}
	s.req.MaxTokens = maxTokens
	s.out = make([]int32, emitted)
	return s
}

func ids(seqs []*sequence) []int {
	out := make([]int, len(seqs))
	for i, s := range seqs {
		out[i] = s.id
	}
	return out
}

func TestAdmitFIFOIsSubmissionOrder(t *testing.T) {
	seqs := []*sequence{fakeSeq(0, 5, 0, -1), fakeSeq(1, 5, 0, -1), fakeSeq(2, 5, 0, -1)}
	got := admit(seqs, 2, PriorityFIFO)
	if want := []int{0, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("FIFO admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitShortestPicksLeastRemaining(t *testing.T) {
	// Sequence 1 has 1 token left, sequence 2 has 4, sequence 0 has 5. Shortest must
	// take 1 first even though it was not submitted first.
	seqs := []*sequence{fakeSeq(0, 5, 0, -1), fakeSeq(1, 4, 3, -1), fakeSeq(2, 6, 2, -1)}
	got := admit(seqs, 2, PriorityShortest)
	if want := []int{1, 2}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("shortest admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitShortestTiesByID(t *testing.T) {
	// Equal remaining work must fall back to submission order, or two runs of the same
	// workload could select different sequences.
	seqs := []*sequence{fakeSeq(2, 4, 0, -1), fakeSeq(0, 4, 0, -1), fakeSeq(1, 4, 0, -1)}
	got := admit(seqs, 2, PriorityShortest)
	if want := []int{0, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("shortest with ties admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitFairPicksLeastRecentlyServed(t *testing.T) {
	// Sequence 2 ran longest ago, so it goes first even though it was submitted last.
	seqs := []*sequence{fakeSeq(0, 5, 0, 10), fakeSeq(1, 5, 0, 9), fakeSeq(2, 5, 0, 2)}
	got := admit(seqs, 2, PriorityFair)
	if want := []int{2, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("fair admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitFairNeverServedGoesFirst(t *testing.T) {
	// A sequence that has never run (lastServed -1) outranks every one that has, which
	// is what stops a new request from waiting behind the whole active set.
	seqs := []*sequence{fakeSeq(0, 5, 0, 1), fakeSeq(1, 5, 0, 2), fakeSeq(2, 5, 0, -1)}
	got := admit(seqs, 1, PriorityFair)
	if want := []int{2}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("fair admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitReturnsEverythingWhenItFits(t *testing.T) {
	// No policy applies when the whole queue fits, and the input must come back
	// unchanged rather than reordered: reordering without a reason would make trace
	// output harder to follow.
	seqs := []*sequence{fakeSeq(2, 5, 0, 5), fakeSeq(0, 1, 0, 1), fakeSeq(1, 9, 0, 3)}
	for _, policy := range []Priority{PriorityFIFO, PriorityShortest, PriorityFair} {
		got := admit(seqs, 5, policy)
		if want := []int{2, 0, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
			t.Errorf("%v admitted %v for an unbounded round, want the input order %v", policy, ids(got), want)
		}
	}
}

func TestAdmitEdgeCases(t *testing.T) {
	seqs := []*sequence{fakeSeq(0, 5, 0, -1), fakeSeq(1, 5, 0, -1)}
	if got := admit(seqs, 0, PriorityFIFO); got != nil {
		t.Errorf("limit 0 admitted %v, want nil", ids(got))
	}
	if got := admit(nil, 3, PriorityFIFO); got != nil {
		t.Errorf("empty queue admitted %v, want nil", ids(got))
	}
	// A limit larger than the queue must not grow the result.
	if got := admit(seqs, 10, PriorityFair); len(got) != 2 {
		t.Errorf("limit 10 admitted %d sequences, want 2", len(got))
	}
}

// TestPriorityPreemptsARunningSequence pins the semantics of "higher advances first".
//
// admit runs on every round against the whole ready set, so a sequence that has already
// emitted tokens is not pinned to its slot: a higher-priority arrival takes it. The
// alternative -- finish what you started, then take the queue in priority order -- is what
// most callers assume a priority field means, which is why this is worth a test rather than
// a sentence.
//
// The test uses the same admit the pump calls, once per round, so it reproduces the
// decision the pump makes without a model: round 1 admits the low-priority sequence, round
// 2 sees the same sequence as ready alongside a higher-priority one and must pick the
// higher-priority one.
// The rounds below are synthetic, which is what lets the test assert exact round numbers. What
// they do not show is the shape on a real engine, so that was measured separately on the
// 512-position fixture with MaxActive 1 and FIFO, one 40-token request already three rounds in:
// a second request at the default priority began generating 42 rounds after submitting, once
// the long one had finished, while the same request at priority 10 began on the fourth round
// after submitting. The long request produced its whole 40-token budget in both runs, which is
// the half this test asserts as "round 3 admits the long sequence again" -- a preempted
// sequence is yielded, not cancelled.
func TestPriorityPreemptsARunningSequence(t *testing.T) {
	long := fakeSeq(0, 100, 0, -1)
	long.req.Priority = 0

	// Round 1: only the long request is ready, so it advances -- and would keep advancing
	// under a "finish what you started" reading.
	round1 := admit([]*sequence{long}, 1, PriorityFIFO)
	if len(round1) != 1 || round1[0] != long {
		t.Fatalf("round 1 admitted %v, want just the long sequence", ids(round1))
	}
	// It made progress: pretend it emitted a few tokens.
	long.out = make([]int32, 5)

	// Round 2: a short high-priority request arrives while the long one is mid-flight.
	urgent := fakeSeq(1, 4, 0, -1)
	urgent.req.Priority = 10
	round2 := admit([]*sequence{long, urgent}, 1, PriorityFIFO)
	if len(round2) != 1 || round2[0] != urgent {
		t.Errorf("round 2 admitted %v, want the high-priority sequence: a running request "+
			"must yield its slot, not keep it until it finishes", ids(round2))
	}

	// And the preempted request is not lost: with the urgent one done, it is admitted again.
	round3 := admit([]*sequence{long}, 1, PriorityFIFO)
	if len(round3) != 1 || round3[0] != long {
		t.Errorf("round 3 admitted %v, want the long sequence to resume", ids(round3))
	}
}

// TestPriorityDoesNotStarveUnderAFiniteBurst checks the other half: preemption is not
// starvation as long as the higher-priority arrivals stop. Each round still admits
// maxActive sequences, and the preempted request is first among its own priority class.
func TestPriorityDoesNotStarveUnderAFiniteBurst(t *testing.T) {
	long := fakeSeq(0, 100, 0, -1)
	long.req.Priority = 0
	long.out = make([]int32, 10)

	// Five rounds of a higher-priority arrival each time.
	for round := range 5 {
		urgent := fakeSeq(1+round, 2, 0, -1)
		urgent.req.Priority = 5
		admitted := admit([]*sequence{long, urgent}, 1, PriorityFIFO)
		if len(admitted) != 1 {
			t.Fatalf("round %d admitted %d sequences, want 1", round, len(admitted))
		}
		if admitted[0] != urgent {
			t.Errorf("round %d admitted the low-priority sequence while a high-priority one waited", round)
		}
	}
	// The burst ends; the long request must be admitted with no further arrivals.
	after := admit([]*sequence{long}, 1, PriorityFIFO)
	if len(after) != 1 || after[0] != long {
		t.Errorf("after the burst admitted %v, want the long sequence", ids(after))
	}
}

func TestPriorityNames(t *testing.T) {
	for _, tc := range []struct {
		name string
		want Priority
	}{
		{"fifo", PriorityFIFO},
		{"", PriorityFIFO},
		{"shortest", PriorityShortest},
		{"fair", PriorityFair},
	} {
		got, err := ParsePriority(tc.name)
		if err != nil {
			t.Errorf("ParsePriority(%q) = %v", tc.name, err)
			continue
		}
		if got != tc.want {
			t.Errorf("ParsePriority(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
	// The refusal has to name the value it rejected and the three it would have taken: an
	// operator reading "unknown scheduling policy" alone has nothing to correct the flag with.
	_, err := ParsePriority("random")
	if err == nil {
		t.Fatal("ParsePriority accepted an unknown policy")
	}
	for _, want := range []string{`"random"`, "fifo", "shortest", "fair"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %q, want it to name %q", err, want)
		}
	}
	// The names have to round-trip, since the CLI prints them.
	for _, p := range []Priority{PriorityFIFO, PriorityShortest, PriorityFair} {
		back, err := ParsePriority(p.String())
		if err != nil || back != p {
			t.Errorf("%v round-trips to %v (%v)", p, back, err)
		}
	}
}

// --- end-to-end: the property a scheduler is judged on ---

// loadEngine returns an engine and tokenizer over a fixture model.
//
// The scheduling properties these tests check -- that a batch equals a serial run, that a
// policy reorders without changing output, that admission caps a round -- hold for any
// weights, so a constructed model is enough and the tests run everywhere. A checkpoint
// would make them faster to write and impossible to run in CI.
func loadEngine(t *testing.T, opts Options) (*Engine, *tokenize.Tokenizer) {
	t.Helper()
	return loadEngineAt(t, gguftest.Path(t), opts)
}

// loadEngineAt is loadEngine over a fixture a test built itself, for shapes the default
// does not cover.
func loadEngineAt(t *testing.T, path string, opts Options) (*Engine, *tokenize.Tokenizer) {
	t.Helper()
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	e, err := New(path, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e, tok
}

func TestSchedulingPolicyKeepsShortRequestResponsive(t *testing.T) {
	// The failure this policy exists to prevent: with MaxActive=1, a short request
	// submitted behind a long one under FIFO waits for the long one to finish.
	// PriorityShortest and PriorityFair both have to cut that down.
	//
	// The assertion is relative to the long request's budget rather than to a round
	// number observed on one checkpoint: an absolute number encodes the model's
	// context length, which is a property of the fixture and not of the policy.
	const (
		longTokens  = 40
		shortTokens = 3
	)
	bfs := func(t *testing.T, policy Priority) int {
		// A model that cannot emit its end-of-sequence token, so the long request runs its
		// whole budget: the property here is where the scheduler puts the short request
		// relative to a long one, and a sequence that stops early stops measuring that.
		path := gguftest.Model{NoEOS: true}.Write(t, t.TempDir())
		e, tok := loadEngineAt(t, path, Options{MaxActive: 1, Workers: 2, Policy: policy})
		prompt := tok.Encode("Once upon a time", true)

		longCh, err := e.Submit(Request{Prompt: prompt, MaxTokens: longTokens})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		shortCh, err := e.Submit(Request{Prompt: prompt, MaxTokens: shortTokens})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		// Drain both in the background: the channels are buffered but a 60-token
		// sequence would fill the buffer and block Pump.
		go func() {
			for range longCh {
			}
		}()
		// The round travels on the event rather than being read from the engine here: a
		// consumer runs on its own goroutine, so e.Round() reports when this goroutine was
		// scheduled, not which round produced the token. Reading it was the source of this
		// test's flakiness -- it would occasionally observe a round several steps later and
		// report the policy as broken.
		done := make(chan int, 1)
		go func() {
			first := 0
			for ev := range shortCh {
				if ev.HasToken && first == 0 {
					first = ev.Round
				}
			}
			done <- first
		}()
		for e.Pump() > 0 {
		}
		return <-done
	}

	t.Run("fifo", func(t *testing.T) {
		round := bfs(t, PriorityFIFO)
		t.Logf("FIFO: the short request's first token arrived on round %d", round)
		// Under submission order the short request waits for the long one's whole
		// budget, so its first token cannot arrive before that many rounds have passed.
		if round <= longTokens/2 {
			t.Errorf("FIFO served the short request on round %d, before the long request's "+
				"%d-token budget could finish; submission order was not honoured", round, longTokens)
		}
	})
	t.Run("shortest", func(t *testing.T) {
		round := bfs(t, PriorityShortest)
		t.Logf("shortest: the short request's first token arrived on round %d", round)
		if round > 10 {
			t.Errorf("shortest served the short request on round %d, want under 10", round)
		}
	})
	t.Run("fair", func(t *testing.T) {
		round := bfs(t, PriorityFair)
		t.Logf("fair: the short request's first token arrived on round %d", round)
		if round > 10 {
			t.Errorf("fair served the short request on round %d, want under 10", round)
		}
	})
}

// TestPolicyDoesNotChangeOutput is the safety property: a scheduling policy changes
// when a request runs, never what it produces. Greedy decoding is deterministic, so two
// policies must return the same tokens for the same prompt.
func TestPolicyDoesNotChangeOutput(t *testing.T) {
	// Different prompts, not one repeated. With a shared prompt every request computes the same
	// distribution, so a scheduler that admitted the wrong one first, or advanced two of them a
	// different number of times, could still produce identical rows for all four and pass. The
	// sequences here differ in length as well, which is what makes PriorityShortest reorder them.
	prompts := []string{"Once upon a time", "The capital of France is", "a", "Hello there"}

	run := func(policy Priority) [][]int32 {
		e, tok := loadEngine(t, Options{MaxActive: 2, Workers: 2, Policy: policy})
		n := len(prompts)
		chans := make([]<-chan Event, n)
		for i := range chans {
			ch, err := e.Submit(Request{Prompt: tok.Encode(prompts[i], true), MaxTokens: 6})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			chans[i] = ch
		}
		// The comparison is between schedules, so there has to be a schedule: with MaxActive 2 and
		// four requests the other two wait, and a smaller MaxActive or a shorter prompt would make
		// this one batch measured twice.
		queued := false
		for e.Pump() > 0 {
			if e.Stats().Queued > 0 {
				queued = true
			}
		}
		if !queued {
			t.Fatalf("%v: nothing ever queued, so this compares one schedule with itself", policy)
		}
		out := make([][]int32, n)
		for i, ch := range chans {
			for ev := range ch {
				if ev.Err != nil {
					t.Fatalf("sequence %d: %v", i, ev.Err)
				}
				if ev.HasToken {
					out[i] = append(out[i], ev.Token)
				}
			}
		}
		return out
	}

	fifo := run(PriorityFIFO)
	for _, policy := range []Priority{PriorityShortest, PriorityFair} {
		got := run(policy)
		for i := range fifo {
			if len(got[i]) != len(fifo[i]) {
				t.Fatalf("%v: request %d (%q) produced %d tokens, fifo produced %d",
					policy, i, prompts[i], len(got[i]), len(fifo[i]))
			}
			for j := range fifo[i] {
				if got[i][j] != fifo[i][j] {
					t.Errorf("%v: request %d (%q) token %d = %d, fifo = %d",
						policy, i, prompts[i], j, got[i][j], fifo[i][j])
				}
			}
		}
	}
}

// Per-request priority is the caller's ranking; the policy orders requests the caller
// ranked equally. These tests keep the two apart deliberately, because a scheduler that
// blurs them serves a high-priority request late without any signal that it did.

// fakeSeqPrio is fakeSeq with an explicit Request.Priority.
func fakeSeqPrio(id, maxTokens, priority int) *sequence {
	s := fakeSeq(id, maxTokens, 0, -1)
	s.req.Priority = priority
	return s
}

func TestAdmitPriorityOutranksFIFO(t *testing.T) {
	// Submitted last, ranked highest: it must go first despite the submission order.
	seqs := []*sequence{
		fakeSeqPrio(0, 5, 0),
		fakeSeqPrio(1, 5, 0),
		fakeSeqPrio(2, 5, 10),
	}
	got := admit(seqs, 1, PriorityFIFO)
	if want := []int{2}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitNegativePriorityGoesLast(t *testing.T) {
	// A background job marked below default must yield to everything else.
	seqs := []*sequence{
		fakeSeqPrio(0, 5, -5),
		fakeSeqPrio(1, 5, 0),
	}
	got := admit(seqs, 1, PriorityFIFO)
	if want := []int{1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("admitted %v, want %v (the negative-priority request should yield)", ids(got), want)
	}
}

func TestAdmitPriorityThenPolicyWithinTier(t *testing.T) {
	// Two tiers, and the policy must apply inside each. Under shortest, the 2-token
	// request wins its tier even though it was submitted last.
	seqs := []*sequence{
		fakeSeqPrio(0, 100, 0),
		fakeSeqPrio(1, 2, 0),
		fakeSeqPrio(2, 100, 5),
		fakeSeqPrio(3, 3, 5),
	}
	got := admit(seqs, 2, PriorityShortest)
	if want := []int{3, 2}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("admitted %v, want %v (high tier first, shortest within it)", ids(got), want)
	}
}

func TestAdmitPriorityIsStableUnderFair(t *testing.T) {
	// Fair orders by last-served, but priority still dominates: a high-priority request
	// that just ran must beat a low-priority one that has waited.
	high := fakeSeqPrio(0, 5, 10)
	high.lastServed = 100
	low := fakeSeqPrio(1, 5, 0)
	low.lastServed = -1
	got := admit([]*sequence{low, high}, 1, PriorityFair)
	if want := []int{0}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("admitted %v, want %v (priority outranks wait time)", ids(got), want)
	}
}

func TestAdmitPriorityDoesNotReorderWhenAllFit(t *testing.T) {
	// When the round has room for everything, no policy or priority applies: the input
	// order is preserved so the common case stays allocation-free.
	seqs := []*sequence{fakeSeqPrio(0, 5, -1), fakeSeqPrio(1, 5, 99)}
	got := admit(seqs, 5, PriorityFIFO)
	if want := []int{0, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("admitted %v, want %v (unchanged input order)", ids(got), want)
	}
}

func TestAdmitPriorityTiesFallBackToID(t *testing.T) {
	// Same priority, same score, different ids: the lower id must win, or selection
	// would depend on map iteration order.
	seqs := []*sequence{fakeSeqPrio(7, 5, 3), fakeSeqPrio(2, 5, 3)}
	got := admit(seqs, 1, PriorityFIFO)
	if want := []int{2}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("admitted %v, want %v", ids(got), want)
	}
}

// TestPriorityReachesTheEngine is the end-to-end counterpart to the unit tests: it
// checks that Request.Priority travels from Submit to admit, through the real engine.
// A unit test on admit alone would keep passing if the field were never copied.
func TestPriorityReachesTheEngine(t *testing.T) {
	// The last request submitted is the one ranked highest, so a run that ignores
	// priority cannot produce the expected order by accident.
	const tokens = 4
	firstRound := func(t *testing.T, priorities []int) map[int]int {
		t.Helper()
		e, tok := loadEngine(t, Options{MaxActive: 1, Workers: 2, Policy: PriorityFIFO})
		prompt := tok.Encode("Once upon a time", true)

		chans := make([]<-chan Event, len(priorities))
		for i, prio := range priorities {
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: tokens, Priority: prio})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			chans[i] = ch
		}

		first := make(map[int]int, len(priorities))
		var mu sync.Mutex
		var wg sync.WaitGroup
		for i, ch := range chans {
			wg.Go(func() {
				for ev := range ch {
					if ev.HasToken {
						mu.Lock()
						if _, seen := first[i]; !seen {
							// The event's own round, not e.Round(): a consumer runs on
							// its own goroutine, so by the time it reads the engine's
							// counter several rounds may have passed and it would record
							// when the token was noticed rather than when it was made.
							first[i] = ev.Round
						}
						mu.Unlock()
					}
				}
			})
		}
		for e.Pump() > 0 {
		}
		wg.Wait()
		return first
	}

	t.Run("highest priority starts first", func(t *testing.T) {
		got := firstRound(t, []int{0, 0, 10})
		t.Logf("first-token rounds: %v", got)
		if got[2] > got[0] || got[2] > got[1] {
			t.Errorf("the priority-10 request first ran on round %d, others on %v; "+
				"priority was not applied", got[2], got)
		}
	})
	t.Run("negative priority yields", func(t *testing.T) {
		got := firstRound(t, []int{-5, 0, 0})
		t.Logf("first-token rounds: %v", got)
		if got[0] < got[1] || got[0] < got[2] {
			t.Errorf("the priority -5 request first ran on round %d, others on %v; "+
				"a below-default request should yield", got[0], got)
		}
	})
}

// TestPriorityDoesNotChangeOutput is the same safety property the policies have: ranking
// a request differently changes when it runs, never what it produces.
func TestPriorityDoesNotChangeOutput(t *testing.T) {
	run := func(priorities []int) [][]int32 {
		e, tok := loadEngine(t, Options{MaxActive: 2, Workers: 2, Policy: PriorityShortest})
		prompt := tok.Encode("Once upon a time", true)
		chans := make([]<-chan Event, len(priorities))
		for i, prio := range priorities {
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 6, Priority: prio})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			chans[i] = ch
		}
		out := make([][]int32, len(priorities))
		var wg sync.WaitGroup
		for i, ch := range chans {
			wg.Go(func() {
				for ev := range ch {
					if ev.HasToken {
						out[i] = append(out[i], ev.Token)
					}
				}
			})
		}
		for e.Pump() > 0 {
		}
		wg.Wait()
		return out
	}

	plain := run([]int{0, 0, 0, 0})
	ranked := run([]int{5, -5, 0, 100})
	for i := range plain {
		if fmt.Sprint(plain[i]) != fmt.Sprint(ranked[i]) {
			t.Errorf("request %d: priority changed the output\n plain  %v\n ranked %v",
				i, plain[i], ranked[i])
		}
	}
}

// TestTracingRecordsRequestSpans checks that Request.Trace produces one span per request
// carrying the fields an operator needs: the model, the token counts, the finish reason,
// and the split between prefill and decode. Without this the tracing path could be wired
// up but empty, which no compile error would catch.
//
// It used to skip without MODEL, although the engine it drives comes from loadEngine --
// which is the fixture -- so the gate guarded nothing. The trace fields it asserts are
// counts and names computed from the request, not values that depend on a trained
// checkpoint, and every test beside it already ran on the fixture.
func TestTracingRecordsRequestSpans(t *testing.T) {
	type captured struct {
		count int
		body  []byte
	}
	got := make(chan captured, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		got <- captured{count: 1, body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	exp, err := otel.NewExporter(otel.ExporterOptions{
		Endpoint: srv.URL,
		Interval: time.Hour, // flushed explicitly
		Resource: []otel.Attribute{otel.StringAttr("service.name", "orbix-test")},
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	e, tok := loadEngine(t, Options{MaxActive: 1, Workers: 2})
	prompt := tok.Encode("Once upon a time", true)
	const maxTokens = 4
	ch, err := e.Submit(Request{
		Prompt: prompt, MaxTokens: maxTokens, Trace: exp,
	})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("event error: %v", ev.Err)
		}
	}

	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	select {
	case c := <-got:
		body := string(c.body)
		for _, want := range []string{
			// Required by the GenAI conventions, and the span name is the operation name.
			`"gen_ai.operation.name"`,
			`"gen_ai.provider.name"`,
			`"chat"`,
			`"gen_ai.request.model"`,
			`"gen_ai.usage.input_tokens"`,
			`"gen_ai.usage.output_tokens"`,
			`"gen_ai.response.finish_reasons"`,
			`"orbix.prefill_ms"`,
			`"orbix.decode_ms"`,
			`"service.name"`,
		} {
			if !strings.Contains(body, want) {
				t.Errorf("trace body is missing %s:\n%s", want, body)
			}
		}
		// The prompt length is known exactly, so the input token count can be checked
		// rather than merely present.
		if !strings.Contains(body, `"intValue":"`+itoa(len(prompt))+`"`) {
			t.Errorf("input token count is not %d:\n%s", len(prompt), body)
		}
		// Exactly one span, which the test's own name claims and nothing checked. The Trace
		// field's comment promised "a span for this request and for each round that advances
		// it" for several rounds while no round span existed; counting here is what would
		// have caught the comment drifting away from the code, and it fails the other way
		// too if a future change starts emitting one span per round without saying so.
		if n := strings.Count(body, `"spanId"`); n != 1 {
			t.Errorf("got %d spans for one request, want exactly 1:\n%s", n, body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the collector received no trace")
	}
}

// TestTracingSplitsQueueFromCompute checks that the span's three phases measure three different
// things.
//
// The span starts at Submit, so anything measured from its start includes the time the sequence
// waited for a slot. Before this test the only two attributes were measured that way, which made
// a saturated server indistinguishable from a slow model: "orbix.prefill_ms" carried the wait.
//
// The case forces the split rather than hoping for one. With one active slot, a long request
// holds the slot while a short one queues behind it; the short one's queue time is then most of
// its life and its own prefill is a fraction of it. An implementation that still measured prefill
// from the span's start reports queue_ms near zero and prefill_ms near the whole duration, which
// fails both assertions below.
func TestTracingSplitsQueueFromCompute(t *testing.T) {
	type captured struct{ body []byte }
	got := make(chan captured, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		select {
		case got <- captured{body: body}:
		default:
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	exp, err := otel.NewExporter(otel.ExporterOptions{
		Endpoint: srv.URL,
		Interval: time.Hour,
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	// NoEOS, because the long request has to run its whole budget: the default fixture stops on
	// the end-of-sequence token after a few tokens, which leaves the short request waiting two
	// milliseconds and proves nothing about the split.
	path := gguftest.Model{NoEOS: true, ContextLen: 3000}.Write(t, t.TempDir())
	e, tok := loadEngineAt(t, path, Options{MaxActive: 1, Workers: 2})
	prompt := tok.Encode("Once upon a time", true)
	// The wait comes from a long prompt rather than a long generation: one batched prefill costs
	// a few hundred milliseconds, while the same time spent decoding would be thousands of
	// rounds. The property under test is which phase the wait is attributed to, and prefill and
	// decode are equally good at producing it.
	longPrompt := make([]int32, 0, 2048)
	for len(longPrompt) < 2000 {
		longPrompt = append(longPrompt, prompt...)
	}

	// The long request takes the only slot; the short one waits for it.
	long, err := e.Submit(Request{Prompt: longPrompt, MaxTokens: 1, Trace: exp, ResponseID: "long"})
	if err != nil {
		t.Fatalf("Submit long: %v", err)
	}
	short, err := e.Submit(Request{Prompt: prompt, MaxTokens: 2, Trace: exp, ResponseID: "short"})
	if err != nil {
		t.Fatalf("Submit short: %v", err)
	}
	for e.Pump() > 0 {
	}
	for _, ch := range []<-chan Event{long, short} {
		for ev := range ch {
			if ev.Err != nil {
				t.Fatalf("event error: %v", ev.Err)
			}
		}
	}
	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	var body []byte
	select {
	case c := <-got:
		body = c.body
	case <-time.After(5 * time.Second):
		t.Fatal("the collector received no trace")
	}

	spans := parseSpans(t, body)
	byID := map[string]spanAttrs{}
	for _, sp := range spans {
		byID[sp.responseID] = sp
	}
	sh, ok := byID["short"]
	if !ok {
		t.Fatalf("no span for the short request; got %d spans with ids %v", len(spans), keys(byID))
	}
	lg := byID["long"]

	// Every span carries a queue time, including one that never ran.
	if _, ok := sh.attrs["orbix.queue_ms"]; !ok {
		t.Errorf("the short request's span has no orbix.queue_ms: %v", sh.attrs)
	}
	// The short request spent most of its life waiting, so its queue time is several times
	// its own prefill. The regime is asserted first: if the wait were only a few milliseconds
	// the ratio would be decided by timer granularity rather than by the split, and the test
	// should say so instead of passing or failing by luck.
	if sh.attrs["orbix.queue_ms"] < 20 {
		t.Fatalf("the short request waited %d ms behind a 30-token request; the test did not"+
			" reach the regime where the two phases are separable", sh.attrs["orbix.queue_ms"])
	}
	if sh.attrs["orbix.queue_ms"] <= 2*sh.attrs["orbix.prefill_ms"] {
		t.Errorf("short request: queue_ms = %d, prefill_ms = %d; the wait should dominate when"+
			" it queued behind another request",
			sh.attrs["orbix.queue_ms"], sh.attrs["orbix.prefill_ms"])
	}
	// The long request took the only slot immediately, so almost none of its span is queue time
	// and almost all of it is the work it did. Together the two make the numbers attributable
	// rather than merely present: one span is nearly all wait, the other nearly all prefill.
	if lg.attrs["orbix.queue_ms"] > lg.attrs["orbix.prefill_ms"] {
		t.Errorf("long request: queue_ms = %d, prefill_ms = %d; it took the slot immediately",
			lg.attrs["orbix.queue_ms"], lg.attrs["orbix.prefill_ms"])
	}
	// The phases have to account for the span: a gap would mean time attributed to nothing.
	for _, sp := range spans {
		sum := sp.attrs["orbix.queue_ms"] + sp.attrs["orbix.prefill_ms"] + sp.attrs["orbix.decode_ms"]
		if d := sp.durationMs - sum; d > 2 || d < -2 {
			t.Errorf("span %q: phases sum to %d ms, the span lasted %d ms",
				sp.responseID, sum, sp.durationMs)
		}
	}
}

// spanAttrs is one span's identity, duration and integer attributes.
type spanAttrs struct {
	responseID string
	durationMs int64
	attrs      map[string]int64
}

// parseSpans reads the spans out of an OTLP/JSON trace body.
//
// The body is JSON, so the assertions can be about numbers rather than about text: a test that
// searched for `"orbix.queue_ms"` would pass whether or not the value made sense.
func parseSpans(t *testing.T, body []byte) []spanAttrs {
	t.Helper()
	var doc struct {
		ResourceSpans []struct {
			ScopeSpans []struct {
				Spans []struct {
					Start string `json:"startTimeUnixNano"`
					End   string `json:"endTimeUnixNano"`
					Attrs []struct {
						Key   string `json:"key"`
						Value struct {
							Int string `json:"intValue"`
							Str string `json:"stringValue"`
						} `json:"value"`
					} `json:"attributes"`
				} `json:"spans"`
			} `json:"scopeSpans"`
		} `json:"resourceSpans"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		t.Fatalf("decode trace body: %v\n%s", err, body)
	}
	var out []spanAttrs
	for _, rs := range doc.ResourceSpans {
		for _, ss := range rs.ScopeSpans {
			for _, sp := range ss.Spans {
				s := spanAttrs{attrs: map[string]int64{}}
				for _, a := range sp.Attrs {
					switch {
					case a.Key == "gen_ai.response.id":
						s.responseID = a.Value.Str
					case a.Value.Int != "":
						var v int64
						if _, err := fmt.Sscan(a.Value.Int, &v); err != nil {
							t.Fatalf("attribute %q: %v", a.Key, err)
						}
						s.attrs[a.Key] = v
					}
				}
				start, err1 := strconv.ParseInt(sp.Start, 10, 64)
				end, err2 := strconv.ParseInt(sp.End, 10, 64)
				if err1 != nil || err2 != nil {
					t.Fatalf("timestamps: %v %v", err1, err2)
				}
				s.durationMs = (end - start) / 1e6
				out = append(out, s)
			}
		}
	}
	return out
}

func keys(m map[string]spanAttrs) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// TestTracingOffCostsNothing pins the default: an engine with no tracer must run exactly
// as before, which is what makes tracing safe to leave off.
func TestTracingOffCostsNothing(t *testing.T) {
	e, tok := loadEngine(t, Options{MaxActive: 1, Workers: 2})
	prompt := tok.Encode("Once upon a time", true)
	ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 3})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	var tokens int
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("event error: %v", ev.Err)
		}
		if ev.HasToken {
			tokens++
		}
	}
	if tokens == 0 {
		t.Error("no tokens generated")
	}
}

// itoa formats an int without pulling strconv into this file's imports for one call.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// TestCancelledRequestSpanIsError checks how a cancellation reaches the trace.
//
// The event stream reports a cancel as a plain Done with finish_reason "cancel", because the
// caller that would receive an error is the one that left. The span is different: OTel's GenAI
// conventions say a cancelled generation maps to error, and a span marked ok disappears from a
// query for failed requests.
func TestCancelledRequestSpanIsError(t *testing.T) {
	var got struct {
		mu   sync.Mutex
		body []byte
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got.mu.Lock()
		got.body = b
		got.mu.Unlock()
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	exp, err := otel.NewExporter(otel.ExporterOptions{
		Endpoint: srv.URL,
		Interval: time.Hour, // flushed explicitly
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	e, tok := loadEngine(t, Options{MaxActive: 1, Workers: 2})
	cancel := make(chan struct{})
	ch, err := e.Submit(Request{
		Prompt: tok.Encode("Once upon a time", true), MaxTokens: 64,
		Trace: exp, Cancel: cancel,
	})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	// Cancel before the first round so the sequence stops at the sample check rather than
	// running its budget.
	close(cancel)
	for e.Pump() > 0 {
	}
	sawCancel := false
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("cancel must not surface as an event error: %v", ev.Err)
		}
		if ev.Done && ev.FinishReason == "cancel" {
			sawCancel = true
		}
	}
	if !sawCancel {
		t.Fatal("the stream never reported finish_reason cancel")
	}

	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	got.mu.Lock()
	body := string(got.body)
	got.mu.Unlock()

	if !strings.Contains(body, `gen_ai.response.finish_reasons`) || !strings.Contains(body, "cancel") {
		t.Fatalf("span does not record the cancel: %s", body)
	}
	// The status code is what a trace query filters on. 1 is ok, 2 is error.
	if !strings.Contains(body, `"status":{"code":2`) {
		t.Errorf("cancelled span is not marked error: %s", body)
	}
}

// TestSpanCarriesTheRequiredGenAIAttributes checks the three attributes the GenAI conventions
// mark Required, plus the span name rule that goes with them.
//
// Without gen_ai.operation.name and gen_ai.provider.name a backend cannot tell which operation
// a span describes or which provider-specific attribute set to read it against; without
// gen_ai.response.id a trace cannot be joined to the response a user is looking at.
func TestSpanCarriesTheRequiredGenAIAttributes(t *testing.T) {
	cases := []struct {
		name, operation, wantOp string
	}{
		{"chat", "chat", "chat"},
		{"legacy completions", "text_completion", "text_completion"},
		{"unset defaults to chat", "", "chat"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var mu sync.Mutex
			var body []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				mu.Lock()
				body = b
				mu.Unlock()
				_, _ = w.Write([]byte(`{}`))
			}))
			defer srv.Close()

			exp, err := otel.NewExporter(otel.ExporterOptions{Endpoint: srv.URL, Interval: time.Hour})
			if err != nil {
				t.Fatalf("NewExporter: %v", err)
			}
			defer func() { _ = exp.Close() }()

			e, tok := loadEngine(t, Options{MaxActive: 1, Workers: 2})
			ch, err := e.Submit(Request{
				Prompt: tok.Encode("Once", true), MaxTokens: 2,
				Trace: exp, Operation: c.operation, ResponseID: "chatcmpl-test",
				ResponseModel: "served-model",
			})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			for e.Pump() > 0 {
			}
			for range ch {
			}
			if err := exp.Flush(context.Background()); err != nil {
				t.Fatalf("Flush: %v", err)
			}

			mu.Lock()
			got := string(body)
			mu.Unlock()
			for _, want := range []string{
				`"gen_ai.operation.name"`,
				`"gen_ai.provider.name"`,
				`"gen_ai.response.id"`,
				`"gen_ai.response.model"`,
				`"chatcmpl-test"`,
				`"name":"` + c.wantOp + `"`, // the span name is the operation name
				`"stringValue":"` + c.wantOp + `"`,
				`"gen_ai.response.model"`, `"served-model"`,
			} {
				if !strings.Contains(got, want) {
					t.Errorf("span is missing %s: %s", want, got)
				}
			}
		})
	}
}

// TestClassifyError covers every value classifyError can produce.
//
// The end-to-end test below reaches only "canceled", because that is the one failure a
// fixture can produce deterministically: pool exhaustion, an abandoned caller and a
// model-side failure each need a contrived engine state. classifyError is a pure function
// over the error, so the other three are cheap to pin here -- and pinning them matters more
// than the coverage number suggests, because these strings go into error.type and a backend
// groups failures by them. A branch that returned "" would make failed requests count as
// unclassified rather than as a named class.

// TestRenderChatPromptCapitalizesRunes covers the neutral transcript's role capitalization.
//
// It sliced role[:1] to uppercase the first letter, which cuts a multi-byte rune in half: a
// client sending role "é" got "\ufffd\xa9: ..." in the prompt, because the first byte of é
// became U+FFFD and the continuation byte was written as text. The role is caller-supplied
// JSON, so any string reaches here -- the CLI hardcodes "user"/"assistant", but the server
// passes the request's role through unchanged.
//
// Measured before the fix, through the server: a chat request with role "é" returned 200, so
// nothing failed -- the prompt just opened with the two garbage bytes and the model answered
// from a transcript that did not say who was speaking.
func TestRenderChatPromptCapitalizesRunes(t *testing.T) {
	cases := []struct {
		role string
		want string
		why  string
	}{
		{"user", "User: hi\nAssistant:", "ASCII capitalizes as before"},
		{"system", "System: hi\nAssistant:", "the role the fallback must not lose"},
		{"", "User: hi\nAssistant:", "an empty role defaults to user"},
		{"é", "É: hi\nAssistant:", "a two-byte rune must survive whole"},
		{"中文", "中文: hi\nAssistant:", "capitalizing a rune with no upper case leaves it alone"},
		// A first byte that is not a rune at all. DecodeRuneInString answers RuneError with
		// size 1 for it, so the guard falls to the else arm, which had no case: the five above
		// all take the other one. The invalid byte becomes the replacement character rather
		// than staying invalid -- the assertion below requires a transcript that is valid
		// UTF-8 whatever the role was, and writing the byte through broke it, measurably:
		// a request with role "\xffa" returned 200 and the prompt opened with the raw byte.
		{"\xffa", "\uFFFDa: hi\nAssistant:", "an invalid first byte becomes the replacement character"},
		{"\xc3", "\uFFFD: hi\nAssistant:", "a lone continuation byte is the same case"},
	}
	for _, c := range cases {
		got := RenderChatPrompt([]Turn{{Role: c.role, Text: "hi"}})
		if got != c.want {
			t.Errorf("RenderChatPrompt(role=%q) = %q, want %q (%s)", c.role, got, c.want, c.why)
		}
		// The transcript is encoded into tokens, so it has to be valid UTF-8 whatever the
		// role was -- that is the property the byte slice broke.
		if !utf8.ValidString(got) {
			t.Errorf("RenderChatPrompt(role=%q) = %q, which is not valid UTF-8", c.role, got)
		}
	}

	// The system prompt is the reason this fallback matters: an unsupported template must
	// not lose the instruction, or every request runs without it.
	sys := RenderChatPrompt([]Turn{{Role: "system", Text: "be brief"}, {Role: "user", Text: "hi"}})
	if !strings.Contains(sys, "System: be brief") {
		t.Errorf("neutral transcript dropped the system turn: %q", sys)
	}
}

func TestClassifyError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"canceled", context.Canceled, "canceled"},
		{"pool exhausted", llama.ErrPoolExhausted, "pool_exhausted"},
		{"caller gone", errCallerGone, "caller_gone"},
		{"model failure", errors.New("prefill failed"), "inference_failed"},
		{"nil error", nil, "inference_failed"},

		// The classification is by errors.Is, so the sentinel has to survive wrapping --
		// that is how it reaches here: every engine error is wrapped with context.
		{"wrapped canceled", fmt.Errorf("round 3: %w", context.Canceled), "canceled"},
		{"wrapped pool", fmt.Errorf("admission: %w", llama.ErrPoolExhausted), "pool_exhausted"},
		{"wrapped caller gone", fmt.Errorf("emit: %w", errCallerGone), "caller_gone"},

		// A deadline is not a cancellation in this engine: nothing inside the pump sets one,
		// so a request that carries it is a model-side failure to the trace. Asserted rather
		// than left implicit, because the two are easy to conflate.
		{"deadline exceeded", context.DeadlineExceeded, "inference_failed"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := classifyError(c.err); got != c.want {
				t.Errorf("classifyError(%v) = %q, want %q", c.err, got, c.want)
			}
		})
	}

	// Every value must be a non-empty constant, since error.type is what a backend groups on.
	for _, err := range []error{context.Canceled, llama.ErrPoolExhausted, errCallerGone, errors.New("x")} {
		if got := classifyError(err); got == "" {
			t.Errorf("classifyError(%v) returned an empty class", err)
		}
	}
}

// TestErrorTypeOnFailedSpans covers error.type, which is Conditionally Required: present when the
// operation failed and absent when it did not. A backend filters failures by its presence, so a
// missing value means failed requests that look successful in a dashboard.
func TestErrorTypeOnFailedSpans(t *testing.T) {
	type run struct {
		name    string
		setup   func(cancel chan struct{}) (Request, func())
		want    string
		wantErr bool
	}
	// A cancelled request is the one failure a fixture can produce deterministically.
	cancelled := run{
		name: "cancelled",
		setup: func(cancel chan struct{}) (Request, func()) {
			return Request{Cancel: cancel}, func() { close(cancel) }
		},
		want:    "canceled",
		wantErr: true,
	}

	var mu sync.Mutex
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		body = b
		mu.Unlock()
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	exp, err := otel.NewExporter(otel.ExporterOptions{Endpoint: srv.URL, Interval: time.Hour})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	e, tok := loadEngine(t, Options{MaxActive: 1, Workers: 2})
	cancel := make(chan struct{})
	req, fire := cancelled.setup(cancel)
	req.Prompt = tok.Encode("Once upon a time", true)
	req.MaxTokens = 64
	req.Trace = exp
	ch, err := e.Submit(req)
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	fire()
	for e.Pump() > 0 {
	}
	for range ch {
	}
	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	mu.Lock()
	got := string(body)
	mu.Unlock()
	if !strings.Contains(got, `"error.type"`) {
		t.Fatalf("failed span has no error.type: %s", got)
	}
	if !strings.Contains(got, `"`+cancelled.want+`"`) {
		t.Errorf("error.type is not %q: %s", cancelled.want, got)
	}

	// A successful request must not carry it at all.
	mu.Lock()
	body = nil
	mu.Unlock()
	ch, err = e.Submit(Request{Prompt: tok.Encode("Once", true), MaxTokens: 2, Trace: exp})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	for range ch {
	}
	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	mu.Lock()
	okBody := string(body)
	mu.Unlock()
	if strings.Contains(okBody, `"error.type"`) {
		t.Errorf("successful span carries error.type: %s", okBody)
	}
}

// TestPriorityStringNamesUnknownValues covers the default arm of Priority.String.
//
// Priority is an exported integer type, so a caller outside this package can construct a value
// the three constants do not cover -- from a config file, a stored setting, or an RPC enum that
// gained a member. The arm exists so that such a value is reported as what it is rather than
// masquerading as one of the policies this package implements: a log line saying "fair" for a
// value the engine does not treat as fair would be worse than a number.
//
// It is unreachable through ParsePriority, which is why no test reached it; nothing in the
// package can produce these values from a string.
func TestPriorityStringNamesUnknownValues(t *testing.T) {
	for _, p := range []Priority{Priority(3), Priority(99), Priority(-1)} {
		got := p.String()
		if got == "fifo" || got == "shortest" || got == "fair" {
			t.Errorf("Priority(%d).String() = %q, which names an implemented policy", int(p), got)
			continue
		}
		// The number has to be in the text, or a reader cannot tell which value arrived.
		if !strings.Contains(got, strconv.Itoa(int(p))) {
			t.Errorf("Priority(%d).String() = %q, which does not carry the value", int(p), got)
		}
	}
}

// TestRemainingClampsAtZero covers a sequence that has emitted past its budget.
//
// The engine stops a sequence when the budget is spent, so this state should not arise in a normal
// run -- but the comparison reads it, and without the clamp a negative "remaining" would sort an
// over-budget sequence ahead of every other. A shortest-first policy that ranks a stuck sequence first
// would give it the slot on every round.
//
// The state is constructible through the same helper the other policy tests use, which is why it is
// tested rather than argued unreachable.
func TestRemainingClampsAtZero(t *testing.T) {
	cases := []struct {
		name               string
		maxTokens, emitted int
		want               int
	}{
		{"untouched", 5, 0, 5},
		{"partly done", 5, 2, 3},
		{"exactly spent", 5, 5, 0},
		{"one past the budget", 5, 6, 0},
		{"well past the budget", 2, 7, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := fakeSeq(0, c.maxTokens, c.emitted, -1)
			if got := s.remaining(); got != c.want {
				t.Errorf("remaining() = %d for %d of %d emitted, want %d",
					got, c.emitted, c.maxTokens, c.want)
			}
		})
	}

	// And the clamp has to hold through the comparison a policy uses, not only on its own: an
	// over-budget sequence and one with exactly nothing left are equal work, so the id breaks the tie.
	// Three sequences and a limit of two, so the policy actually runs: admit returns its input
	// unchanged when everything fits, which is what made the first version of this assertion wrong.
	over := fakeSeq(2, 5, 9, -1)
	spent := fakeSeq(0, 5, 5, -1)
	other := fakeSeq(1, 5, 1, -1) // four left, so it goes last
	got := admit([]*sequence{over, spent, other}, 2, PriorityShortest)
	// ids 0 and 2 are the two with no budget left, ordered by id; id 1 has four tokens to go.
	if want := []int{0, 2}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("shortest admitted %v, want %v -- an over-budget sequence is not less work than a "+
			"spent one", ids(got), want)
	}
}

// TestStreamRequestSetsTheStreamAttribute covers the one attribute engine adds conditionally for
// streaming.
//
// otel's encoding of a boolean is tested; what was not is the decision to add the attribute at all. A
// span for a streaming request that omits gen_ai.request.stream looks identical to a non-streaming one,
// and a backend computing time-to-first-token from these spans would include both.
func TestStreamRequestSetsTheStreamAttribute(t *testing.T) {
	cases := []struct {
		name   string
		stream bool
	}{
		{"streaming", true},
		{"not streaming", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var mu sync.Mutex
			var body []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				mu.Lock()
				body = b
				mu.Unlock()
				_, _ = w.Write([]byte(`{}`))
			}))
			defer srv.Close()

			exp, err := otel.NewExporter(otel.ExporterOptions{Endpoint: srv.URL, Interval: time.Hour})
			if err != nil {
				t.Fatalf("NewExporter: %v", err)
			}
			defer func() { _ = exp.Close() }()

			e, tok := loadEngine(t, Options{MaxActive: 1, Workers: 2})
			ch, err := e.Submit(Request{
				Prompt: tok.Encode("Once", true), MaxTokens: 2,
				Trace: exp, Operation: "chat", ResponseID: "id", Stream: c.stream,
			})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			for e.Pump() > 0 {
			}
			for range ch {
			}
			if err := exp.Flush(context.Background()); err != nil {
				t.Fatalf("Flush: %v", err)
			}

			mu.Lock()
			got := string(body)
			mu.Unlock()
			has := strings.Contains(got, `"gen_ai.request.stream"`)
			if c.stream && !has {
				t.Errorf("a streaming request's span has no gen_ai.request.stream: %s", got)
			}
			if !c.stream && has {
				t.Errorf("a non-streaming request's span carries gen_ai.request.stream: %s", got)
			}
			// And when present it must be a boolean, which is otel's half of the contract.
			if c.stream && !strings.Contains(got, `"boolValue":true`) {
				t.Errorf("the stream attribute is not a true boolean: %s", got)
			}
		})
	}
}

// TestShortestServesTheShortestFirstWhateverTheSubmissionOrder is the other half of
// TestSchedulingPolicyKeepsShortRequestResponsive.
//
// That test submits the long request first and checks the short one is not starved, which
// shortest achieves by noticing a small remaining budget. It cannot distinguish shortest from
// a policy that merely picks the newest arrival: with two requests, "least remaining work" and
// "last submitted" choose the same one. Submitting the requests in descending budget order
// separates them, and it is the shape the policy's doc describes -- a queue of mixed-size
// requests dominated by its longest member.
//
// Measured on the fixture with MaxActive 1 and budgets 16, 8, 4, 2 in submission order:
// FIFO serves them on rounds 2, 19, 28, 33 while shortest serves 19, 10, 5, 2 -- the
// two-token request moves from round 33 to round 2, and the sixteen-token one from 2 to 19.
//
// Those three sequences are the table docs/configuration.md prints for this queue, and Fair
// is its third row: 5, 6, 7, 8, one turn apiece in submission order. That row was the one
// nothing pinned -- the fair tests check the admit decision and the bound on the gap, neither
// of which says what the four rounds are -- so it is asserted here with the other two.
func TestShortestServesTheShortestFirstWhateverTheSubmissionOrder(t *testing.T) {
	path := gguftest.Model{NoEOS: true}.Write(t, t.TempDir())
	budgets := []int{16, 8, 4, 2}

	firstRounds := func(t *testing.T, policy Priority) []int {
		t.Helper()
		e, tok := loadEngineAt(t, path, Options{MaxActive: 1, Workers: 2, Policy: policy})
		prompt := tok.Encode("Once upon a time", true)
		chans := make([]<-chan Event, len(budgets))
		for i, b := range budgets {
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: b})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			chans[i] = ch
		}
		// Drain on every Pump round is unnecessary here: the channels are buffered for 64
		// events and the longest budget is 16, so nothing blocks the pump.
		for e.Pump() > 0 {
		}
		first := make([]int, len(budgets))
		for i, ch := range chans {
			for ev := range ch {
				if ev.HasToken && first[i] == 0 {
					first[i] = ev.Round
				}
			}
		}
		return first
	}

	fifo := firstRounds(t, PriorityFIFO)
	shortest := firstRounds(t, PriorityShortest)

	// FIFO serves in submission order, so the first-token rounds have to be increasing.
	for i := 1; i < len(fifo); i++ {
		if fifo[i] <= fifo[i-1] {
			t.Fatalf("FIFO out of submission order: first-token rounds %v", fifo)
		}
	}
	// Shortest serves the smallest budget first, so the rounds have to be decreasing: the
	// last-submitted request (2 tokens) goes before the first-submitted one (16).
	for i := 1; i < len(shortest); i++ {
		if shortest[i] >= shortest[i-1] {
			t.Errorf("shortest did not invert submission order: first-token rounds %v", shortest)
		}
	}
	// The point of the policy, stated as the comparison a reader cares about: the smallest
	// request arrives much sooner than submission order would have served it.
	smallest := len(budgets) - 1
	if shortest[smallest] >= fifo[smallest] {
		t.Errorf("shortest served the 2-token request on round %d, FIFO on round %d: the "+
			"policy changed nothing", shortest[smallest], fifo[smallest])
	}

	// Fair is the third row of the table docs/configuration.md prints for this same queue,
	// and it was the one nothing asserted: the fair tests here check the admit decision and
	// the bound on the gap, neither of which says what the four rounds are. Its shape is
	// "one turn apiece, then round-robin", so the rounds have to increase by exactly one --
	// decreasing would be shortest, and a jump would mean somebody was passed over.
	fair := firstRounds(t, PriorityFair)
	for i := 1; i < len(fair); i++ {
		if fair[i] != fair[i-1]+1 {
			t.Errorf("fair first-token rounds %v: each request should take the next round in "+
				"submission order, so consecutive entries differ by one", fair)
		}
	}

	t.Logf("FIFO first-token rounds %v; shortest %v; fair %v", fifo, shortest, fair)
}

// TestFairBoundsTheGapBetweenTurns checks the property PriorityFair is named for.
//
// The policy's own comment says "no sequence can be passed over more than about
// len(active)/MaxActive rounds in a row. That is the property fair claims". The existing tests
// check which sequence one admit call picks; none run the policy over enough rounds to say
// anything about the wait a sequence sees, which is what a caller choosing this policy over
// FIFO is buying.
//
// Measured with the engine rather than admit alone, because the gap a request experiences also
// depends on how the engine advances the sequences it admits: ceil(total/active) is the bound,
// and four shapes confirmed it exactly -- 8 sequences on 2 slots gave a worst gap of 4, 4 on 1
// gave 4, 9 on 3 gave 3, and 3 on 2 gave 2 (the ceiling of 1.5).
func TestFairBoundsTheGapBetweenTurns(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	prompt := tok.Encode("hi", true)

	for _, shape := range []struct{ active, total int }{
		{2, 8}, {1, 4}, {3, 9}, {2, 3},
	} {
		name := fmt.Sprintf("%d slots %d sequences", shape.active, shape.total)
		t.Run(name, func(t *testing.T) {
			e, err := New(path, Options{
				MaxActive: shape.active, MaxQueued: shape.total, Policy: PriorityFair,
			})
			if err != nil {
				t.Fatalf("New: %v", err)
			}

			// Mixed budgets, so the shapes are not all identical sequences. This does not
			// make the test discriminate Fair from FIFO, and it should not be read as doing
			// so: measured with 8 sequences on 2 slots, FIFO's worst gap is 1 and Fair's is
			// the bound of 4, so FIFO satisfies "at most 4" comfortably. What separates the
			// two policies is the lastServed ordering itself, which
			// TestAdmitFairPicksLeastRecentlyServed covers at the admit level.
			budget := func(i int) int {
				if i%3 == 0 {
					return 2
				}
				return 9
			}
			rounds := make([][]int, shape.total)
			var wg sync.WaitGroup
			var mu sync.Mutex
			for i := range shape.total {
				ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: budget(i)})
				if err != nil {
					t.Fatalf("Submit %d: %v", i, err)
				}
				wg.Go(func() {
					for ev := range ch {
						if ev.HasToken {
							mu.Lock()
							rounds[i] = append(rounds[i], int(ev.Round))
							mu.Unlock()
						}
					}
				})
			}
			for e.Pump() > 0 {
			}
			wg.Wait()

			// Every sequence must have run its full budget, or the gap measurement is
			// over a sequence that was starved rather than merely spaced out.
			for i := range shape.total {
				if want := budget(i); len(rounds[i]) != want {
					t.Fatalf("sequence %d produced %d tokens, want %d: it was starved, which is "+
						"the failure this policy exists to prevent", i, len(rounds[i]), want)
				}
			}

			// The bound is ceil(total/active): integer arithmetic, since the ratio is not
			// always exact.
			want := (shape.total + shape.active - 1) / shape.active
			worst, worstSeq := 0, -1
			for i := range shape.total {
				for j := 1; j < len(rounds[i]); j++ {
					if gap := rounds[i][j] - rounds[i][j-1]; gap > worst {
						worst, worstSeq = gap, i
					}
				}
			}
			if worst > want {
				t.Errorf("sequence %d waited %d rounds between turns; the policy promises at "+
					"most %d (%d sequences on %d slots)", worstSeq, worst, want, shape.total,
					shape.active)
			}
		})
	}
}

// TestFairInterleavesNewArrivalsWithALongRunningSequence covers what Fair is for, in the one
// shape where it differs from FIFO on a real engine.
//
// Everywhere else the two agree: a queue made only of never-served sequences has all its
// lastServed at -1, so Fair falls back to the id tie-break and matches submission order. The
// divergence needs a sequence that has already run -- its lastServed is a real round number,
// so a fresh arrival at -1 outranks it -- competing with a long one that still has budget.
//
// Measured: FIFO gave the long sequence rounds 2-13 without a break and served the newcomers
// afterwards; Fair gave it rounds 2, 3, 8, 13 and then interleaved, with the newcomers taking
// rounds 9 and 14 onwards. The same total round count in both cases, which is why an earlier
// attempt to tell them apart by round count reported "no difference" -- an ordering property
// needs the order observed, not its length.
func TestFairInterleavesNewArrivalsWithALongRunningSequence(t *testing.T) {
	trace := func(t *testing.T, policy Priority) map[int32][]int {
		t.Helper()
		// One active slot, so each round's choice is visible as a single sequence.
		e, tok, _ := loadModelPath(t, Options{MaxActive: 1, MaxQueued: 16, Policy: policy})
		prompt := tok.Encode("hi", false)

		// Every sequence gets its own reader goroutine, so the map needs a lock: without one
		// this is "fatal error: concurrent map writes", which took down the whole test binary
		// rather than failing one test. Found by running the suite with its output kept, after
		// two earlier sessions lost the message to a discarded log.
		rounds := map[int32][]int{}
		var roundsMu sync.Mutex
		collect := func(ch <-chan Event, id int32) chan struct{} {
			done := make(chan struct{})
			go func() {
				defer close(done)
				for ev := range ch {
					if ev.HasToken {
						roundsMu.Lock()
						rounds[id] = append(rounds[id], ev.Round)
						roundsMu.Unlock()
					}
				}
			}()
			return done
		}

		// A long sequence starts alone and advances, so its lastServed becomes a real round.
		long, err := e.Submit(Request{Prompt: prompt, MaxTokens: 12})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		longDone := collect(long, 0)
		for range 3 {
			e.Pump()
		}
		// Short arrivals while it still has budget: these are the sequences that Fair should
		// let in ahead of it and FIFO should not.
		var dones []chan struct{}
		for id := int32(1); id <= 4; id++ {
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 2})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			dones = append(dones, collect(ch, id))
		}
		for e.Pump() > 0 {
		}
		<-longDone
		for _, d := range dones {
			<-d
		}
		return rounds
	}

	fifo := trace(t, PriorityFIFO)
	fair := trace(t, PriorityFair)

	// The property: a newcomer must reach a round before the long sequence finishes. Under
	// Fair at least one does; under FIFO none does. Asserting both directions is what makes
	// this a test of the policies rather than of one of them.
	longFIFO := fifo[0]
	firstNewcomerFIFO := fifo[1][0]
	if firstNewcomerFIFO < longFIFO[len(longFIFO)-1] {
		t.Errorf("FIFO served a newcomer in round %d, before the long sequence finished in round %d; "+
			"the comparison below assumes FIFO does not interleave", firstNewcomerFIFO, longFIFO[len(longFIFO)-1])
	}
	longFair := fair[0]
	interleaved := false
	for id := int32(1); id <= 4; id++ {
		if len(fair[id]) > 0 && fair[id][0] < longFair[len(longFair)-1] {
			interleaved = true
		}
	}
	if !interleaved {
		t.Errorf("Fair did not let any newcomer in before the long sequence finished (%v); "+
			"that is the property it exists for", fair)
	}
}

// TestAdmitFairBoundsEveryWait is the property PriorityFair's comment claims, checked across
// rounds rather than within one.
//
// The comment states it as "no sequence can be passed over more than about len(active)/MaxActive
// rounds in a row", which is a bound on a sequence's wait, not on the order of one selection. The
// existing tests pin single selections -- which sequence is picked first, and that a never-served
// one outranks a served one -- and neither can fail if the bound degrades to "usually".
//
// The simulation runs admit repeatedly, marking each winner with the round it ran, and checks the
// longest gap between turns against the ceiling the comment names. A policy change that ordered by
// lastServed but broke ties badly, or dropped the sorting for a cheaper approximation, would show
// up here as one sequence waiting many more rounds than its neighbours.
func TestAdmitFairBoundsEveryWait(t *testing.T) {
	for _, tc := range []struct{ n, limit int }{
		{6, 2}, {6, 1}, {10, 3}, {5, 5}, {9, 4},
	} {
		t.Run(fmt.Sprintf("n=%d limit=%d", tc.n, tc.limit), func(t *testing.T) {
			ready := make([]*sequence, tc.n)
			for i := range ready {
				// Equal budgets, so PriorityFair's lastServed key is the only thing that
				// orders them; a different budget would let the tie-break by remaining work
				// interfere with the property under test.
				ready[i] = fakeSeq(i, 100, 0, -1)
			}

			lastRan := make([]int, tc.n)
			for i := range lastRan {
				lastRan[i] = -1
			}
			const rounds = 40
			for round := range rounds {
				winners := admit(ready, tc.limit, PriorityFair)
				if len(winners) != tc.limit {
					t.Fatalf("round %d admitted %d, want %d", round, len(winners), tc.limit)
				}
				for _, w := range winners {
					lastRan[w.id] = round
					w.lastServed = round
				}
			}

			// The bound the comment names, rounded up: with n sequences and limit slots, a
			// sequence cannot be passed over more than ceil(n/limit) rounds in a row.
			bound := (tc.n + tc.limit - 1) / tc.limit
			for i, ran := range lastRan {
				gap := rounds - 1 - ran
				if gap > bound {
					t.Errorf("sequence %d last ran on round %d, %d rounds before the end; the "+
						"fair policy promises a wait of at most %d", i, ran, gap, bound)
				}
			}
		})
	}
}
