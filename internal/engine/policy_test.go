package engine

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
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
	if _, err := ParsePriority("random"); err == nil {
		t.Error("ParsePriority accepted an unknown policy")
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
	run := func(policy Priority) [][]int32 {
		e, tok := loadEngine(t, Options{MaxActive: 2, Workers: 2, Policy: policy})
		prompt := tok.Encode("Once upon a time", true)
		const n = 4
		chans := make([]<-chan Event, n)
		for i := range chans {
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 6})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			chans[i] = ch
		}
		for e.Pump() > 0 {
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
				t.Fatalf("%v: request %d produced %d tokens, fifo produced %d", policy, i, len(got[i]), len(fifo[i]))
			}
			for j := range fifo[i] {
				if got[i][j] != fifo[i][j] {
					t.Errorf("%v: request %d token %d = %d, fifo = %d", policy, i, j, got[i][j], fifo[i][j])
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
func TestTracingRecordsRequestSpans(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}

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
	case <-time.After(5 * time.Second):
		t.Fatal("the collector received no trace")
	}
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
