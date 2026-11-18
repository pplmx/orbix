package engine

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
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
			`"chat.completion"`,
			`"gen_ai.request.model"`,
			`"gen_ai.usage.input_tokens"`,
			`"gen_ai.usage.output_tokens"`,
			`"gen_ai.response.finish_reason"`,
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

	if !strings.Contains(body, `gen_ai.response.finish_reason`) || !strings.Contains(body, "cancel") {
		t.Fatalf("span does not record the cancel: %s", body)
	}
	// The status code is what a trace query filters on. 1 is ok, 2 is error.
	if !strings.Contains(body, `"status":{"code":2`) {
		t.Errorf("cancelled span is not marked error: %s", body)
	}
}
