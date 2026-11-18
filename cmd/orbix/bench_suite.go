package main

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// suiteOptions is what every scenario in the suite needs to build a model.
//
// Each scenario constructs its own engine: the mechanisms being measured are configured
// per engine (cache shape, admission policy, worker count), so sharing one would mean
// measuring the previous scenario's leftovers.
type suiteOptions struct {
	model   string
	prompt  []int32
	maxTok  int
	workers int
}

// scenario is one measured configuration.
type scenario struct {
	name string
	// note explains what the row is for, since a table of numbers without a question
	// attached does not tell a reader whether a difference is expected or a regression.
	note string
	run  func(o suiteOptions) (result, error)
}

// result is one row's measurements. Fields that do not apply stay zero and are left
// blank in the table rather than printed as 0, which would read as "measured as zero".
type result struct {
	tokens int
	// elapsed is wall time to finish every request in the scenario.
	elapsed time.Duration
	// firstToken is the FIRST column: a latency, but of a different request depending on
	// the row. For a throughput row it is how long until *any* request produced a token,
	// which is essentially the prefill time. For the policy rows it is the short
	// request's own first token -- not "any", because the long request that runs first
	// would supply that number under every policy and the column would say nothing. The
	// policies reorder a fixed amount of work, so where the short request lands is the
	// only thing they change.
	firstToken time.Duration
	// prefixReused is how many prompt positions came from a shared prefix rather than a
	// forward pass, so a row can say whether its cache setting had anything to reuse
	// instead of leaving the reader to infer it from throughput.
	prefixReused uint64
	// firstRound is the engine round in which the measured request first produced a
	// token, for the scenarios whose property is ordering rather than speed.
	//
	// The wall time is printed too, but it is the wrong thing to assert on: on a fast
	// model two policies differ by a few milliseconds and the comparison comes down to
	// which goroutine woke first. A round is a property of the schedule.
	firstRound int
	// earlyStops counts requests that ended before their token budget because the model
	// emitted an end-of-sequence token. Without it a row can print fewer tokens than its
	// neighbours for a reason the table does not explain, which reads like a bug in the
	// measurement rather than like the model having nothing more to say.
	earlyStops int
	extra      string
}

// tokPerSec is the throughput this row demonstrated.
func (r result) tokPerSec() float64 {
	if r.elapsed <= 0 {
		return 0
	}
	return float64(r.tokens) / r.elapsed.Seconds()
}

// scenarioList is the suite, in the order it runs.
//
// The order matters only for reading: each scenario is independent, and the sequence is
// arranged so the table answers its questions top to bottom -- how fast is one stream,
// what does batching buy, what does each cache shape cost, and does reuse help.
func scenarioList() []scenario {
	return []scenario{
		{
			name: "single stream",
			note: "baseline: one request, contiguous cache",
			run: func(o suiteOptions) (result, error) {
				return runRequests(o, 1, 1, engine.Options{MaxActive: 1, Workers: o.workers})
			},
		},
		{
			name: "batch x4",
			note: "batched decode, contiguous caches",
			run: func(o suiteOptions) (result, error) {
				return runRequests(o, 4, 4, engine.Options{MaxActive: 4, Workers: o.workers})
			},
		},
		{
			name: "batch x16",
			note: "where batched decode stops scaling",
			run: func(o suiteOptions) (result, error) {
				return runRequests(o, 16, 16, engine.Options{MaxActive: 16, Workers: o.workers})
			},
		},
		{
			name: "paged x16",
			note: "same batch, block cache instead of whole-context reservations",
			run: func(o suiteOptions) (result, error) {
				return runRequests(o, 16, 16, engine.Options{
					MaxActive: 16, Workers: o.workers,
					BlockSize: 16, Blocks: 4096,
				})
			},
		},
		{
			name: "prefix no reuse",
			note: "four requests sharing one prompt, one after another, reuse off; the prompt is repeated to span two blocks, since a shorter one never fills a block to reuse",
			run: func(o suiteOptions) (result, error) {
				return runSequential(o, 4, engine.Options{
					Workers: o.workers, BlockSize: 16, Blocks: 4096,
				})
			},
		},
		{
			name: "prefix reuse x4",
			note: "the same four one after another, reuse on -- the pair is the comparison",
			run: func(o suiteOptions) (result, error) {
				return runSequential(o, 4, engine.Options{
					Workers: o.workers, BlockSize: 16, Blocks: 4096, PrefixCache: true,
				})
			},
		},
		{
			name: "policy shortest",
			note: "MaxActive 1, a short request queued behind a long one",
			run: func(o suiteOptions) (result, error) {
				return runPolicy(o, engine.PriorityShortest)
			},
		},
		{
			name: "policy fifo",
			note: "the same queue without reordering, for comparison",
			run: func(o suiteOptions) (result, error) {
				return runPolicy(o, engine.PriorityFIFO)
			},
		},
	}
}

// runSuite runs every scenario and prints one table.
func runSuite(o suiteOptions) error {
	fmt.Printf("model:       %s\n", o.model)
	fmt.Printf("prompt:      %d tokens\n", len(o.prompt))
	fmt.Printf("generating:  %d tokens per request\n", o.maxTok)
	// The machine and the worker count belong with the numbers, not left to whoever
	// copies the table into a document: a throughput figure without them cannot be
	// compared against anything, and a reader has no way to tell whether two runs
	// differed because of the code or because of the hardware.
	workers := o.workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	fmt.Printf("machine:     %s/%s, %d CPUs, %d kernel workers\n",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), workers)
	fmt.Printf("go:          %s\n\n", runtime.Version())

	type row struct {
		scenario scenario
		res      result
		err      error
	}
	rows := make([]row, 0, len(scenarioList()))
	for _, sc := range scenarioList() {
		res, err := sc.run(o)
		rows = append(rows, row{scenario: sc, res: res, err: err})
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	var werr error
	writef := func(format string, args ...any) {
		if werr != nil {
			return
		}
		if _, err := fmt.Fprintf(tw, format, args...); err != nil {
			werr = err
		}
	}
	writef("SCENARIO\tTOKENS\tELAPSED\tTOK/S\tFIRST\tNOTE\n")
	for _, r := range rows {
		if r.err != nil {
			writef("%s\t-\t-\t-\t-\tfailed: %v\n", r.scenario.name, r.err)
			continue
		}
		note := r.scenario.note
		if r.res.extra != "" {
			note += "; " + r.res.extra
		}
		// Explain a token total that differs from the row above it. The number is real --
		// a request emitted its end-of-sequence token -- but without this a reader sees
		// two rows that should match and starts looking for a measurement bug.
		if r.res.earlyStops > 0 {
			note += fmt.Sprintf("; %d request(s) stopped early on EOS", r.res.earlyStops)
		}
		// Say what the cache setting actually bought. Without this the two prefix rows
		// print nearly the same throughput and the reader has no way to tell whether
		// reuse happened and did not help, or never happened at all.
		if r.res.prefixReused > 0 {
			note += fmt.Sprintf("; %d prompt positions reused", r.res.prefixReused)
		}
		first := "-"
		if r.res.firstToken > 0 {
			first = r.res.firstToken.Round(time.Millisecond).String()
		}
		writef("%s\t%d\t%s\t%.1f\t%s\t%s\n",
			r.scenario.name, r.res.tokens, r.res.elapsed.Round(time.Millisecond),
			r.res.tokPerSec(), first, note)
	}
	if werr == nil {
		werr = tw.Flush()
	}
	if werr != nil {
		return fmt.Errorf("bench: write table: %w", werr)
	}

	// A scenario failing should not hide the ones that ran, but it must not read as
	// success either.
	var failed []string
	for _, r := range rows {
		if r.err != nil {
			failed = append(failed, r.scenario.name)
		}
	}
	if len(failed) > 0 {
		return fmt.Errorf("bench: %d scenario(s) failed: %v", len(failed), failed)
	}
	return nil
}

// newSuiteEngine builds an engine with the suite's shared settings plus opts.
func newSuiteEngine(o suiteOptions, opts engine.Options) (*engine.Engine, error) {
	opts.Workers = o.workers
	return engine.New(o.model, opts)
}

// runRequests submits n requests at once and measures the wall time to finish them all.
//
// Requests are submitted together and pumped together, so the elapsed time reflects
// whatever batching the engine actually achieves rather than the caller's pacing.
func runRequests(o suiteOptions, n, concurrency int, opts engine.Options) (result, error) {
	opts.MaxActive = min(opts.MaxActive, concurrency)
	eng, err := newSuiteEngine(o, opts)
	if err != nil {
		return result{}, err
	}
	prompt := o.prompt

	chans := make([]<-chan engine.Event, n)
	start := time.Now()
	for i := range chans {
		ch, err := eng.Submit(engine.Request{
			Prompt:    prompt,
			MaxTokens: o.maxTok,
			Sampler:   llama.Greedy{},
		})
		if err != nil {
			return result{}, fmt.Errorf("submit: %w", err)
		}
		chans[i] = ch
	}
	// Each drain runs on its own goroutine, one per request, so the counters it updates
	// are guarded: two goroutines writing tokens++ without a lock is a data race that
	// usually produces the right total and occasionally does not.
	var mu sync.Mutex
	var tokens int
	var firstToken time.Duration
	var earlyStops int
	drain := func(ch <-chan engine.Event) error {
		var mine int
		for ev := range ch {
			if ev.Err != nil {
				// The lock is not held here, so take it only around the state updates.
				return ev.Err
			}
			if ev.HasToken {
				mu.Lock()
				if firstToken == 0 {
					firstToken = time.Since(start)
				}
				mu.Unlock()
				mine++
			}
		}
		mu.Lock()
		tokens += mine
		// A request that stopped before its budget did so on an end-of-sequence token.
		// Counting those makes "this row has fewer tokens than its neighbour" an
		// explained number rather than an apparent measurement bug.
		if mine < o.maxTok {
			earlyStops++
		}
		mu.Unlock()
		return nil
	}
	// Drain concurrently with pumping: the buffers are bounded, so a scenario with a
	// large token budget would block Pump waiting for a reader.
	errs := make(chan error, len(chans))
	for _, ch := range chans {
		go func(ch <-chan engine.Event) { errs <- drain(ch) }(ch)
	}
	for eng.Pump() > 0 {
	}
	for range chans {
		if err := <-errs; err != nil {
			return result{}, err
		}
	}
	r := result{tokens: tokens, elapsed: time.Since(start), firstToken: firstToken, earlyStops: earlyStops}
	r.prefixReused = eng.Stats().PrefixReused
	// Report what the cache actually was, since that is the variable the row exists to
	// isolate and a misconfigured flag would otherwise look like a throughput change.
	if pool := eng.Pool(); pool != nil {
		r.extra = fmt.Sprintf("paged %dx%d", pool.Blocks(), pool.BlockSize())
	}
	return r, nil
}

// runPolicy measures how soon a short request is served behind a long one, which is the
// property the admission policies exist to change.
func runPolicy(o suiteOptions, policy engine.Priority) (result, error) {
	eng, err := newSuiteEngine(o, engine.Options{MaxActive: 1, Policy: policy})
	if err != nil {
		return result{}, err
	}
	// The long request needs enough tokens for the policy to have something to reorder:
	// a queue of two requests that both finish in three tokens measures the scheduler's
	// startup, not its ordering. The floor is therefore raised rather than the row being
	// skipped, and the note says so -- without that the table contradicts its own header,
	// which prints the caller's --max-tokens (measured: "generating: 4 tokens per request"
	// above a policy row reporting 63).
	long := o.maxTok
	raised := false
	if long < 60 {
		long = 60
		raised = true
	}
	start := time.Now()
	longCh, err := eng.Submit(engine.Request{Prompt: o.prompt, MaxTokens: long, Sampler: llama.Greedy{}})
	if err != nil {
		return result{}, err
	}
	shortCh, err := eng.Submit(engine.Request{Prompt: o.prompt, MaxTokens: 3, Sampler: llama.Greedy{}})
	if err != nil {
		return result{}, err
	}

	// Pump until both requests finish, draining their channels between rounds. Draining
	// is not optional: the event buffers are bounded, and a 60-token sequence fills its
	// buffer long before it is done, which would block Pump forever.
	//
	// Each drain runs on its own goroutine and reports the count it saw rather than
	// incrementing a shared total. Two goroutines doing tokens++ without a lock is a data
	// race -- the same defect runRequests had, fixed there by a mutex and left here -- and
	// the race detector reports it as soon as the suite test drives this scenario.
	type shortStats struct {
		tokens   int
		round    int
		latency  time.Duration
		firstTok bool
	}
	shortFirst := make(chan shortStats, 1)
	go func() {
		var st shortStats
		for ev := range shortCh {
			if ev.HasToken {
				if !st.firstTok {
					st.firstTok = true
					st.round = eng.Round()
					st.latency = time.Since(start)
				}
				st.tokens++
			}
		}
		shortFirst <- st
	}()
	longTokens := make(chan int, 1)
	go func() {
		var n int
		for ev := range longCh {
			if ev.HasToken {
				n++
			}
		}
		longTokens <- n
	}()

	for eng.Pump() > 0 {
	}
	st := <-shortFirst
	tokens := st.tokens + <-longTokens
	return result{
		tokens:  tokens,
		elapsed: time.Since(start),
		// The short request's own latency is the property under test. Total elapsed
		// time is dominated by the long request, which every policy must also serve, so
		// it says nothing about the policy.
		firstToken: st.latency,
		firstRound: st.round,
		extra: fmt.Sprintf("short request's first token on round %d%s",
			st.round, raisedNote(raised, o.maxTok, long)),
	}, nil
}

// raisedNote explains a policy row that generated more tokens than the caller asked for.
//
// It is empty when the floor did not apply, so the ordinary note is unchanged.
func raisedNote(raised bool, asked, used int) string {
	if !raised {
		return ""
	}
	return fmt.Sprintf("; the long request used %d tokens, raised from --max-tokens %d so the "+
		"queue has something to order", used, asked)
}

// benchLoad opens a model just far enough to read its tokenizer and config, which is
// what the suite needs before it can encode a prompt or size a cache.
func benchLoad(path string) (*tokenize.Tokenizer, llama.Config, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, llama.Config{}, err
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		return nil, llama.Config{}, err
	}
	cfg, err := llama.LoadConfig(gf)
	if err != nil {
		return nil, llama.Config{}, err
	}
	return tok, cfg, nil
}

// runSequential submits its requests one at a time, waiting for each to finish before the
// next starts.
//
// This is what makes prefix caching measurable. runRequests submits everything at once, so
// with MaxActive as large as the request count they all prefill together and no request ever
// arrives after another has cached its prefix: the reuse path is never taken, and the two
// rows print the same throughput with no way to tell why.
//
// A real deployment looks more like this than like the simultaneous case -- requests arrive
// over time, and the second one with a popular system prompt finds it already cached.
func runSequential(o suiteOptions, n int, opts engine.Options) (result, error) {
	opts.MaxActive = 1
	eng, err := newSuiteEngine(o, opts)
	if err != nil {
		return result{}, err
	}
	// The prompt has to span at least one whole block for anything to be published to the
	// prefix index: a block is remembered only when its last position is written, so a
	// prompt shorter than the block never produces a reusable prefix and reuse measures as
	// zero however the cache is configured. Repeating the caller's prompt gets there without
	// inventing text, and keeps the same prefix across the requests in the row.
	block := opts.BlockSize
	if block <= 0 {
		block = 16
	}
	prompt := o.prompt
	for len(prompt) < 2*block {
		prompt = append(prompt[:len(prompt):len(prompt)], o.prompt...)
	}
	var tokens int
	var firstToken time.Duration
	start := time.Now()
	for range n {
		ch, err := eng.Submit(engine.Request{
			Prompt:    prompt,
			MaxTokens: o.maxTok,
			Sampler:   llama.Greedy{},
		})
		if err != nil {
			return result{}, fmt.Errorf("submit: %w", err)
		}
		// Drain and pump concurrently: the pump is what produces the events, so reading
		// first would wait for a token that cannot arrive.
		done := make(chan struct{})
		go func(ch <-chan engine.Event) {
			defer close(done)
			for ev := range ch {
				if !ev.HasToken {
					continue
				}
				tokens++
				if firstToken == 0 {
					firstToken = time.Since(start)
				}
			}
		}(ch)
		for eng.Pump() > 0 {
		}
		<-done
	}
	r := result{
		tokens:       tokens,
		elapsed:      time.Since(start),
		firstToken:   firstToken,
		prefixReused: eng.Stats().PrefixReused,
	}
	if pool := eng.Pool(); pool != nil {
		r.extra = fmt.Sprintf("paged %dx%d", pool.Blocks(), pool.BlockSize())
	}
	return r, nil
}
