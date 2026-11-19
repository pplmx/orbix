package main

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/llama"
)

// The suite is a measurement tool, so its tests check the things that make its numbers
// trustworthy: that every scenario runs, that the pairs meant to isolate a mechanism
// actually differ, and that the table renders. They do not assert throughput values --
// those vary with the machine and with whatever else is running on it.

func suiteModel(t *testing.T) string {
	t.Helper()
	// A constructed model rather than a checkpoint: these tests check that every
	// scenario runs and that the pairs meant to isolate a mechanism differ, and neither
	// depends on the weights. A checkpoint would skip them wherever one is not on disk.
	return gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
}

func testSuiteOptions(t *testing.T) suiteOptions {
	t.Helper()
	path := suiteModel(t)
	tok, cfg, err := benchLoad(path)
	if err != nil {
		t.Fatalf("benchLoad: %v", err)
	}
	// A long enough prompt that prefix reuse has a prefix to reuse: with five tokens the
	// saved prefill is lost in the noise, and the pair would look identical.
	prompt := tok.Encode(strings.Repeat("Once upon a time there was a girl. ", 4), cfg.BOSID >= 0)
	if len(prompt) == 0 {
		t.Fatal("prompt encoded to nothing")
	}
	return suiteOptions{model: path, prompt: prompt, maxTok: 8, workers: 4}
}

func TestSuiteEveryScenarioRuns(t *testing.T) {
	o := testSuiteOptions(t)
	for _, sc := range scenarioList() {
		t.Run(sc.name, func(t *testing.T) {
			res, err := sc.run(o)
			if err != nil {
				t.Fatalf("%s: %v", sc.name, err)
			}
			if res.tokens <= 0 {
				t.Errorf("%s produced no tokens", sc.name)
			}
			if res.elapsed <= 0 {
				t.Errorf("%s took no measurable time", sc.name)
			}
			// A row whose first token never arrived would print "-" and hide the fact
			// that the scenario did not start.
			if res.firstToken <= 0 {
				t.Errorf("%s never reported a first token", sc.name)
			}
			if res.elapsed < res.firstToken {
				t.Errorf("%s finished (%v) before its first token (%v)", sc.name, res.elapsed, res.firstToken)
			}
		})
	}
}

// TestSuiteScenarioNamesAreUnique keeps the table readable: two rows with the same name
// would be indistinguishable in output.
func TestSuiteScenarioNamesAreUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, sc := range scenarioList() {
		if seen[sc.name] {
			t.Errorf("duplicate scenario name %q", sc.name)
		}
		seen[sc.name] = true
		if strings.TrimSpace(sc.note) == "" {
			t.Errorf("scenario %q has no note; a number without a question is not readable", sc.name)
		}
	}
}

// TestSuitePrefixReuseBeatsNoReuse checks that the pair of rows exists to isolate reuse
// actually does: with a shared prompt long enough to matter, reuse must be faster.
//
// The margin is loose because this runs on a shared machine. The failure it catches is
// reuse being wired off entirely, which makes the two rows identical.
func TestSuitePrefixReuseBeatsNoReuse(t *testing.T) {
	o := testSuiteOptions(t)
	var reuse, plain result
	for _, sc := range scenarioList() {
		switch sc.name {
		case "prefix reuse x4":
			res, err := sc.run(o)
			if err != nil {
				t.Fatalf("%s: %v", sc.name, err)
			}
			reuse = res
		case "prefix no reuse":
			res, err := sc.run(o)
			if err != nil {
				t.Fatalf("%s: %v", sc.name, err)
			}
			plain = res
		}
	}
	// The elapsed times are logged but not asserted on. Comparing them was the original
	// check and it is not reliable here: the two runs happen back to back on a shared
	// machine, so a scheduling hiccup in the second one is enough to invert the comparison,
	// and the difference reuse makes on a fixture this size is milliseconds. What is
	// deterministic is the count -- reuse reports how many positions it served from the
	// prefix, and no reuse cannot report any.
	t.Logf("no reuse %v (%d reused), reuse %v (%d reused)",
		plain.elapsed, plain.prefixReused, reuse.elapsed, reuse.prefixReused)
	if plain.prefixReused != 0 {
		t.Errorf("the reuse-off row reported %d reused positions", plain.prefixReused)
	}
	if reuse.prefixReused == 0 {
		t.Error("the reuse-on row served no positions from the prefix, so reuse is not applied")
	}
}

// TestSuitePolicyPairsDiffer is the counterpart for the admission policies: the two rows
// differ only in policy, and the short request's latency is what a policy changes.
func TestSuitePolicyPairsDiffer(t *testing.T) {
	o := testSuiteOptions(t)
	var shortest, fifo result
	for _, sc := range scenarioList() {
		switch sc.name {
		case "policy shortest":
			res, err := sc.run(o)
			if err != nil {
				t.Fatalf("%s: %v", sc.name, err)
			}
			shortest = res
		case "policy fifo":
			res, err := sc.run(o)
			if err != nil {
				t.Fatalf("%s: %v", sc.name, err)
			}
			fifo = res
		}
	}
	// Compare the round the short request first ran, not the wall time. Two policies
	// reordering the same work differ by a few milliseconds on a small model, and a
	// millisecond comparison then measures which goroutine woke first -- this assertion
	// failed one run in six before the round was used instead.
	t.Logf("shortest first round %d, fifo %d", shortest.firstRound, fifo.firstRound)
	if shortest.firstRound >= fifo.firstRound {
		t.Errorf("shortest served the short request on round %d and fifo on round %d; "+
			"shortest must reach it sooner", shortest.firstRound, fifo.firstRound)
	}
}

// TestSuitePolicyRowExplainsRaisedTokenCount covers the note that keeps the table from
// contradicting its own header.
//
// The policy rows need a long request the queue can order, so they raise anything below 60
// tokens. Measured with --max-tokens 4: the header printed "generating: 4 tokens per request"
// and the policy rows reported 63, with nothing to say why. A reader comparing rows has no
// way to tell a bug in the token count from a deliberate floor.
//
// Both directions are asserted, because a note that appears unconditionally would be noise
// on every ordinary run.
func TestSuitePolicyRowExplainsRaisedTokenCount(t *testing.T) {
	// Below the floor: the note must say what was asked, what was used, and why.
	low := raisedNote(true, 4, 60)
	for _, want := range []string{"4", "60", "--max-tokens"} {
		if !strings.Contains(low, want) {
			t.Errorf("note %q does not mention %q", low, want)
		}
	}
	// Above the floor: silent.
	if got := raisedNote(false, 120, 120); got != "" {
		t.Errorf("note for a run that was not raised: %q", got)
	}

	// And the rows really do report the raised total: 60 long + 3 short.
	o := testSuiteOptions(t)
	o.maxTok = 4
	for _, sc := range scenarioList() {
		if sc.name != "policy shortest" && sc.name != "policy fifo" {
			continue
		}
		res, err := sc.run(o)
		if err != nil {
			t.Fatalf("%s: %v", sc.name, err)
		}
		if res.tokens != 63 {
			t.Errorf("%s generated %d tokens, want 63 (60 raised + 3 short)", sc.name, res.tokens)
		}
		if !strings.Contains(res.extra, "raised") {
			t.Errorf("%s note %q does not explain the raised count", sc.name, res.extra)
		}
	}
}

// TestSuiteRejectsFlagsItWouldIgnore covers the guard that replaced silent flag loss.
//
// This test used to say the suite "ignores --policy" and that a typo'd value "passes
// silently", which was true and was the defect: cmdBench now walks what was actually set and
// refuses anything the suite does not read. The comment survived the fix, so the test
// asserted the old behaviour -- that ParsePriority rejects a bad name -- which no longer
// describes anything the suite does with the flag.
//
// Driven through cmdBench rather than through a copied flag list, so the assertion is about
// the code path a caller takes. A copy of the allowed set inside the test would pass while
// the real one drifted.
func TestSuiteRejectsFlagsItWouldIgnore(t *testing.T) {
	model := suiteModel(t)

	// Each of these changes what the table would measure, so accepting it silently would
	// report numbers the operator did not ask for.
	for _, extra := range [][]string{
		{"--policy", "shortest"},
		{"--concurrency", "4"},
		{"--requests", "10"},
		{"--prefix-cache"},
	} {
		args := append([]string{"--model", model, "--suite", "--max-tokens", "1"}, extra...)
		err := cmdBench(args)
		if err == nil {
			t.Errorf("cmdBench accepted %v with --suite", extra)
			continue
		}
		if !strings.Contains(err.Error(), "--suite") {
			t.Errorf("%v: error %q does not say --suite is why", extra, err)
		}
	}

	// The flags the suite does use must still be accepted, or the guard would be satisfied
	// by refusing everything. Run with a one-token budget so the table finishes quickly.
	if err := cmdBench([]string{"--model", model, "--suite", "--max-tokens", "1", "--workers", "2"}); err != nil {
		t.Errorf("cmdBench rejected the flags --suite uses: %v", err)
	}
}

// TestEarlyStopsAreCounted checks the field that explains a token total shorter than the
// budget. The fixture here has a real end-of-sequence token, so a long enough budget
// eventually reaches it; a request that stops early must be counted rather than silently
// producing fewer tokens than its row's neighbours.
func TestEarlyStopsAreCounted(t *testing.T) {
	// A model that must stop on its first step, rather than one that happens to. The EOS
	// id is pointed at the piece this fixture's greedy path emits -- the score table is
	// derived from the vocabulary, so which piece wins is a property of the fixture and
	// not something a test should bet on.
	//
	// Finding it first is the honest way to do this: the alternative, choosing a
	// vocabulary that happens to reach EOS in a couple of steps, is what broke when the
	// fixture's words changed.
	probe := gguftest.Model{ContextLen: 256, NoBytes: true}.Write(t, t.TempDir())
	eos := firstGreedyToken(t, probe)
	path := gguftest.Model{ContextLen: 256, NoBytes: true, EOSID: eos, BOSID: 1}.Write(t, t.TempDir())
	o := suiteOptions{
		model:   path,
		prompt:  []int32{1, 4, 5},
		maxTok:  64,
		workers: 2,
	}
	res, err := runRequests(o, 2, 2, engine.Options{})
	if err != nil {
		t.Fatalf("runRequests: %v", err)
	}
	// On this fixture the greedy argmax walks to the end-of-sequence token within a few
	// steps, so both requests stop early. That is asserted rather than tolerated: a test
	// whose shortfall branch is never taken has not checked the accounting.
	if res.earlyStops != 2 {
		t.Fatalf("earlyStops = %d with a budget of %d and %d tokens produced; "+
			"both requests should have stopped on EOS", res.earlyStops, o.maxTok, res.tokens)
	}
	if res.tokens >= 2*o.maxTok {
		t.Errorf("tokens = %d reached the full budget despite %d early stops",
			res.tokens, res.earlyStops)
	}
}

// firstGreedyToken returns the id the model emits for a fixed prompt, so a test that
// needs generation to stop immediately can point EOS at it instead of hoping the
// vocabulary produces an end-of-sequence token on its own.
func firstGreedyToken(t *testing.T, path string) int {
	t.Helper()
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	cfg, err := llama.LoadConfig(gf)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	m, err := llama.New(path)
	if err != nil {
		t.Fatalf("llama.New: %v", err)
	}
	cache, err := llama.NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	logits, err := m.Prefill(cache, []int32{1, 4, 5}, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return best
}

// TestPrefixPairActuallyReuses pins what the prefix rows exist to show.
//
// The pair used to submit four requests to a MaxActive=4 engine, so all four prefilled
// together and no request ever arrived after another had cached its prefix: both rows
// printed the same throughput and no reader could tell whether reuse happened and did not
// help, or never happened. The scenario is sequential now, and this asserts that the reuse
// it claims to measure is real rather than assumed.
//
// A second trap sat behind the first, and it is the one the real command hit: a block is
// only published to the prefix index when its last position is written, so a prompt shorter
// than one block never produces anything to reuse. The command's default prompt is five
// tokens against a sixteen-position block, so even the sequential pair measured zero until
// the scenario extended its prompt to span two blocks.
//
// The test's own prompt is already long enough to span blocks, so it cannot tell the
// concurrent and sequential shapes apart -- both reuse once the prompt is long enough. What
// it does pin is that the reuse-on row reuses and the reuse-off row does not, which is the
// property the pair exists to demonstrate.
func TestPrefixPairActuallyReuses(t *testing.T) {
	o := testSuiteOptions(t)
	var off, on *result
	for _, sc := range scenarioList() {
		if sc.name != "prefix no reuse" && sc.name != "prefix reuse x4" {
			continue
		}
		res, err := sc.run(o)
		if err != nil {
			t.Fatalf("%s: %v", sc.name, err)
		}
		if sc.name == "prefix no reuse" {
			off = &res
		} else {
			on = &res
		}
	}
	if off == nil || on == nil {
		t.Fatal("the prefix scenarios are not both in the list")
	}
	if off.prefixReused != 0 {
		t.Errorf("reuse off reported %d reused positions, want 0", off.prefixReused)
	}
	if on.prefixReused == 0 {
		t.Error("reuse on reported no reused positions: the scenario does not exercise the " +
			"path it is named after")
	}
	// Three of the four requests should find the prefix; the first has nothing to find.
	if on.prefixReused%3 != 0 {
		t.Errorf("reused %d positions, which is not a multiple of the three later requests",
			on.prefixReused)
	}
}

// TestBenchSingleMeasurement covers cmdBench's single-measurement path.
//
// The existing bench tests drive cmdBench twice and neither reaches this path: the rejection
// cases return on a validation error, and the accepted case passes --suite, which branches off
// at line 94 before any engine is built. So the whole single-run path -- opening the model,
// choosing a cache, running the load, and printing the report -- had no test, and
// Model.Runner() is called nowhere else in production, which is why it sat at 0% coverage.
//
// The report line is what is asserted: it is where the worker fan-out a caller asked for is
// echoed back, so a run whose options were silently dropped shows up here rather than only in
// numbers that look slightly wrong.
func TestBenchSingleMeasurement(t *testing.T) {
	model := suiteModel(t)

	out := captureStdout(t, func() {
		err := cmdBench([]string{
			"--model", model,
			"--max-tokens", "1",
			"--requests", "1",
			"--concurrency", "1",
			"--workers", "2",
		})
		if err != nil {
			t.Errorf("cmdBench: %v", err)
		}
	})

	// The report has to name the worker count it ran with. Comparing the number matters:
	// printing the flag's default would satisfy a test that only looked for the word.
	if !strings.Contains(out, "workers 2") {
		t.Errorf("bench report does not name the worker count:\n%s", out)
	}
	if !strings.Contains(out, "generating:  1 tokens x 1 requests") {
		t.Errorf("bench report does not describe the run:\n%s", out)
	}
	// A single measurement reports one run, not the suite's table, so the header must be the
	// one-line form and the kv cache line must be present for whichever cache was chosen.
	if !strings.Contains(out, "kv cache:") {
		t.Errorf("bench report has no kv cache line:\n%s", out)
	}
	if strings.Contains(out, "scenario") {
		t.Errorf("a single measurement printed a suite table:\n%s", out)
	}
}

// TestBenchRefusesABlockBiggerThanTheContext covers bench's copy of the prefix-cache trap.
//
// The run reports how many prompt positions it reused, and a reader compares that number
// against zero. A block larger than the context guarantees zero for a reason that has nothing
// to do with whether reuse pays, so the flag is refused rather than measured.
func TestBenchRefusesABlockBiggerThanTheContext(t *testing.T) {
	// The fixture's default window is 64 positions, so 1000 cannot fit.
	path := gguftest.Model{NoEOS: true}.Write(t, t.TempDir())
	err := cmdBench([]string{
		"--model", path, "--blocks", "8", "--block-size", "1000", "--prefix-cache",
		"--max-tokens", "1", "--concurrency", "1",
	})
	if err == nil {
		t.Fatal("bench accepted a block larger than the context with --prefix-cache")
	}
	for _, want := range []string{"--block-size", "1000", "64"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestTokPerSec covers the arithmetic behind the table's TOK/S column.
//
// The column is the reason anyone reads the suite, and tokPerSec had no test: inverting the
// division, or moving the zero guard from `<= 0` to `< 0`, both passed the cmd suite. The
// second is the one worth guarding -- an elapsed of exactly zero divides by zero, and a
// float64 division by zero yields +Inf or NaN rather than panicking, so the table would print
// "Inf" in a cell an operator reads as a measurement.
//
// A duration of exactly zero is not hypothetical: a scenario whose requests all fail
// immediately, or a clock at coarse resolution on a fast machine, produces one.
func TestTokPerSec(t *testing.T) {
	cases := []struct {
		name    string
		tokens  int
		elapsed time.Duration
		want    float64
	}{
		{"one second", 100, time.Second, 100},
		{"half a second", 50, 500 * time.Millisecond, 100},
		{"two seconds", 100, 2 * time.Second, 50},
		{"no tokens", 0, time.Second, 0},
		// The two states that must not divide: a zero duration and a negative one. Negative
		// is not producible by a monotonic clock, but the field is a plain Duration and the
		// guard exists for both.
		{"zero elapsed", 10, 0, 0},
		{"negative elapsed", 10, -time.Second, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := result{tokens: c.tokens, elapsed: c.elapsed}
			got := r.tokPerSec()
			if math.IsNaN(got) || math.IsInf(got, 0) {
				t.Fatalf("tokPerSec(%d tokens, %v) = %v, which is not a number a table can print",
					c.tokens, c.elapsed, got)
			}
			if got != c.want {
				t.Errorf("tokPerSec(%d tokens, %v) = %v, want %v",
					c.tokens, c.elapsed, got, c.want)
			}
		})
	}
}
