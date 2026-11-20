package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
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

// TestRowNoteExplainsTheRow covers the NOTE cell, which is where a surprising number is
// explained rather than left to look like a measurement bug.
//
// runSuite writes to os.Stdout, so none of this was reachable from a test before rowNote was
// extracted: raising the early-stop threshold, the reuse threshold or the failure threshold
// by one each passed the cmd suite. The clauses matter more than they look -- a row that
// generated fewer tokens than its neighbours, or two prefix rows that print the same
// throughput, are exactly the shapes a reader interprets as the tool being wrong.
func TestRowNoteExplainsTheRow(t *testing.T) {
	sc := scenario{name: "row", note: "baseline"}
	cases := []struct {
		name string
		res  result
		want string
	}{
		{"plain", result{}, "baseline"},
		{"extra", result{extra: "paged 8x16"}, "baseline; paged 8x16"},
		{"one early stop", result{earlyStops: 1},
			"baseline; 1 request(s) stopped early on EOS"},
		{"two early stops", result{earlyStops: 2},
			"baseline; 2 request(s) stopped early on EOS"},
		{"one reused position", result{prefixReused: 1},
			"baseline; 1 prompt positions reused"},
		{"many reused positions", result{prefixReused: 96},
			"baseline; 96 prompt positions reused"},
		{"all three", result{extra: "paged 8x16", earlyStops: 1, prefixReused: 32},
			"baseline; paged 8x16; 1 request(s) stopped early on EOS; 32 prompt positions reused"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := rowNote(sc, c.res); got != c.want {
				t.Errorf("rowNote = %q, want %q", got, c.want)
			}
		})
	}

	// A scenario's own note is not optional: every entry in the list has one, and the cell
	// opening with the description is what makes the table readable without the code.
	for _, s := range scenarioList() {
		if s.note == "" {
			t.Errorf("scenario %q has no note", s.name)
		}
	}
}

// TestRowLine covers the two shapes a row can take that are not a plain measurement.
//
// runSuite writes to os.Stdout, so the rendering inside it was unobservable: printing a failed
// scenario as a row of dashes, or a missing first-token time as "0s", both passed the cmd suite.
// Either would be read as a result rather than as an absence.
func TestRowLine(t *testing.T) {
	sc := scenario{name: "single", note: "baseline"}

	t.Run("a measurement", func(t *testing.T) {
		got := rowLine(sc, result{
			tokens:     64,
			elapsed:    500 * time.Millisecond,
			firstToken: 41 * time.Millisecond,
		}, nil)
		want := "single\t64\t500ms\t128.0\t41ms\tbaseline"
		if got != want {
			t.Errorf("rowLine = %q, want %q", got, want)
		}
		// Six cells, so a row cannot silently lose a column.
		if n := len(strings.Split(got, "\t")); n != 6 {
			t.Errorf("rowLine produced %d cells, want 6: %q", n, got)
		}
	})

	t.Run("a scenario that did not run", func(t *testing.T) {
		got := rowLine(sc, result{}, errors.New("model would not load"))
		if !strings.Contains(got, "failed:") {
			t.Errorf("rowLine = %q, want it to say the scenario failed", got)
		}
		if !strings.Contains(got, "model would not load") {
			t.Errorf("rowLine = %q, want it to carry the reason", got)
		}
		// Every measurement cell is a dash: a zero would read as a measurement of nothing.
		cells := strings.Split(got, "\t")
		for i := 1; i <= 4; i++ {
			if cells[i] != "-" {
				t.Errorf("cell %d = %q, want a dash for a scenario that did not run", i, cells[i])
			}
		}
	})

	t.Run("no first-token time recorded", func(t *testing.T) {
		got := rowLine(sc, result{tokens: 8, elapsed: time.Second}, nil)
		cells := strings.Split(got, "\t")
		if cells[4] != "-" {
			t.Errorf("first-token cell = %q, want a dash when none was recorded", cells[4])
		}
		// The rest of the row still has its numbers, or the dash would hide a real row.
		if cells[1] != "8" {
			t.Errorf("token cell = %q, want 8", cells[1])
		}
	})
}

// TestPrefixRowsActuallyDiffer covers the one comparison the suite exists to make.
//
// "prefix no reuse" and "prefix reuse x4" are printed as a pair, and the note says so: the
// reader is meant to compare them. Making them measure the same thing would leave the table
// looking correct while saying nothing -- the reuse row would report the same throughput for
// the same reason, and nothing distinguishes that from "reuse is having no effect".
//
// Every setting runSequential needs is checked here through the engine's own counter rather
// than through timing, so the assertion does not depend on how fast the machine is. Changing
// MaxActive from 1, shrinking the prompt-repetition target to one block, or dropping
// PrefixCache from the second row each made the two rows identical.
func TestPrefixRowsActuallyDiffer(t *testing.T) {
	o := testSuiteOptions(t)

	byName := map[string]scenario{}
	for _, sc := range scenarioList() {
		byName[sc.name] = sc
	}
	off, ok := byName["prefix no reuse"]
	if !ok {
		t.Fatal("the suite has no \"prefix no reuse\" row")
	}
	on, ok := byName["prefix reuse x4"]
	if !ok {
		t.Fatal("the suite has no \"prefix reuse x4\" row")
	}

	// Sequential order, so the two runs do not interfere through shared timing.
	offRes, err := off.run(o)
	if err != nil {
		t.Fatalf("%s: %v", off.name, err)
	}
	onRes, err := on.run(o)
	if err != nil {
		t.Fatalf("%s: %v", on.name, err)
	}

	// Reuse off must reuse nothing -- that is what makes the row a baseline rather than a
	// second measurement of the same thing.
	if offRes.prefixReused != 0 {
		t.Errorf("the reuse-off row reported %d reused positions, want 0", offRes.prefixReused)
	}
	// Reuse on must reuse something. A zero here means either the cache was never enabled for
	// this row, or the requests arrived together and had nothing to reuse from.
	if onRes.prefixReused == 0 {
		t.Errorf("the reuse-on row reported 0 reused positions; the pair cannot be compared")
	}
	// And both rows still did the same work, so the difference is the cache and not the load.
	if onRes.tokens != offRes.tokens {
		t.Errorf("the rows generated %d and %d tokens; they must do the same work",
			onRes.tokens, offRes.tokens)
	}
}

// TestRunSequentialRepeatsAShortPrompt covers the loop that makes a short prompt reusable.
//
// A block is published only when its last position is written, so a prompt shorter than one
// block never produces a reusable prefix and the reuse row measures zero however the cache is
// configured. The suite's own prompt is 90 tokens against a 16-position block, so that loop
// never runs in the table -- the protection is for a caller who passes something short, which is
// exactly the case a reader would not think to try.
//
// The assertion is on the engine's reuse counter rather than on the repeated text, because the
// count is what the row prints.
func TestRunSequentialRepeatsAShortPrompt(t *testing.T) {
	o := testSuiteOptions(t)
	// Three tokens against a 16-position block: nothing can be published without repeating.
	if len(o.prompt) < 3 {
		t.Fatalf("the suite prompt is only %d tokens", len(o.prompt))
	}
	o.prompt = o.prompt[:3]

	const blockSize = 16
	res, err := runSequential(o, 4, engine.Options{
		Workers: 2, BlockSize: blockSize, Blocks: 4096, PrefixCache: true,
	})
	if err != nil {
		t.Fatalf("runSequential: %v", err)
	}
	if res.prefixReused == 0 {
		t.Errorf("a 3-token prompt against a %d-position block reused nothing; the prompt has "+
			"to be repeated before anything can be published", blockSize)
	}
	// At least one whole block's worth of positions came from the shared prefix, which cannot
	// happen unless the prompt spans one.
	if res.prefixReused < blockSize {
		t.Errorf("reused %d positions, want at least one %d-position block",
			res.prefixReused, blockSize)
	}
}

// TestSuiteReuseRowPadsAShortPrompt covers the repetition runSequential applies to the caller's
// prompt.
//
// A prompt shorter than one block publishes nothing to the prefix index, so the reuse row would
// measure zero however the cache is configured. Repeating it to two blocks is what makes the row
// meaningful. The existing fixture's prompt is 90 tokens -- far past both one block and two -- so
// the loop never runs there and lowering the threshold from 2*block to 1*block changed nothing
// measurable.
//
// This test uses a three-token prompt, so the loop has to run, and asserts the row reports reuse
// across at least one block per request: four requests reusing one block each is the floor.
func TestSuiteReuseRowPadsAShortPrompt(t *testing.T) {
	path := suiteModel(t)
	tok, cfg, err := benchLoad(path)
	if err != nil {
		t.Fatalf("benchLoad: %v", err)
	}
	prompt := tok.Encode("hi", cfg.BOSID >= 0)
	if n := len(prompt); n >= 16 {
		t.Fatalf("the prompt is already %d tokens, so it does not need padding", n)
	}

	o := suiteOptions{model: path, prompt: prompt, maxTok: 4, workers: 1}
	var sc scenario
	for _, c := range scenarioList() {
		if c.name == "prefix reuse x4" {
			sc = c
		}
	}
	if sc.run == nil {
		t.Fatal("scenarioList has no prefix reuse row")
	}
	res, err := sc.run(o)
	if err != nil {
		t.Fatalf("%s: %v", sc.name, err)
	}
	// Four requests over a two-block prompt: the first fills it, and each of the other three reuses
	// both blocks, so four requests' worth is 96 positions -- measured. Padding to one block instead
	// reports 48, which is why the floor is two blocks rather than one.
	//
	// The number is not asserted exactly: it is the model's block count times the requests that
	// reused, and a different fixture would report a different total. What the threshold pins is
	// that the padded prompt spans two blocks, which is what the row's NOTE promises the reader.
	// Measured on this fixture: two blocks of padding report 96 reused positions and one block
	// reports 48, so the floor that separates them is above 48 and at or below 96. Three requests
	// reuse both blocks and the first reuses none, which is where 96 comes from; padding to one
	// block halves it.
	const twoBlocksPerRequest = 96
	if res.prefixReused <= twoBlocksPerRequest/2 {
		t.Errorf("a %d-token prompt reported %d reused positions; padding to one block instead of "+
			"two reports 48, so the prompt was padded to one block and the NOTE's \"two blocks\" "+
			"disagrees with the measurement", len(prompt), res.prefixReused)
	}
}

// TestResolveWorkers covers the default the printed header reports.
//
// The count is measurement metadata: a table pasted into a document says how many kernel workers
// produced it, so the header has to agree with what the run actually used. Zero and negative both mean
// "one per CPU", which is what the engine does when given none.
func TestResolveWorkers(t *testing.T) {
	cpus := runtime.GOMAXPROCS(0)

	// Spelled out rather than table-driven, because the default's value is the machine's CPU count and
	// a literal in a table would pin this test to the machine it was written on.
	if got := resolveWorkers(0); got != cpus {
		t.Errorf("resolveWorkers(0) = %d, want the CPU count %d", got, cpus)
	}
	if got := resolveWorkers(-5); got != cpus {
		t.Errorf("resolveWorkers(-5) = %d, want the CPU count %d", got, cpus)
	}
	// An explicit count is used as given, including one below the CPU count and one above it.
	if got := resolveWorkers(1); got != 1 {
		t.Errorf("resolveWorkers(1) = %d, want 1", got)
	}
	if got := resolveWorkers(cpus + 7); got != cpus+7 {
		t.Errorf("resolveWorkers(%d) = %d, want it unchanged", cpus+7, got)
	}
}

// TestResolveWorkersIsWhatTheHeaderPrints ties the helper to the line a reader sees.
//
// The header line is the only place this value surfaces, so a helper that returns the right number
// while the header interpolates something else would still be wrong.
func TestResolveWorkersIsWhatTheHeaderPrints(t *testing.T) {
	src, err := os.ReadFile("bench_suite.go")
	if err != nil {
		t.Fatalf("reading bench_suite.go: %v", err)
	}
	// The printf that emits the header must take its fourth argument from resolveWorkers' result.
	if !strings.Contains(string(src), "workers := resolveWorkers(o.workers)") {
		t.Error("the suite header no longer derives its worker count from resolveWorkers")
	}
	if strings.Contains(string(src), "workers := o.workers") {
		t.Error("the suite still uses the raw option for its header")
	}
}

// TestBenchPropagatesSetupErrors covers the failures that happen before any measurement.
//
// Each is a flag the operator got wrong, and each must reach the caller rather than being defaulted
// over: a bad --policy silently replaced by fifo would run the wrong experiment and print a table that
// looks right.
//
// The assertions are on what the error says, because a caller sees only the message; the policy
// error's own text comes from internal/engine, so the check is that it survives the trip unchanged.
func TestBenchPropagatesSetupErrors(t *testing.T) {
	model := suiteModel(t)

	t.Run("an unknown scheduling policy", func(t *testing.T) {
		err := cmdBench([]string{"--model", model, "--policy", "bogus", "--max-tokens", "1", "--concurrency", "1"})
		if err == nil {
			t.Fatal("bench ran with an unknown policy")
		}
		// The value the operator typed and the accepted set, both from the engine's message.
		for _, want := range []string{"bogus", "fifo", "shortest", "fair"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not mention %q", err, want)
			}
		}
	})

	t.Run("a prompt that fills the context", func(t *testing.T) {
		// The fixture's context is 512 positions; "a " is one token per repeat and the fixture's two
		// special tokens add two more, so this is 522 and comfortably past the limit.
		long := strings.Repeat("a ", 520)
		err := cmdBench([]string{"--model", model, "--prompt", long, "--max-tokens", "1", "--concurrency", "1"})
		if err == nil {
			t.Fatal("bench accepted a prompt longer than the context")
		}
		if !strings.Contains(err.Error(), "context is") {
			t.Errorf("error %q does not state the context limit", err)
		}
		// The two counts, since the message carries both and a reader compares them.
		if !strings.Contains(err.Error(), "512") {
			t.Errorf("error %q does not state the context size", err)
		}
	})

	// The boundary: 510 repeats encode to exactly 512 tokens, filling the context with no room to
	// generate. This is what pins `>=`; a 513-token prompt is refused either way.
	t.Run("a prompt that exactly fills the context", func(t *testing.T) {
		full := strings.Repeat("a ", 510)
		err := cmdBench([]string{"--model", model, "--prompt", full, "--max-tokens", "1", "--concurrency", "1"})
		if err == nil {
			t.Fatal("bench accepted a prompt with no room left to generate")
		}
		if !strings.Contains(err.Error(), "context is 512") {
			t.Errorf("error %q does not state the context size", err)
		}
	})
}

// TestRunRefusesAnEmptyEncoding covers the guard that fires before the model runs.
//
// A model that declares no BOS, given an empty prompt, encodes to nothing. Without the check the
// engine is handed an empty prompt and the failure surfaces there as a context complaint, which names
// neither the prompt nor the reason.
//
// built through the same fixture entry point the other CLI tests use; BOSID -1 is what removes the only
// token this prompt would have had.
func TestRunRefusesAnEmptyEncoding(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64, BOSID: -1}.Write(t, t.TempDir())
	err := cmdRun([]string{"--model", path, "--max-tokens", "1", ""})
	if err == nil {
		t.Fatal("run accepted a prompt that encodes to nothing")
	}
	if !strings.Contains(err.Error(), "no tokens") {
		t.Errorf("error %q does not say the prompt encoded to nothing", err)
	}
}

// TestRunSeedZeroIsNotSeedZero covers the sentinel meaning "pick a seed".
//
// --seed 0 asks for a random seed rather than the reproducible zero, so two runs with it must be free
// to differ while two runs at --seed 7 must not. The check is on the sampler's behaviour through a real
// generation, since that is what the flag controls.
func TestRunSeedZeroIsNotSeedZero(t *testing.T) {
	model := suiteModel(t)

	// The same fixed seed twice: identical output, because the sampler is the only source of
	// variation and it is seeded the same way.
	first := captureStdout(t, func() {
		_ = cmdRun([]string{"--model", model, "--max-tokens", "8", "--temp", "1.2", "--seed", "7", "Once"})
	})
	second := captureStdout(t, func() {
		_ = cmdRun([]string{"--model", model, "--max-tokens", "8", "--temp", "1.2", "--seed", "7", "Once"})
	})
	if first != second {
		t.Errorf("two runs with --seed 7 differ:\n%q\n%q", first, second)
	}

	// Seed zero picks from the clock, so two runs must differ. Sixty-four samples at temperature 1.5
	// over a 32k vocabulary: identical text would mean both runs used the same seed, which is exactly
	// the defect this covers.
	//
	// Skipping on equality would defeat it -- the mutation that removes the sentinel makes the two runs
	// identical by construction, and a skip reports that as a pass.
	a := captureStdout(t, func() {
		_ = cmdRun([]string{"--model", model, "--max-tokens", "64", "--temp", "1.5", "--seed", "0", "Once"})
	})
	b := captureStdout(t, func() {
		_ = cmdRun([]string{"--model", model, "--max-tokens", "64", "--temp", "1.5", "--seed", "0", "Once"})
	})
	if a == b {
		t.Errorf("two --seed 0 runs produced identical text, so the sentinel did not reach the clock: %q", a)
	}
}

// writeFailer accepts writes until it has taken limit of them, then fails.
//
// A writer that always fails cannot show whether later writes are skipped: the first failure already
// ends the table, so the skip guard never gets a chance to matter.
type writeFailer struct {
	limit int
	took  int // successful writes
	calls int // every write, including the failing ones
	err   error
}

func (w *writeFailer) Write(p []byte) (int, error) {
	w.calls++
	if w.took >= w.limit {
		return 0, w.err
	}
	w.took++
	return len(p), nil
}

// TestWriteTableReportsAWriteFailure covers the path the suite takes when its output is gone.
//
// Nothing could reach it: the table is written to os.Stdout, which a test cannot make fail. It matters
// because a broken pipe is an ordinary outcome -- a benchmark piped into head does it -- and the caller
// needs to be told rather than getting a truncated table reported as success.
func TestWriteTableReportsAWriteFailure(t *testing.T) {
	wantErr := errors.New("broken pipe")
	rows := []suiteRow{
		{scenario: scenario{name: "first"}},
		{scenario: scenario{name: "second"}},
	}

	// Fails on the very first write: the header never lands.
	err := writeTable(&writeFailer{limit: 0, err: wantErr}, rows)
	if err == nil {
		t.Fatal("writeTable reported success with a failing writer")
	}
	if !errors.Is(err, wantErr) {
		t.Errorf("error %v does not carry the writer's error %v", err, wantErr)
	}
	if !strings.Contains(err.Error(), "write table") {
		t.Errorf("error %q does not say the table is what could not be written", err)
	}

	// Fails after the header: the rows are what goes missing, and the error is the same shape.
	err = writeTable(&writeFailer{limit: 1, err: wantErr}, rows)
	if err == nil {
		t.Fatal("writeTable reported success when the rows could not be written")
	}
	if !errors.Is(err, wantErr) {
		t.Errorf("error %v does not carry the writer's error %v", err, wantErr)
	}
}

// TestWriteTableSkipsWritesAfterTheFirstFailure covers the guard that stops appending to a broken
// stream.
//
// The first error is the one reported, and later rows must not each pay another failing write. The
// count is what distinguishes the guard from its absence: without it the writer sees one call per row.
func TestWriteTableSkipsWritesAfterTheFirstFailure(t *testing.T) {
	rows := make([]suiteRow, 0, 20)
	for i := range 20 {
		rows = append(rows, suiteRow{scenario: scenario{name: strings.Repeat("x", i)}})
	}
	// tabwriter buffers, so a failing Write only surfaces once a line is flushed. Two successful
	// writes then a failure is enough to make the guard observable: with it, no further Write happens.
	w := &writeFailer{limit: 2, err: errors.New("gone")}
	if err := writeTable(w, rows); err == nil {
		t.Fatal("writeTable reported success with a failing writer")
	}
	// The count of every write is what distinguishes the guard from its absence. Without it the
	// writer is called once per flushed line -- measured at 20 for ten rows -- so a bound of 4 sits
	// well clear of both outcomes.
	if w.calls > 4 {
		t.Errorf("the writer was called %d times; writes after the first failure were not skipped",
			w.calls)
	}
}

// TestRunEchoPrintsThePromptFirst covers --echo, which had no test.
//
// Without it a run prints only what the model generated, so a reader copying the output into a
// document gets an answer with no question. With it the prompt is printed first and the generated text
// follows, which is also why the code decodes only the generated part -- printing the whole output would
// repeat the prompt.
func TestRunEchoPrintsThePromptFirst(t *testing.T) {
	model := suiteModel(t)
	const prompt = "Once upon a time"

	withEcho := captureStdout(t, func() {
		if err := cmdRun([]string{"--model", model, "--max-tokens", "4", "--echo", prompt}); err != nil {
			t.Errorf("run --echo: %v", err)
		}
	})
	withoutEcho := captureStdout(t, func() {
		if err := cmdRun([]string{"--model", model, "--max-tokens", "4", prompt}); err != nil {
			t.Errorf("run: %v", err)
		}
	})

	if !strings.HasPrefix(withEcho, prompt) {
		t.Errorf("--echo output does not start with the prompt: %q", withEcho)
	}
	if strings.HasPrefix(withoutEcho, prompt) {
		t.Errorf("output without --echo starts with the prompt: %q", withoutEcho)
	}
	// And the prompt appears exactly once: the generated part must not repeat it.
	if n := strings.Count(withEcho, prompt); n != 1 {
		t.Errorf("--echo printed the prompt %d times, want 1: %q", n, withEcho)
	}
}

// TestRunSeedSelectsTheStream adds the half TestRunSeedZeroIsNotSeedZero cannot check.
//
// That test compares two runs at the same seed, which stays green if the seed is ignored
// entirely -- an ignored seed also produces two identical runs. The property a seed actually
// promises is that it selects the stream, so two different seeds must produce different text.
//
// The greedy half is the doc comment on --seed made checkable: it says seed "has no effect
// without --temp, because greedy decoding is already deterministic". Three different seeds at a
// fixed budget are compared for equality, and only the absence of --temp makes that the right
// answer.
func TestRunSeedSelectsTheStream(t *testing.T) {
	model := suiteModel(t)
	run := func(args ...string) string {
		return captureStdout(t, func() {
			_ = cmdRun(append([]string{"--model", model}, args...))
		})
	}

	// Two different seeds must not agree. Temperature 1.5 and a 48-token budget put enough
	// samples on the stream that agreement would mean the seed is not reaching the sampler.
	const budget = "48"
	a := run("--max-tokens", budget, "--temp", "1.5", "--seed", "7", "Once")
	b := run("--max-tokens", budget, "--temp", "1.5", "--seed", "8", "Once")
	if a == b {
		t.Errorf("--seed 7 and --seed 8 produced identical text, so the seed does not select the "+
			"stream: %q", a)
	}

	// And with no temperature the flag must do nothing: greedy is already deterministic, so
	// every seed is the same run.
	first := run("--max-tokens", "24", "--seed", "1", "Once")
	for _, seed := range []string{"2", "99", "12345"} {
		got := run("--max-tokens", "24", "--seed", seed, "Once")
		if got != first {
			t.Errorf("--seed %s changed greedy output, so the flag is not ignored without --temp "+
				"as its help says:\nseed 1: %q\nseed %s: %q", seed, first, seed, got)
		}
	}
}

// TestRowLineKeepsAFailureOnOneLine covers what a multi-line error would do to the table.
//
// tabwriter ends a row at a newline, so an error carrying one splits the failure across two
// rows: the measurement cells of the second row are whatever the message happened to contain,
// and every row below shifts out of alignment. One message makes the whole table unreadable,
// which is worse than a truncated one because a reader trusts a table that lines up.
//
// The errors bench receives today are single-line -- ErrQueueFull and ErrPoolExhausted and the
// wrappers around them -- so this was invisible rather than wrong. It is pinned because the
// table's readability depends on a property of code owned by another package.
func TestRowLineKeepsAFailureOnOneLine(t *testing.T) {
	sc := scenario{name: "single"}
	for _, c := range []struct {
		name string
		err  error
		want []string
	}{
		{"a newline", errors.New("first line\nsecond line"), []string{"first line", "second line"}},
		{"a carriage return", errors.New("first\rsecond"), []string{"first", "second"}},
		{"a tab", errors.New("first\tsecond"), []string{"first", "second"}},
		{"all three", errors.New("alpha\nbeta\ngamma\tdelta"), []string{"alpha", "beta", "gamma", "delta"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := rowLine(sc, result{}, c.err)
			if strings.ContainsAny(got, "\n\r") {
				t.Errorf("rowLine = %q, which ends the row early or rewinds the line", got)
			}
			// Six cells exactly. A tab inside the message would have made seven, which
			// tabwriter renders as a column that nothing else has.
			if n := len(strings.Split(got, "\t")); n != 6 {
				t.Errorf("rowLine produced %d cells, want 6: %q", n, got)
			}
			// Every word of the message survives, so the failure is still diagnosable after
			// the separator was replaced.
			for _, want := range c.want {
				if !strings.Contains(got, want) {
					t.Errorf("rowLine = %q does not contain %q", got, want)
				}
			}
		})
	}
}

// TestScenarioFailureDoesNotStopTheSuite covers the property runScenarios exists for.
//
// A scenario that fails has to print its reason in the NOTE column and let the rest run: the
// table is the deliverable, and a row missing because an earlier row failed reads as a scenario
// that was never asked for. The exit status still has to be non-zero, or a failed row would
// read as a successful run.
//
// Nothing pinned the continuation before this: the existing suite test asserts that no scenario
// fails on a good model and that a failed one prints "failed:", both of which hold whether or
// not the loop breaks early.
func TestScenarioFailureDoesNotStopTheSuite(t *testing.T) {
	var ran []string
	ok := func(name string) scenario {
		return scenario{name: name, run: func(suiteOptions) (result, error) {
			ran = append(ran, name)
			return result{}, nil
		}}
	}
	broken := scenario{name: "broken", run: func(suiteOptions) (result, error) {
		ran = append(ran, "broken")
		return result{}, errors.New("deliberate")
	}}

	// runScenarios writes the table to os.Stdout, so the output is captured rather than
	// asserted on: what this test is about is which scenarios ran and what was returned.
	out := captureStdout(t, func() {
		err := runScenarios(suiteOptions{model: "m", maxTok: 1},
			[]scenario{ok("first"), broken, ok("last")})
		if err == nil {
			t.Error("runScenarios reported success despite a failed scenario")
		} else if !strings.Contains(err.Error(), "broken") {
			t.Errorf("error %q does not name the failed scenario", err)
		}
	})

	if want := []string{"first", "broken", "last"}; fmt.Sprint(ran) != fmt.Sprint(want) {
		t.Errorf("scenarios that ran = %v, want %v: a failure stopped later scenarios", ran, want)
	}
	// Both the failure and the scenario after it have to reach the table.
	for _, want := range []string{"failed: deliberate", "last"} {
		if !strings.Contains(out, want) {
			t.Errorf("the table does not contain %q:\n%s", want, out)
		}
	}
}
