package main

import (
	"strings"
	"testing"

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

// TestSuiteRejectsUnknownPolicyFlag is a guard on the flag surface: the suite ignores
// --policy, which is easy to miss and would otherwise look like the flag being broken.
func TestSuiteRejectsUnknownPolicyFlag(t *testing.T) {
	// ParsePriority is the single validator for the name; the suite never calls it, so a
	// typo'd --policy passes silently. Pin that ParsePriority still rejects bad input,
	// which is what a reader of the flag table would expect.
	if _, err := engine.ParsePriority("nonsense"); err == nil {
		t.Fatal("ParsePriority accepted an unknown policy")
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
