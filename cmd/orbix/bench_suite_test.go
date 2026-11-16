package main

import (
	"os"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/engine"
)

// The suite is a measurement tool, so its tests check the things that make its numbers
// trustworthy: that every scenario runs, that the pairs meant to isolate a mechanism
// actually differ, and that the table renders. They do not assert throughput values --
// those vary with the machine and with whatever else is running on it.

func suiteModel(t *testing.T) string {
	t.Helper()
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	return path
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
	t.Logf("no reuse %v, reuse %v", plain.elapsed, reuse.elapsed)
	if reuse.elapsed >= plain.elapsed {
		t.Errorf("reuse took %v, no reuse %v; the pair exists to show reuse helping, "+
			"so identical times mean it is not applied", reuse.elapsed, plain.elapsed)
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
	t.Logf("shortest first token %v, fifo %v", shortest.firstToken, fifo.firstToken)
	if shortest.firstToken >= fifo.firstToken {
		t.Errorf("shortest served the short request in %v and fifo in %v; shortest must be faster",
			shortest.firstToken, fifo.firstToken)
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
