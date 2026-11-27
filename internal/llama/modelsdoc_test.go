package llama

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// TestModelsDocListsEverySupportedArchitecture keeps the table and the table disagreeing from
// going unnoticed.
//
// docs/models.md opens with the architectures the project supports, as a table a user reads to
// decide whether their checkpoint will load. archSpecs is the same list in code: LoadConfig looks
// the file's general.architecture up there and refuses anything it does not find. Adding an entry
// without a row leaves the document describing a smaller surface than the binary has; removing one
// without a row leaves it promising a checkpoint that will be refused.
//
// Neither breaks a build, and neither is visible unless someone compares the two by hand.
//
// internal/arch/archdocs_test.go checks the same pair through the source text, and the two are
// not interchangeable: this one counts every backticked word in a cell, so the Mixtral row's
// "`llama` with experts" keeps `llama` documented even if its own row is deleted. That is the
// case the arch version catches and this one misses; keep both.
func TestModelsDocListsEverySupportedArchitecture(t *testing.T) {
	doc, err := os.ReadFile(filepath.Join("..", "..", "docs", "models.md"))
	if err != nil {
		t.Fatalf("reading docs/models.md: %v", err)
	}

	documented := documentedArchitectures(t, string(doc))
	if len(documented) < 5 {
		// A parser that stopped matching would pass the comparison below for the wrong reason.
		t.Fatalf("read %d architectures from the document's table, too few to be the real one",
			len(documented))
	}

	var inCode []string
	for name := range archSpecs {
		inCode = append(inCode, name)
	}
	sort.Strings(inCode)

	for _, name := range inCode {
		if !documented[name] {
			t.Errorf("archSpecs has %q, which docs/models.md does not list", name)
		}
	}
	var inDoc []string
	for name := range documented {
		inDoc = append(inDoc, name)
	}
	sort.Strings(inDoc)
	for _, name := range inDoc {
		if _, ok := archSpecs[name]; !ok {
			t.Errorf("docs/models.md lists %q, which archSpecs does not have -- a checkpoint "+
				"with that architecture is refused at load", name)
		}
	}
}

// documentedArchitectures reads the first table's Architectures column.
//
// The cell may name more than one -- Mixtral's says "`llama` with experts", because it is a
// llama-shaped file with expert tensors rather than an architecture of its own -- so every
// backticked word in the column counts. The first table is the architecture table; the second is
// quantization, and its cells are formats rather than architectures, so reading stops at the row
// whose first cell is "Quantization".
func documentedArchitectures(t *testing.T, text string) map[string]bool {
	t.Helper()
	out := make(map[string]bool)
	for line := range strings.SplitSeq(text, "\n") {
		if !strings.HasPrefix(line, "|") {
			if len(out) > 0 {
				break // the table ended
			}
			continue
		}
		cells := strings.Split(strings.Trim(line, "|"), "|")
		if len(cells) < 3 {
			continue
		}
		if strings.TrimSpace(cells[0]) == "Quantization" {
			break
		}
		for _, m := range tableCode.FindAllStringSubmatch(cells[1], -1) {
			out[m[1]] = true
		}
	}
	return out
}

// tableCode matches a backticked word in a table cell.
var tableCode = regexp.MustCompile("`([a-z][a-z0-9]*)`")

// TestModelsDocStatesWhichCandidatesTheLogitOffsetRulesOut keeps the offset section from drifting
// back into a guess.
//
// docs/models.md used to say the ~0.5 logit difference from llama.cpp was "a different rounding of
// the final norm or the output projection". That was a plausible sentence nobody had measured, and
// measuring it took an afternoon and refuted it: float64 recomputation moves 1e-6, and quantising
// every projection the way llama.cpp does moves 0.128. What replaced it is a list of five
// candidates, each with the observation that rules it out.
//
// The list is a set of claims about code, so it is exactly the kind that goes stale silently. The
// guard checks the claims that are cheap to check -- that each candidate is still named, and that
// the numbers the paragraph quotes still match the ones the experiments produce -- rather than
// pretending to re-run the experiments.
func TestModelsDocStatesWhichCandidatesTheLogitOffsetRulesOut(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "docs", "models.md"))
	if err != nil {
		t.Fatalf("reading docs/models.md: %v", err)
	}
	doc := string(raw)

	// The section exists and still says what the offset is, rather than reverting to a guess.
	if !strings.Contains(doc, "What it is not, checked one candidate at a time") {
		t.Fatal("docs/models.md no longer carries the section listing what the logit offset is" +
			" not; if it was rewritten, update this test rather than deleting it")
	}

	// Each ruled-out candidate, by the name the paragraph uses. A name going missing means either
	// the reasoning was dropped or the experiment was redone, and both want a human to look.
	for _, candidate := range []string{
		"float32 accumulation",
		"q8_0 activation quantisation",
		"RMSNorm arithmetic",
		"the Gemma norm shift",
		"attention scale",
	} {
		if !strings.Contains(doc, candidate) {
			t.Errorf("docs/models.md no longer rules out %q as the source of the logit offset;"+
				" either the claim was dropped or the experiment was repeated, and both need the"+
				" paragraph updated", candidate)
		}
	}

	// The two measured numbers that decide the quantisation candidate. They come from a probe that
	// is not committed, so the document is the only record; a reader who changes one has to change
	// the other or lose the reason the candidate is excluded.
	for _, number := range []string{"-0.128", "+0.071"} {
		if !strings.Contains(doc, number) {
			t.Errorf("docs/models.md no longer quotes %s from the quantisation experiment", number)
		}
	}

	// And the rule that keeps the section honest about what remains.
	if !strings.Contains(strings.ReplaceAll(doc, "\n", " "), "not worth closing") {
		t.Error("docs/models.md no longer states whether the offset is worth closing; the" +
			" paragraph is meant to end in a decision, not in an open question")
	}
}
