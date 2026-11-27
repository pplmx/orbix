package llama

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
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

// TestGemma2HasAWindowWhenTheFileOmitsTheKey covers the one architecture whose window is part of
// what it is rather than an optional refinement.
//
// llama.cpp's src/models/gemma2.cpp sets `swa_type = LLAMA_SWA_TYPE_STANDARD` and
// `n_swa = 4096` *before* it reads `attention.sliding_window`, so the read is an override and not
// the only source. Hugging Face's Gemma2Config carries the same 4096 as the field default. A file
// that omits the key is therefore a converter predating it, not a checkpoint trained without a
// window -- and treating the absence as "unbounded" made every layer attend over the whole context
// instead of every second one.
//
// The default has to stay per-architecture: Mistral is the opposite case, where an absent key or a
// zero means full attention, which is why the other architectures must not pick this up.
func TestGemma2HasAWindowWhenTheFileOmitsTheKey(t *testing.T) {
	load := func(arch string, extra ...gguftest.Model) Config {
		t.Helper()
		m := gguftest.Model{Arch: arch, Layers: 4, Hidden: 32, Heads: 4, FFN: 64,
			QKNorm: arch == "gemma3", PostNorms: arch != "llama"}
		if len(extra) > 0 {
			m = extra[0]
			m.Arch = arch
		}
		path := m.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		cfg, err := LoadConfig(f)
		if err != nil {
			t.Fatalf("LoadConfig: %v", err)
		}
		return cfg
	}

	// gemma2 with no window key gets 4096, and every second layer is the one that is bounded.
	cfg := load("gemma2")
	if cfg.WindowLen != 4096 {
		t.Errorf("gemma2 with no sliding_window key: WindowLen = %d, want llama.cpp's 4096 "+
			"default", cfg.WindowLen)
	}
	for l, want := range []int{4097, 0, 4097, 0} {
		if got := windowFor(cfg, l); got != want {
			t.Errorf("gemma2 layer %d window = %d, want %d", l, got, want)
		}
	}

	// The file still wins when it says something else: 2048 in the file is 2048 here.
	cfg = load("gemma2", gguftest.Model{Layers: 4, Hidden: 32, Heads: 4, FFN: 64,
		PostNorms: true, WindowLen: 2048})
	if cfg.WindowLen != 2048 {
		t.Errorf("gemma2 with sliding_window 2048: WindowLen = %d, want the file's 2048",
			cfg.WindowLen)
	}

	// And no other architecture inherits it. llama and gemma3 both leave an absent key alone.
	for _, arch := range []string{"llama", "gemma3"} {
		if got := load(arch).WindowLen; got != 0 {
			t.Errorf("%s with no sliding_window key: WindowLen = %d, want 0 -- the default is "+
				"gemma2's and applying it to any windowed architecture would bound a model that "+
				"trained with full attention", arch, got)
		}
	}
}

// TestModelsDocExplainsThePatternElementTypes keeps the note about a genuine upstream mismatch from
// being tidied away.
//
// The paragraph it guards describes two implementations that cannot read each other's output: the
// converter writes a BOOL array, llama.cpp's reader is get_arr<uint32_t> and throws on BOOL. That
// is the kind of fact a later editor reads as an over-explanation and deletes -- and deleting it
// loses the reason the loader accepts both forms instead of only the one this repository's own
// writer produces.
func TestModelsDocExplainsThePatternElementTypes(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "docs", "models.md"))
	if err != nil {
		t.Fatalf("reading docs/models.md: %v", err)
	}
	// Compared with paragraph wrapping collapsed, because the document caps lines at 88 columns and
	// every one of these phrases straddles a break. Requiring them on one line would make the guard
	// fail the moment the prose was re-wrapped for an unrelated reason -- and then it gets deleted
	// rather than fixed.
	doc := strings.ToLower(strings.Join(strings.Fields(string(raw)), " "))
	for _, want := range []string{
		"attention.sliding_window_pattern",
		"sequence[bool]",
		"get_arr<uint32_t>",
		"neither side can read what the other writes",
	} {
		if !strings.Contains(doc, want) {
			t.Errorf("docs/models.md no longer states %q; the loader accepts both element types "+
				"for a reason, and that reason has to stay written down", want)
		}
	}
}
