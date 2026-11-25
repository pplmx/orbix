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
