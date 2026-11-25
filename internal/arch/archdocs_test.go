package arch

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"testing"
)

// TestModelsDocListsEveryArchitecture checks docs/models.md against the architecture table in
// internal/llama/config.go, in both directions.
//
// The table is how a person decides whether a checkpoint will load. An entry naming an
// architecture the code does not know is the same failure as a flag that no longer exists: the
// reader tries it and gets a refusal. The other direction is silent -- an architecture accepted
// but undocumented is one nobody outside this repository will try.
//
// Read from the source rather than through the package: archSpecs is unexported and the only
// exported accessor returns a joined string, which would make this a test of the formatting.
func TestModelsDocListsEveryArchitecture(t *testing.T) {
	root := filepath.Join("..", "..")

	config, err := os.ReadFile(filepath.Join(root, "internal", "llama", "config.go"))
	if err != nil {
		t.Fatalf("read config.go: %v", err)
	}
	block := archSpecsBlock.FindStringSubmatch(string(config))
	if block == nil {
		t.Fatal("could not find the archSpecs table in config.go; the declaration changed shape")
	}
	var code []string
	for _, m := range archKey.FindAllStringSubmatch(block[1], -1) {
		code = append(code, m[1])
	}
	slices.Sort(code)
	if len(code) < 5 {
		t.Fatalf("found %d architectures in archSpecs, too few to be checking anything", len(code))
	}

	doc, err := os.ReadFile(filepath.Join(root, "docs", "models.md"))
	if err != nil {
		t.Fatalf("read models.md: %v", err)
	}
	documented := map[string]bool{}
	for _, m := range archTableRow.FindAllStringSubmatch(string(doc), -1) {
		documented[m[1]] = true
	}
	if len(documented) < 5 {
		t.Fatalf("found %d architectures in models.md, too few to be checking anything", len(documented))
	}

	for _, a := range code {
		if !documented[a] {
			t.Errorf("archSpecs has %q and docs/models.md does not list it", a)
		}
	}
	for a := range documented {
		if !slices.Contains(code, a) {
			t.Errorf("docs/models.md lists %q and archSpecs has no such entry", a)
		}
	}
}

// archSpecsBlock captures the body of the archSpecs map literal.
var archSpecsBlock = regexp.MustCompile(`(?s)var archSpecs = map\[string\]archSpec\{(.*?)\n}`)

// archKey matches a key in that literal. Commented lines cannot match because the quote is not at
// the start of a stripped line, and the table's comments are indented the same way as its entries.
var archKey = regexp.MustCompile(`(?m)^\t"([a-z0-9]+)":`)

// archTableRow matches a row of the table whose Architectures cell is a single backticked name.
// Rows like "`llama` with experts" are skipped by design: they name an architecture that is
// already listed, with a property rather than a new entry.
var archTableRow = regexp.MustCompile("(?m)^\\|\\s*\\*\\*[^|]+\\*\\*\\s*\\|\\s*`([a-z0-9]+)`\\s*\\|")
