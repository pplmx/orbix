package arch

import (
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// TestDocSymbolsExist checks that every `package.Symbol` the docs cite is exported by that package.
//
// A rename reaches the code, the tests and the doc.go that describes it, but not necessarily the
// prose in docs/ -- and rumdl has nothing to say about an identifier. The reference then reads as
// authoritative while pointing at something that no longer compiles, which is the same failure
// testnames_test.go guards for test names and this is the other half of.
//
// Exported set only: a doc citing an unexported name would be pointing a reader at something they
// cannot call, so the absence is worth failing on as well.
func TestDocSymbolsExist(t *testing.T) {
	root := filepath.Join("..", "..")

	// Every internal package, so a doc reference to the wrong package is caught too: engine.Foo
	// passes only if engine exports Foo, not if some other package does.
	pkgs := map[string][]string{}
	for _, dir := range []string{"engine", "llama", "gguf", "quant", "kernel", "tokenize", "chat", "otel", "server", "gguftest"} {
		// From the module root: go doc resolves its argument against the working directory, and
		// this test runs in internal/arch.
		cmd := exec.Command("go", "doc", "-all", "./internal/"+dir)
		cmd.Dir = root
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("go doc ./internal/%s: %v", dir, err)
		}
		var names []string
		for _, line := range strings.Split(string(out), "\n") {
			// Both a plain declaration and a method: go doc writes a method as
			// `func (e *Engine) Pump()`, and matching only the first form reported every method
			// the docs cite as missing.
			if m := regexp.MustCompile(`^(?:func|type|var|const) (?:[A-Z][A-Za-z0-9_]*|\([^)]*\) [A-Z][A-Za-z0-9_]*)`).FindStringSubmatch(line); m != nil {
				parts := strings.Fields(m[0])
				names = append(names, strings.TrimSuffix(parts[len(parts)-1], "("))
			}
		}
		// quant exports four; the floor is here to catch a go doc that returned nothing useful,
		// not to require a particular size.
		if len(names) < 3 {
			t.Fatalf("only %d exported names found in internal/%s, so this check is looking at the wrong thing", len(names), dir)
		}
		pkgs[dir] = names
	}

	files, err := filepath.Glob(filepath.Join(root, "docs", "*.md"))
	if err != nil {
		t.Fatalf("globbing docs: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("no docs found")
	}

	names := mapsKeys(pkgs)
	slices.Sort(names)
	ref := regexp.MustCompile("`(" + strings.Join(names, "|") + `)\.([A-Z][A-Za-z0-9_]*)` + "`")
	seen := 0
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("reading %s: %v", f, err)
		}
		for _, m := range ref.FindAllStringSubmatch(string(raw), -1) {
			seen++
			pkg, sym := m[1], m[2]
			if !slices.Contains(pkgs[pkg], sym) {
				t.Errorf("%s cites %s.%s, which internal/%s does not export", filepath.Base(f), pkg, sym, pkg)
			}
		}
	}
	if seen < 20 { // measured: 25
		t.Fatalf("only %d package-qualified references were found, so the pattern is not matching", seen)
	}
}

func mapsKeys(m map[string][]string) []string { return slices.Collect(maps.Keys(m)) }
