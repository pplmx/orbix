package arch

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// module is the import prefix every internal package shares.
const module = "github.com/pplmx/orbix/"

// layer assigns each internal package its position in the stack, lowest first.
//
// The order is the one docs/architecture.md draws: the container format at the bottom, then the
// numeric helpers, then the operators, then the pieces that understand a language or a model, then
// the scheduler, then HTTP, with the CLI on top. otel is lowest rather than highest because it
// depends on nothing -- it is a cross-cutting concern, not a layer, and putting it at the top would
// make every edge into it look like a violation.
var layer = map[string]int{
	"otel":     0,
	"gguf":     1,
	"f16":      2,
	"quant":    3,
	"kernel":   4,
	"tokenize": 5,
	"chat":     6,
	"llama":    7,
	"engine":   8,
	"server":   9,
}

// allowed holds the exceptions, each with the reason it is not a layering mistake.
//
// An entry here is a claim that the edge is deliberate, and the message says why so the next
// reader does not have to reconstruct it. Nothing is in this map because it was inconvenient to
// fix.
var allowed = map[[2]string]string{}

// packageImports returns the internal packages a directory's non-test files import.
//
// Test files are excluded deliberately: a package's tests may reach for anything they need to
// build a fixture, and internal/gguftest exists for exactly that. The rule is about what ships.
func packageImports(t *testing.T, dir string) map[string]bool {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir(%s): %v", dir, err)
	}
	out := make(map[string]bool)
	any := false
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		any = true
		f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(dir, name), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("ParseFile(%s): %v", name, err)
		}
		for _, spec := range f.Imports {
			p, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				t.Fatalf("%s: unquoting %s: %v", name, spec.Path.Value, err)
			}
			rest, ok := strings.CutPrefix(p, module)
			if !ok {
				continue // standard library or an external dependency
			}
			if inner, ok := strings.CutPrefix(rest, "internal/"); ok {
				// A package path has no slashes past internal/<name> today. Keeping the first
				// segment means a future subpackage is attributed to its parent rather than
				// silently skipped.
				name, _, _ := strings.Cut(inner, "/")
				out[name] = true
			}
			if rest == "cmd/orbix" {
				// The CLI is above everything and imports freely; a package importing it back
				// would be a cycle the compiler already refuses. Recorded so the check below
				// does not report it as unknown.
				out["cmd/orbix"] = true
			}
		}
	}
	if !any {
		// An empty package would pass this test vacuously, which is how an architecture guard
		// silently stops guarding: a wrong directory looks exactly like a clean result.
		t.Fatalf("%s has no non-test Go files, so this check read nothing", dir)
	}
	return out
}

// TestInternalPackagesOnlyImportDownward is the rule AGENTS.md states in prose.
//
// It has been violated once already: internal/server imported internal/llama in four files to name
// a sampler, a bias entry and a log-probability alternative. That edge put a KV-cache package on
// the HTTP layer's import path and was found by reading, not by a gate. This is the gate.
func TestInternalPackagesOnlyImportDownward(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	internal := filepath.Join(root, "..")
	packages := 0
	for name, rank := range layer {
		dir := filepath.Join(internal, name)
		if _, err := os.Stat(dir); err != nil {
			t.Fatalf("layer names a package with no directory: %s", name)
		}
		packages++
		for dep := range packageImports(t, dir) {
			depRank, known := layer[dep]
			if !known {
				t.Errorf("%s imports %s, which is not in the layer table; add it with its rank "+
					"or the rule does not cover it", name, dep)
				continue
			}
			if depRank >= rank {
				if why, ok := allowed[[2]string{name, dep}]; ok {
					t.Logf("%s -> %s allowed: %s", name, dep, why)
					continue
				}
				t.Errorf("%s (layer %d) imports %s (layer %d): dependencies may only point down. "+
					"If this edge is deliberate, add it to the allowed map with the reason.",
					name, rank, dep, depRank)
			}
		}
	}
	if packages < 10 {
		t.Fatalf("only %d packages were checked, so this is probably looking at the wrong tree",
			packages)
	}
}

// TestTheGuardSeesRealImports checks the guard itself against a package known to import others.
//
// A parser that returned nothing would pass the test above for every package. This asserts one
// package's known edges are seen, so "no violations" means something.
func TestTheGuardSeesRealImports(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	engine := filepath.Join(root, "..", "engine")
	got := packageImports(t, engine)
	for _, want := range []string{"llama", "chat", "gguf"} {
		if !got[want] {
			t.Errorf("internal/engine imports internal/%s but the guard did not see it; "+
				"the checker is broken, not the code", want)
		}
	}
	// And one it must not have: the server is above the engine.
	if got["server"] {
		t.Error("internal/engine reported an import of internal/server, which it does not have")
	}
}
