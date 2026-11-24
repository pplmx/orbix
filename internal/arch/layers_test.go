package arch

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
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
//
// One package checks its own tests as well -- internal/server/imports_test.go refuses an import of
// internal/llama in any file including tests, because a test that reaches for the model package is
// how a production import creeps back. That is a stricter rule than this one and it applies to one
// package; this check covers the whole tree, production code only.
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

// TestArchitectureDocAgreesWithTheLayerTable keeps the two copies of the order from drifting.
//
// docs/architecture.md lists the packages in the order a reader should meet them, with the
// comment that the concepts "arrive in the order a request travels through them". That is a
// dependency order, so it has to agree with the table above: a package listed before something it
// imports sends a reader into code they have not met yet.
//
// It disagreed once. The document had engine at 6 and chat at 7, while internal/engine imports
// internal/chat -- so a reader following the list reached the scheduler before the renderer it
// calls. Nothing noticed, because the list was prose.
//
// Only the relative order is checked, not the rank numbers: the document puts f16 and quant on one
// row because a reader meets them together, and a check that demanded a distinct rank per package
// would reject a table that is correct.
func TestArchitectureDocAgreesWithTheLayerTable(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	doc, err := os.ReadFile(filepath.Join(root, "..", "..", "docs", "architecture.md"))
	if err != nil {
		t.Fatalf("reading architecture.md: %v", err)
	}

	// docPos is each package's row number, the smallest for a package named on two rows.
	docPos := make(map[string]int)
	for line := range strings.SplitSeq(string(doc), "\n") {
		fields := strings.Split(line, "|")
		if len(fields) < 3 {
			continue
		}
		rank, err := strconv.Atoi(strings.TrimSpace(fields[1]))
		if err != nil {
			continue // the header or the divider
		}
		for _, m := range internalLink.FindAllStringSubmatch(fields[2], -1) {
			name := m[1]
			if prev, seen := docPos[name]; seen && prev <= rank {
				continue
			}
			docPos[name] = rank
		}
	}
	if len(docPos) < 9 {
		t.Fatalf("read %d packages from the document's table, too few to be the real one -- "+
			"the parser probably stopped matching", len(docPos))
	}

	// Every pair the document orders must be ordered the same way by the layer table, unless the
	// document puts them on the same row.
	for up, upPos := range docPos {
		for down, downPos := range docPos {
			if upPos >= downPos {
				continue // only the document's own ordering
			}
			upLayer, ok1 := layer[up]
			downLayer, ok2 := layer[down]
			if !ok1 || !ok2 {
				t.Errorf("the document lists internal/%s and internal/%s, and the layer table is "+
					"missing one of them", up, down)
				continue
			}
			if upLayer > downLayer {
				t.Errorf("the document lists internal/%s (#%d) before internal/%s (#%d), but the "+
					"layer table has them the other way round", up, upPos, down, downPos)
			}
			// And the import direction: the later package may import the earlier one, never the
			// reverse, or a reader following the list meets the caller before the callee.
			if imports(t, root, up, down) {
				t.Errorf("the document lists internal/%s (#%d) before internal/%s (#%d), but "+
					"internal/%s imports internal/%s -- a reader meets the caller first",
					up, upPos, down, downPos, up, down)
			}
		}
	}
}

// internalLink matches an internal package link in a Markdown table cell.
var internalLink = regexp.MustCompile(`internal/([a-z0-9]+)/doc\.go`)

// imports reports whether package up imports package down, reading production files only.
func imports(t *testing.T, root, up, down string) bool {
	t.Helper()
	dir := filepath.Join(root, "..", up)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir(%s): %v", dir, err)
	}
	want := module + "internal/" + down
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(dir, name), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("ParseFile(%s): %v", name, err)
		}
		for _, spec := range f.Imports {
			p, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				continue
			}
			if p == want {
				return true
			}
		}
	}
	return false
}
