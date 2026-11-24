package server

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// forbiddenImports is what this package must not depend on.
//
// internal/server is the HTTP layer: it turns JSON into engine submissions and engine events
// back into JSON or server-sent events. It should not know how a KV cache is laid out, and
// before this list existed it did -- four of its files imported internal/llama to name a
// sampler, a bias entry and a log-probability alternative. Which sampler a request gets is a
// scheduling decision; naming log-probability alternatives is a protocol concern. Neither needs
// the model package, and the import made every one of those files a place a cache change could
// ripple into.
//
// The types now live in internal/engine (Sampler, Alternative, TokenBias), and llama's own
// implementations satisfy them structurally, so this removal cost no capability.
//
// A comment saying "do not import llama" would not survive a refactor; this does. It is the
// same reasoning as the dependency-direction rule in docs/architecture.md, except that rule was
// prose and this one fails a build.
//
// internal/arch/layers_test.go enforces the same edge from the other side -- it walks every
// package and refuses a backward dependency. The two are not duplicates: that one covers the whole
// tree and reads production files only, while this one covers this package and counts tests, which
// is stricter because a test is the first place a forbidden import appears.
var forbiddenImports = map[string]string{
	"internal/llama": "the HTTP layer decides sampling and protocol, not cache or tensor layout; " +
		"engine owns the types for both (see internal/engine/sampler.go)",
}

// TestPackageDoesNotImportTheModelPackage walks this directory's source files.
//
// It parses rather than greps, because a grep for "llama" matches the comments that cite
// llama.cpp as a reference implementation -- of which there are several, and all of them should
// stay.
func TestPackageDoesNotImportTheModelPackage(t *testing.T) {
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}

	checked := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") {
			continue
		}
		// Test files count. A test that imports the model package to build a sampler is the
		// same dependency, and it is how the production import would creep back: the test
		// compiles first, then the code it exercises starts using the same helper.
		checked++
		path := filepath.Join(dir, name)
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("ParseFile(%s): %v", name, err)
		}
		for _, spec := range f.Imports {
			p, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				t.Fatalf("%s: unquoting %s: %v", name, spec.Path.Value, err)
			}
			for forbidden, why := range forbiddenImports {
				if strings.HasSuffix(p, "/"+forbidden) {
					t.Errorf("%s imports %s: %s", name, p, why)
				}
			}
		}
	}

	// A test that checked nothing passes vacuously, which is the failure mode this file exists
	// to prevent one level up -- measured, the first version of a similar guard in this
	// repository silently matched no files and reported success.
	if checked < 5 {
		t.Fatalf("only %d source files were checked, so this is probably looking at the wrong "+
			"directory", checked)
	}
}

// TestForbiddenImportsActuallyMatchesASuffix is a check on the test itself.
//
// The comparison is a suffix match so that a module path change does not break it, and a suffix
// match that was too loose would flag an unrelated import while one that was too strict would
// match nothing -- which is how the guard above would pass while checking nothing.
func TestForbiddenImportsActuallyMatchesASuffix(t *testing.T) {
	const want = "github.com/pplmx/orbix/internal/llama"
	matched := false
	for forbidden := range forbiddenImports {
		if strings.HasSuffix(want, "/"+forbidden) {
			matched = true
		}
	}
	if !matched {
		t.Errorf("no entry in forbiddenImports matches %q, so the walk above would never fire",
			want)
	}
}
