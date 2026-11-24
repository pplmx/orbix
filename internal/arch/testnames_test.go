package arch

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// TestDocsCiteRealTestNames keeps a documented `go test -run` from matching nothing.
//
// docs/models.md and several doc.go files name the test that verifies a claim, because a reader
// checking one of them wants the command rather than the prose. The name is the only link: rename
// the test and the document keeps pointing at it.
//
// The failure mode is worse than a broken link. `go test -run TestDoesNotExist` exits 0 and prints
// "ok" -- no test matched, and the output says success. Measured: docs/models.md said
// TestMoEReachesTheExperts while the test is TestMoEReachesTheExpertsOnARealModel, and the
// documented command reported ok in 0.004s having run nothing.
func TestDocsCiteRealTestNames(t *testing.T) {
	root := repoRoot(t)
	defined := testFuncNames(t, root)

	checked := 0
	for _, path := range docFiles(t, root) {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading %s: %v", path, err)
		}
		rel, _ := filepath.Rel(root, path)
		for _, name := range citedTestNames(string(raw)) {
			checked++
			if !defined[name] {
				t.Errorf("%s cites %s, which no _test.go file defines: the command it appears in "+
					"exits 0 having run nothing", rel, name)
			}
		}
	}
	if checked < 10 {
		t.Fatalf("only %d test names were checked, so this guard is no longer looking at the "+
			"documents it was written for", checked)
	}
}

// citedTestNames returns the Test* identifiers a document mentions.
//
// The pattern is deliberately loose about what surrounds the name -- backticks, a -run flag, or
// plain prose -- because all three forms appear and a name that rots is broken in all of them.
// TestMain and testing are excluded: both appear in prose about the framework rather than as
// references to a specific test.
func citedTestNames(text string) []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range testNamePattern.FindAllString(text, -1) {
		if m == "TestMain" || m == "Testing" || seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	return out
}

// testNamePattern matches an exported-looking test name.
//
// Two or more characters after Test, so the word "Test" alone does not match and "Tests" as an
// English plural does not become a name. Real Go test names are CamelCase beyond that.
var testNamePattern = regexp.MustCompile(`\bTest[A-Z]\w+`)

// docFiles lists the Markdown and doc.go files this repository cites tests from.
func docFiles(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	matches, err := filepath.Glob(filepath.Join(root, "docs", "*.md"))
	if err != nil {
		t.Fatalf("globbing docs: %v", err)
	}
	out = append(out, matches...)
	internalDocs, err := filepath.Glob(filepath.Join(root, "internal", "*", "doc.go"))
	if err != nil {
		t.Fatalf("globbing internal doc.go: %v", err)
	}
	out = append(out, internalDocs...)
	if len(out) < 10 {
		t.Fatalf("found %d documents, want at least 10", len(out))
	}
	return out
}

// testFuncNames collects every Test* function the repository defines.
func testFuncNames(t *testing.T, root string) map[string]bool {
	t.Helper()
	out := map[string]bool{}
	files, err := filepath.Glob(filepath.Join(root, "internal", "*", "*_test.go"))
	if err != nil {
		t.Fatalf("globbing tests: %v", err)
	}
	cmdTests, err := filepath.Glob(filepath.Join(root, "cmd", "*", "*_test.go"))
	if err != nil {
		t.Fatalf("globbing cmd tests: %v", err)
	}
	for _, path := range append(files, cmdTests...) {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading %s: %v", path, err)
		}
		for _, m := range testFuncPattern.FindAllStringSubmatch(string(raw), -1) {
			out[m[1]] = true
		}
	}
	if len(out) < 400 {
		t.Fatalf("found %d test functions, which is too few to be the whole repository", len(out))
	}
	return out
}

// testFuncPattern matches a test function's declaration.
var testFuncPattern = regexp.MustCompile(`(?m)^func (Test\w+)\(`)

// repoRoot walks up from the test's working directory to the module root.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	for range 6 {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		dir = filepath.Dir(dir)
	}
	t.Fatal("no go.mod found above the test's directory")
	return ""
}
