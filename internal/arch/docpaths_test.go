package arch

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// TestDocsCiteRealFiles checks that every repository path a document names exists.
//
// The other guards cover symbols, test names, recipes and the layer table -- things written as
// identifiers. A path in prose is the same kind of claim and had no guard: a file moved in a
// refactor leaves the sentence pointing at nothing, and Markdown renders straight through it.
//
// Only paths are checked, not line numbers or anchors, because a path is what a reader follows.
// Scoped to internal/, cmd/ and scripts/: a bare filename appears in prose often enough that
// matching it would produce mostly noise.
func TestDocsCiteRealFiles(t *testing.T) {
	root := filepath.Join("..", "..")

	docs := []string{
		filepath.Join(root, "README.md"),
		filepath.Join(root, "AGENTS.md"),
		filepath.Join(root, "CONTRIBUTING.md"),
	}
	extra, err := filepath.Glob(filepath.Join(root, "docs", "*.md"))
	if err != nil {
		t.Fatalf("glob docs: %v", err)
	}
	docs = append(docs, extra...)
	// The packages' own doc.go files cite paths too -- engine/doc.go points at
	// internal/server/imports_test.go as the thing holding an import edge shut -- and they were
	// not in this list. Measured before adding them: rewriting that path to a file that does not
	// exist left this test green, so the one citation a reader is most likely to follow was the
	// one nothing checked.
	//
	// Only paths with a directory prefix are matched, so the bare filenames these files also
	// mention (rope.go, tools.go) do not become noise here.
	for _, pattern := range []string{
		filepath.Join(root, "internal", "*", "doc.go"),
		filepath.Join(root, "cmd", "*", "doc.go"),
		// And the rest of the source, for the same reason one step further out: comments in
		// ordinary Go files cite paths too. Measured before adding them, ten such citations
		// existed -- internal/llama/config.go, internal/chat/eval.go, cmd/orbix/run.go,
		// scripts/count.go and six more -- and every one of them was correct by luck. A comment
		// naming a file that moved is the same broken pointer whether it sits in doc.go or in a
		// function, and the second kind is far more common.
		filepath.Join(root, "internal", "*", "*.go"),
		filepath.Join(root, "cmd", "*", "*.go"),
		filepath.Join(root, "scripts", "*.go"),
		// And one level down, because a command that becomes its own package moves there.
		// scripts/genfixture is the case: its package comment documented a `go run` of the
		// file itself with a .go suffix, which is not how a package directory is run, and
		// the pattern above matched only scripts/count.go -- so the broken command sat in
		// the tree unread by anything. Measured before adding this: rewriting that comment
		// to name a path that does not exist left this test green.
		filepath.Join(root, "scripts", "*", "*.go"),
	} {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			t.Fatalf("glob %s: %v", pattern, err)
		}
		docs = append(docs, matches...)
	}

	cited := map[string]bool{}
	for _, doc := range docs {
		body, err := os.ReadFile(doc)
		if err != nil {
			t.Fatalf("read %s: %v", doc, err)
		}
		for _, m := range repoPath.FindAllString(string(body), -1) {
			cited[m] = true
		}
	}
	if len(cited) < 10 {
		t.Fatalf("found %d cited paths, which is too few to be checking anything", len(cited))
	}

	for path := range cited {
		if _, err := os.Stat(filepath.Join(root, path)); err != nil {
			t.Errorf("a document names %s and it does not exist", path)
		}
	}
}

// repoPath matches a path under one of the directories this repository owns. The trailing
// extension is required so a directory reference does not match a file that does not exist yet.
var repoPath = regexp.MustCompile(`\b(?:internal|cmd|scripts|docs)/[a-zA-Z0-9_/.]+\.(?:go|py|sh|md)\b`)

// TestDocLinksResolve checks that a relative Markdown link points at a file.
//
// Relative to the document that contains it, not to the repository root: docs/usage.md links to
// models.md meaning docs/models.md, and resolving from the root reports that as missing. Checking
// it the wrong way is how a first attempt at this found a broken link that was not broken.
func TestDocLinksResolve(t *testing.T) {
	root := filepath.Join("..", "..")

	docs := []string{
		filepath.Join(root, "README.md"),
		filepath.Join(root, "AGENTS.md"),
		filepath.Join(root, "CONTRIBUTING.md"),
	}
	extra, err := filepath.Glob(filepath.Join(root, "docs", "*.md"))
	if err != nil {
		t.Fatalf("glob docs: %v", err)
	}
	docs = append(docs, extra...)

	checked := 0
	for _, doc := range docs {
		body, err := os.ReadFile(doc)
		if err != nil {
			t.Fatalf("read %s: %v", doc, err)
		}
		for _, m := range markdownLink.FindAllStringSubmatch(string(body), -1) {
			target := filepath.Join(filepath.Dir(doc), m[1])
			if _, err := os.Stat(target); err != nil {
				t.Errorf("%s links to %s, which does not exist", doc, m[1])
			}
			checked++
		}
	}
	if checked < 5 {
		t.Fatalf("found %d relative links, which is too few to be checking anything", checked)
	}
}

// markdownLink captures the target of a relative .md link. An absolute URL is skipped: this
// checks the repository, not the network.
var markdownLink = regexp.MustCompile(`\]\(([a-zA-Z0-9_./-]+\.md)\)`)
