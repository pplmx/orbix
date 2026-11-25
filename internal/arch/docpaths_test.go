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
var repoPath = regexp.MustCompile(`\b(?:internal|cmd|scripts)/[a-zA-Z0-9_/.]+\.(?:go|py|sh)\b`)
