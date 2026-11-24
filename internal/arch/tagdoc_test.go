package arch

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestDocsExplainTheirVersionMarkers keeps the status columns from reading as history.
//
// docs/models.md and docs/cli.md mark every row with a release version, and the repository has no
// tags -- git tag is empty and there is no CHANGELOG. A reader who sees "v0.3" and finds no release
// concludes the project is further along than it is.
//
// The check is deliberately narrow: it does not judge which version a row should carry, because
// that is a plan and plans are not checkable here. It checks that the paragraph explaining the
// column is present, and it starts doing more once the first tag exists -- at that point the
// markers become claims about history and the skip below says so rather than the test pretending
// to have verified them.
func TestDocsExplainTheirVersionMarkers(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	if tags := gitTags(t, filepath.Join(root, "..", "..")); len(tags) > 0 {
		t.Skipf("%d tag(s) exist (%s), so this check's premise -- that the version markers are "+
			"plans -- no longer holds; the markers are now history and could be checked against "+
			"the tag list", len(tags), strings.Join(tags, ", "))
	}

	checked := 0
	for _, name := range []string{"models.md", "cli.md"} {
		path := filepath.Join(root, "..", "..", "docs", name)
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading %s: %v", name, err)
		}
		text := string(raw)
		if !versionMarker.MatchString(text) {
			t.Errorf("docs/%s no longer carries version markers; this check and the paragraph "+
				"it guards are both obsolete and should be removed together", name)
			continue
		}
		checked++
		// The sentence has to say no release has been made. Case and wording are free; the
		// claim is not.
		lower := strings.ToLower(text)
		if !strings.Contains(lower, "no release has been") {
			t.Errorf("docs/%s marks rows with release versions but never says that none has "+
				"been tagged, so the markers read as shipped features", name)
		}
	}
	if checked < 2 {
		t.Fatalf("only %d documents were checked, so the list above is out of date", checked)
	}
}

// versionMarker matches a v0.N marker.
var versionMarker = regexp.MustCompile(`\bv0\.(\d+)\b`)

// gitTags lists the repository's tags.
func gitTags(t *testing.T, dir string) []string {
	t.Helper()
	out, err := exec.Command("git", "-C", dir, "tag", "--list").Output()
	if err != nil {
		t.Fatalf("git tag: %v", err)
	}
	return strings.Fields(string(out))
}
