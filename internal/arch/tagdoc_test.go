package arch

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestDocsExplainTheirVersionMarkers keeps the status columns honest about what a marker means.
//
// docs/models.md and docs/cli.md mark every row with a release version. When this was written the
// repository had no tags, so the marker was a plan and the paragraph had to say so -- otherwise a
// reader who saw "v0.3" and went looking for a release would conclude the project was further
// along than it was.
//
// There are tags now (v0.1.0, v0.2.0, v0.3.0), which inverts the requirement rather than removing
// it: a marker at or below the newest tag names a release that exists, and a marker above it is
// still a plan. So the paragraphs have to distinguish the two. This test used to skip as soon as
// any tag existed, with a message saying the markers "could be checked against the tag list" --
// and that skip is why both documents went on saying "no release has been tagged yet" for two
// releases. A guard that disables itself when its premise changes is a guard that stops reporting
// exactly when it starts mattering.
//
// It checks three things per document: that markers exist at all (so the paragraph and this test
// are removed together when they are no longer wanted), that the paragraph names the tagged
// releases, and that every marker either matches a tag or is above the newest one. It still does
// not judge which version a row should carry.
func TestDocsExplainTheirVersionMarkers(t *testing.T) {
	root := repoRoot(t)
	allTags := gitTags(t, root)
	newest := newestTagMinor(t, allTags)

	checked := 0
	for _, name := range []string{"models.md", "cli.md"} {
		path := filepath.Join(root, "docs", name)
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

		// Every marker has to be a release that exists, or one past the newest. A marker in
		// between would name a version nobody can find.
		for _, m := range versionMarker.FindAllString(text, -1) {
			minor := markerMinor(t, m)
			if minor > newest {
				continue
			}
			if !tagExists(allTags, m) {
				t.Errorf("docs/%s marks a row %s, which is at or below the newest tag (v0.%d) "+
					"but has no tag of its own; either the marker is wrong or the release was "+
					"never made", name, m, newest)
			}
		}

		// The paragraph has to say which releases exist, because a reader needs to know
		// whether a marker is history or a plan.
		lower := strings.ToLower(text)
		if strings.Contains(lower, "no release has been") {
			t.Errorf("docs/%s says no release has been tagged, but %d tags exist "+
				"(v0.1.0 .. v0.%d.0): the markers now name releases that exist and the "+
				"paragraph has to say so", name, newest, newest)
		}
		if !strings.Contains(lower, "tagged") {
			t.Errorf("docs/%s marks rows with release versions without saying anything about "+
				"which of them have been tagged, so a reader cannot tell history from a plan",
				name)
		}
	}
	if checked < 2 {
		t.Fatalf("only %d documents were checked, so the list above is out of date", checked)
	}
}

// newestTagMinor returns the minor number of the highest v0.N.0 tag, and fails when there is none.
//
// The helper this replaced returned the tag list and the caller skipped on it being non-empty.
// Reading only the newest minor keeps the arithmetic in one place: a marker is history when its
// minor is at or below this number.
func newestTagMinor(t *testing.T, tags []string) int {
	t.Helper()
	if len(tags) == 0 {
		t.Fatalf("no tags exist; this test's premise is that some do. Restore the " +
			"\"no release has been tagged yet\" check or delete this test with its paragraph")
	}
	newest := 0
	for _, tag := range tags {
		if m := regexp.MustCompile(`^v0\.(\d+)\.(\d+)$`).FindStringSubmatch(tag); m != nil {
			newest = max(newest, archAtoi(t, m[1]))
		}
	}
	if newest == 0 {
		t.Fatalf("tags %v carry no v0.N.M version this test can read", tags)
	}
	return newest
}

// markerMinor reads the minor number out of a v0.N marker.
func markerMinor(t *testing.T, marker string) int {
	t.Helper()
	m := regexp.MustCompile(`^v0\.(\d+)$`).FindStringSubmatch(marker)
	if m == nil {
		t.Fatalf("%q is not a v0.N marker; versionMarker and this pattern disagree", marker)
	}
	return archAtoi(t, m[1])
}

// tagExists reports whether a v0.N marker has a matching v0.N.M tag.
//
// It takes the already-listed tags rather than running git again, so the caller's single `git tag`
// is the one this test's conclusions rest on.
func tagExists(tags []string, marker string) bool {
	for _, tag := range tags {
		if strings.HasPrefix(tag, marker+".") {
			return true
		}
	}
	return false
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
