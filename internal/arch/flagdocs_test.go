package arch

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// TestConfigurationListsEveryFlag checks docs/configuration.md against the flags the commands
// actually define, in both directions.
//
// The document is the reference a user reads before running anything, so a flag it describes that
// no longer exists is worse than an undocumented one: the reader types it and gets a usage error.
// The reverse matters too -- a flag added for an operational reason and never written down is
// invisible to anyone who did not write it.
//
// `go run` rather than the built binary: a contributor who has not run `just build` should still
// pass this. It costs one compile per command.
func TestConfigurationListsEveryFlag(t *testing.T) {
	root := filepath.Join("..", "..")

	doc, err := os.ReadFile(filepath.Join(root, "docs", "configuration.md"))
	if err != nil {
		t.Fatalf("read configuration.md: %v", err)
	}
	documented := map[string]bool{}
	for _, m := range flagRow.FindAllStringSubmatch(string(doc), -1) {
		documented[strings.TrimPrefix(m[1], "--")] = true
	}
	if len(documented) < 20 {
		t.Fatalf("found %d flags in configuration.md, too few to be checking anything", len(documented))
	}

	// Every command that registers flags. A command missing here is a command whose flags are not
	// checked; the count assertion below notices if this list stops matching reality.
	var actual []string
	for _, cmd := range []string{"run", "chat", "serve", "bench"} {
		c := exec.Command("go", "run", "./cmd/orbix", cmd, "-h")
		// The test runs in internal/arch; the module path is relative to the repository root.
		c.Dir = root
		help, err := c.CombinedOutput()
		if err != nil {
			// -h exits non-zero on some commands; the usage text is still what we came for.
			if len(help) == 0 {
				t.Fatalf("%s -h produced nothing: %v", cmd, err)
			}
		}
		for _, m := range helpFlag.FindAllStringSubmatch(string(help), -1) {
			actual = append(actual, m[1])
		}
	}
	slices.Sort(actual)
	actual = slices.Compact(actual)
	if len(actual) < 20 {
		t.Fatalf("parsed %d flags from the commands, too few to be checking anything", len(actual))
	}

	for _, f := range actual {
		if !documented[f] {
			t.Errorf("the commands define --%s and docs/configuration.md does not list it", f)
		}
	}
	seen := map[string]bool{}
	for _, f := range actual {
		seen[f] = true
	}
	for f := range documented {
		if !seen[f] {
			t.Errorf("docs/configuration.md lists --%s and no command defines it", f)
		}
	}
}

// flagRow matches a table row whose first cell is a long flag in backticks.
var flagRow = regexp.MustCompile("(?m)^\\|\\s*`(--[a-z-]+)`")

// helpFlag matches a flag name in the standard flag package's usage output, which prints one per
// line as "  -name type".
var helpFlag = regexp.MustCompile(`(?m)^  -([a-z][a-z-]*)[\s\n]`)
