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

// TestCLIDocOnlyCitesFlagsThatExist checks the flag names docs/cli.md mentions against the ones
// the commands register.
//
// TestConfigurationListsEveryFlag covers the other direction for one document: every flag a
// command defines appears in configuration.md, and every flag that document lists is defined. It
// reads no other file. docs/cli.md cites thirteen flags in prose -- "`--blocks`, `--suite`" and so
// on -- and nothing checked those, so renaming a flag would leave the CLI document telling a
// reader to pass an option the command rejects.
//
// Only the names are compared, not the descriptions: what a flag does is prose, and a check that
// graded it would be guessing. What is mechanical is whether the name exists.
//
// This is the same gap TestDocsCiteRealTestNames had for AGENTS.md, found the same way: by asking
// which documents each guard actually reads rather than which claims it sounds like it covers.
func TestCLIDocOnlyCitesFlagsThatExist(t *testing.T) {
	root := filepath.Join("..", "..")

	// The same source of truth the other guard uses: what the commands print.
	// Every command, not just the four that register flags through the flag package. `version`
	// and `model` write their own usage line -- "Usage: orbix version [-json]" -- and the flag
	// package's one-per-line shape does not appear in them, so a list of four missed `--json`
	// entirely. That was found by this test failing on the flag it was written for.
	defined := map[string]bool{}
	for _, cmd := range []string{"run", "chat", "serve", "bench", "version", "model", "healthcheck"} {
		c := exec.Command("go", "run", "./cmd/orbix", cmd, "-h")
		c.Dir = root
		help, err := c.CombinedOutput()
		if err != nil && len(help) == 0 {
			t.Fatalf("%s -h produced nothing: %v", cmd, err)
		}
		for _, m := range helpFlag.FindAllStringSubmatch(string(help), -1) {
			defined[m[1]] = true
		}
		// The bracketed form the hand-written usage lines use. It writes one dash where a table
		// writes two, so the name is normalised the same way both times.
		for _, m := range usageFlag.FindAllStringSubmatch(string(help), -1) {
			defined[strings.TrimLeft(m[1], "-")] = true
		}
	}
	if len(defined) < 20 {
		t.Fatalf("parsed %d flags from the commands, too few to be checking anything", len(defined))
	}

	doc, err := os.ReadFile(filepath.Join(root, "docs", "cli.md"))
	if err != nil {
		t.Fatalf("read cli.md: %v", err)
	}
	// `--name` in backticks is how this document names an option; a bare --name in prose is not
	// matched, so prose about "the --flag convention" cannot fail this.
	cited := map[string]bool{}
	for _, m := range cliDocFlag.FindAllStringSubmatch(string(doc), -1) {
		// The pattern keeps the leading dashes because that is how the document writes it; the
		// set from the help output drops them, which is what the flag package prints.
		cited[strings.TrimPrefix(m[1], "--")] = true
	}
	if len(cited) < 8 {
		t.Fatalf("found %d flags in cli.md, too few to be checking anything -- the citation "+
			"style probably changed", len(cited))
	}

	names := slices.Sorted(maps.Keys(cited))
	for _, f := range names {
		if !defined[f] {
			t.Errorf("docs/cli.md cites --%s, which no command registers", f)
		}
	}
}

// cliDocFlag matches a flag name written in backticks in docs/cli.md.
var cliDocFlag = regexp.MustCompile("`(--[a-z][a-z-]+)`")

// usageFlag matches a flag inside a hand-written usage line, as in "orbix version [-json]".
var usageFlag = regexp.MustCompile(`\[(-{1,2}[a-z][a-z-]*)`)
