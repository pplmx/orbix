package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"testing"
)

// documentedCommand is one command whose flags a document claims to list.
type documentedCommand struct {
	name string
	doc  string
}

// flagDocuments are the documents that claim to enumerate flags.
//
// docs/configuration.md is the authority: it opens with "Every setting is a flag today" and then
// tables them. docs/cli.md is not in this list on purpose -- it explains behaviour, names the flags
// that matter to the explanation, and deliberately does not repeat the table. Checking it would
// demand flags in a document whose job is not to list them.
var flagDocuments = []documentedCommand{
	{"run", "../../docs/configuration.md"},
	{"bench", "../../docs/configuration.md"},
	{"chat", "../../docs/configuration.md"},
	{"serve", "../../docs/configuration.md"},
}

// flagsOf runs the command with -h and returns the flags it advertises.
//
// The binary is built by go run rather than linked in, because the flags are registered inside
// cmdServe and friends where they read as a block; lifting them out for testability would have
// meant a second refactor of four commands purely to serve this check. Measured, four invocations
// cost about 2.7 seconds together.
func flagsOf(t *testing.T, command string) []string {
	t.Helper()
	cmd := exec.Command("go", "run", "./", command, "-h")
	out, _ := cmd.CombinedOutput()
	// A -h exit status is non-zero on some paths, so the output is what matters, not the error.
	matches := realFlag.FindAllStringSubmatch(string(out), -1)
	if len(matches) == 0 {
		t.Fatalf("%s -h produced no flags: %s", command, out)
	}
	seen := make(map[string]bool)
	var names []string
	for _, m := range matches {
		if !seen[m[1]] {
			seen[m[1]] = true
			names = append(names, m[1])
		}
	}
	sort.Strings(names)
	return names
}

// realFlag matches a flag line in Go's flag package help output: two spaces, one dash, the name.
var realFlag = regexp.MustCompile(`(?m)^  -([a-z][a-z-]*)`)

// TestConfigurationDocListsEveryFlag is the check that a command's flags and the document agree.
//
// The document opens by claiming to list every setting, and nothing enforced that. A flag added
// without a table row would leave it describing a smaller surface than the binary has; a flag
// renamed would leave the document naming something nobody can pass. Neither breaks a build, and
// neither is noticeable unless someone re-reads the table against -h -- which is what this does.
func TestConfigurationDocListsEveryFlag(t *testing.T) {
	accepts := make(map[string][]string)
	docs := make(map[string]string)
	for _, c := range flagDocuments {
		raw, err := readFile(c.doc)
		if err != nil {
			t.Fatalf("reading %s: %v", c.doc, err)
		}
		docs[c.doc] = raw
	}

	for _, c := range flagDocuments {
		t.Run(c.name, func(t *testing.T) {
			text := docs[c.doc]
			// Only flags the document names in backticks with a double dash count as documented;
			// a bare mention inside prose about another command is not a claim about this one.
			documented := make(map[string]bool)
			for _, m := range docFlag.FindAllStringSubmatch(text, -1) {
				documented[m[1]] = true
			}

			real := flagsOf(t, c.name)
			for _, flag := range real {
				if !documented[flag] {
					t.Errorf("%s accepts --%s, which %s does not list", c.name, flag, c.doc)
				}
			}
			accepts[c.name] = real
		})
	}
}

// docFlag matches a flag in backticks: `--name`.
var docFlag = regexp.MustCompile("`--([a-z][a-z-]*)`")

// readFile is os.ReadFile with the test's naming, so the failure says which document.
func readFile(path string) (string, error) {
	b, err := os.ReadFile(filepath.Clean(path))
	return string(b), err
}

// TestConfigurationDocNamesOnlyRealFlags is the other direction, without attributing a flag to a
// command.
//
// Attributing would need the document's sections parsed, and the sections are grouped by feature
// rather than by command -- "Paged cache, on serve, chat and bench" is one table covering three.
// An attempt at it reported eleven flags as belonging to the wrong command, all of them false.
//
// So this asks the weaker question that is still worth asking: is every flag the document names a
// flag some command accepts? That catches a rename on the document side and a flag deleted from
// the code, which are the two ways the table can promise something nobody can pass.
func TestConfigurationDocNamesOnlyRealFlags(t *testing.T) {
	// A document-only flag would otherwise be caught by nothing: the per-command test above only
	// fails when a real flag is missing from the document, not when the document invents one.
	union := make(map[string]bool)
	for _, c := range flagDocuments {
		for _, name := range flagsOf(t, c.name) {
			union[name] = true
		}
	}
	if len(union) < 25 { // measured: 26
		t.Fatalf("only %d distinct flags were found across four commands, so this check is "+
			"looking at the wrong thing", len(union))
	}

	raw, err := os.ReadFile(filepath.Clean(flagDocuments[0].doc))
	if err != nil {
		t.Fatalf("reading %s: %v", flagDocuments[0].doc, err)
	}
	for _, m := range docFlag.FindAllStringSubmatch(string(raw), -1) {
		if !union[m[1]] {
			t.Errorf("%s names --%s, which no command accepts", flagDocuments[0].doc, m[1])
		}
	}
}
