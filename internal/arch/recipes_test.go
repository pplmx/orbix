package arch

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// TestEveryRecipeIsDocumented checks that a recipe added to the justfile is named in a document a
// contributor or an agent would read.
//
// The two lists drifted: four recipes were added over this session and only AGENTS.md gained all of
// them, so CONTRIBUTING.md listed fourteen of eighteen. Nothing noticed, because a missing line in a
// list is not a broken link and rumdl has nothing to say about it. This is the same shape as
// testnames_test.go and docsymbols_test.go -- a fact written in two places, with a guard on the
// places rather than on a habit of remembering.
//
// `just --list` is the source of truth, so a recipe renamed in the justfile fails here until the
// documents follow. The default recipe is exempt: it is invoked as `just`, which no list spells out.
func TestEveryRecipeIsDocumented(t *testing.T) {
	root := "../.."

	listed, err := exec.Command("just", "--list").Output()
	if err != nil {
		// A failure rather than a skip. The guard exists to catch a recipe that no document
		// names, and skipping it when `just` is missing means the CI job that does not install
		// just -- test-darwin -- runs this and reports green while checking nothing. Measured:
		// that job installs Go and nothing else, so every recipe in this file was unverified on
		// macOS.
		//
		// The cost is that a contributor without just cannot run the suite. That is already
		// true of every other gate here -- the justfile is the entry point for all of them --
		// so this adds a message where there was a silent pass, not a new prerequisite.
		t.Fatalf("just is not available: %v\nrun `cargo install just` or use your package "+
			"manager; every gate in this repository is a just recipe", err)
	}
	recipes := parseRecipes(string(listed))
	if len(recipes) == 0 {
		t.Fatal("parsed no recipes from just --list; the output format changed")
	}

	// Each list has to be complete on its own. Checking the union passed with a line removed from
	// CONTRIBUTING.md because AGENTS.md still named it -- and that is the drift this is here to
	// catch, not a hypothetical. Both files list every recipe one per line, so the same rule holds
	// for both.
	for _, doc := range []string{"CONTRIBUTING.md", "AGENTS.md"} {
		body, err := os.ReadFile(filepath.Join(root, doc))
		if err != nil {
			t.Fatalf("read %s: %v", doc, err)
		}
		documented := map[string]bool{}
		for _, m := range justMention.FindAllStringSubmatch(string(body), -1) {
			documented[m[1]] = true
		}

		var missing []string
		for _, r := range recipes {
			if r == "default" || documented[r] {
				continue
			}
			missing = append(missing, r)
		}
		slices.Sort(missing)
		for _, r := range missing {
			t.Errorf("recipe %q is in the justfile but not named in %s;"+
				" add a line for it where the other recipes are listed", r, doc)
		}
	}
}

// justMention matches a `just <recipe>` reference in prose or a code block. Only the first word
// after `just` is captured, so `just RANGE=base..HEAD commits` is recorded as "RANGE=base..HEAD"
// and the recipe it names is picked up by the plain mention the same document already has.
var justMention = regexp.MustCompile("`?just ([a-z][a-z-]*)")

// parseRecipes pulls the recipe names out of `just --list`, whose lines look like
// "    build           # compile to bin/orbix".
func parseRecipes(out string) []string {
	var names []string
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || !strings.HasPrefix(line, "    ") {
			continue
		}
		name := fields[0]
		if name == "" || strings.ContainsAny(name, "#") {
			continue
		}
		names = append(names, name)
	}
	return names
}

// TestArchitectureReviewCIIobCountMatchesTheWorkflow checks the job count AGENTS.md states
// against the workflow it describes.
//
// This is the third number in this repository to drift in the same way, and it drifted from
// two directions in one session: nine jobs became ten when the jinja-diff job was added, and
// became eleven when the four xcheck recipes were folded into one job. Nobody edits a sentence
// like "九个 job 各自验证一件事" by hand and gets it right; the sentence has to be rewritten
// wholesale, and that is when the count is most likely to be copied from the previous version.
//
// Only the count is checked. Which job checks what is a sentence a person writes, and it has
// no mechanical form -- but a count that disagrees is proof the sentence is stale, which is
// enough to make someone read it.
func TestArchitectureReviewCIIobCountMatchesTheWorkflow(t *testing.T) {
	workflow, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Fatalf("read ci.yml: %v", err)
	}
	jobs := countCIJobs(string(workflow))
	if jobs < 5 {
		t.Fatalf("counted %d jobs in ci.yml, too few to be the real matrix -- the parser probably "+
			"stopped matching", jobs)
	}

	agents, err := os.ReadFile(filepath.Join("..", "..", "AGENTS.md"))
	if err != nil {
		t.Fatalf("read AGENTS.md: %v", err)
	}
	claim := ciJobCount.FindStringSubmatch(string(agents))
	if claim == nil {
		t.Fatalf("AGENTS.md no longer states the job count as \"N 个 job 各自验证一件事\"; " +
			"restore that phrasing or update this pattern")
	}
	if want := chineseNumeral(claim[1]); want != jobs {
		t.Errorf("AGENTS.md says %s jobs; ci.yml defines %d", claim[1], jobs)
	}
}

// countCIJobs counts the entries under the top-level `jobs:` key.
//
// A line-based scan rather than a YAML dependency: this repository has none in go.mod, and the
// shape being read is two spaces of indentation under a single top-level key. A job is a line
// matching `^  <name>:$` after `jobs:` appears.
func countCIJobs(src string) int {
	inJobs := false
	n := 0
	for line := range strings.SplitSeq(src, "\n") {
		if line == "jobs:" {
			inJobs = true
			continue
		}
		if !inJobs {
			continue
		}
		// Any other column-zero key ends the block.
		if line != "" && !strings.HasPrefix(line, " ") && !strings.HasPrefix(line, "#") {
			break
		}
		if ciJobLine.MatchString(line) {
			n++
		}
	}
	return n
}

var (
	ciJobLine  = regexp.MustCompile(`^  [a-z][a-z0-9-]*:$`)
	ciJobCount = regexp.MustCompile(`([一二三四五六七八九十]+|\d+) ?个 job 各自验证一件事`)
)

// chineseNumeral converts the counts this sentence plausibly uses. The map stops where the
// matrix would have to be written in digits, which is the form a contributor copies from a tool
// rather than from prose.
func chineseNumeral(s string) int {
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}
	return chineseNumerals[s]
}

// TestAgentDocCLIScaleMatchesTheTree checks the CLI scale sentence in AGENTS.md.
//
// The sentence read "6 个命令、约 3400 行非测试代码的 CLI"，and the same sentence's parenthetical
// went on to say 3607 lines -- a document contradicting itself within one paragraph, and both
// halves stale: the dispatch has eight commands, and the tree is 3884 lines. Neither number had
// a guard, so each could drift again as soon as something was added.
//
// The counts come from the tree rather than from a copy here, for the reason the other guards in
// this file record: a copy agrees with itself forever.
func TestAgentDocCLIScaleMatchesTheTree(t *testing.T) {
	doc, err := os.ReadFile(filepath.Clean(filepath.Join("..", "..", "AGENTS.md")))
	if err != nil {
		t.Fatalf("read AGENTS.md: %v", err)
	}

	// Commands, from the dispatch in main.go: the first switch is the one that names them.
	mainSrc, err := os.ReadFile(filepath.Clean(filepath.Join("..", "..", "cmd", "orbix", "main.go")))
	if err != nil {
		t.Fatalf("read main.go: %v", err)
	}
	commands := dispatchCases(t, string(mainSrc))
	if len(commands) == 0 {
		t.Fatal("found no command cases in main.go; the dispatch moved or its shape changed")
	}

	// Lines, from the files the sentence's own parenthetical describes: non-test Go files under
	// cmd/orbix, excluding the build-tag file that CGO_ENABLED=0 does not compile.
	dir := filepath.Clean(filepath.Join("..", "..", "cmd", "orbix"))
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read cmd/orbix: %v", err)
	}
	lines, files := 0, 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		if name == "cgo_enabled.go" {
			// Not built when CGO_ENABLED=0, which is the only supported build.
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		lines += strings.Count(string(raw), "\n")
		files++
	}

	// The sentence under test, and the two numbers it states.
	m := cliScaleSentence.FindStringSubmatch(string(doc))
	if m == nil {
		t.Fatalf("AGENTS.md no longer states the CLI scale as \"**N** 个命令（…）、**M** 行非测试代码\"; " +
			"restore that shape or update this pattern")
	}
	if got, want := archAtoi(t, m[1]), len(commands); got != want {
		t.Errorf("AGENTS.md says %d commands; the dispatch in main.go has %d (%v)",
			got, want, commands)
	}
	if got, want := archAtoi(t, m[2]), lines; got != want {
		t.Errorf("AGENTS.md says %d lines of non-test CLI code across %d files; measured %d",
			got, files, want)
	}
}

// archAtoi parses a number this file read out of a document.
func archAtoi(t *testing.T, s string) int {
	t.Helper()
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("%q is not a number: %v", s, err)
	}
	return n
}

// cliScaleSentence captures the command count and the line count from the AGENTS.md sentence.
var cliScaleSentence = regexp.MustCompile(`\*\*(\d+)\*\* 个命令（[^）]*）、\*\*(\d+)\*\* 行非测试代码`)

// dispatchCases returns the command names from the first switch in main.go.
//
// It reads the first switch rather than every switch in the file: helpFor has a second one that
// names a subset of the same commands, and counting that would double them.
func dispatchCases(t *testing.T, src string) []string {
	t.Helper()
	start := strings.Index(src, "switch")
	if start < 0 {
		t.Fatalf("main.go has no switch")
	}
	rest := src[start:]
	end := strings.Index(rest, "\n\t}\n")
	if end < 0 {
		t.Fatalf("the first switch in main.go has no close this test can find")
	}
	body := rest[:end]

	var names []string
	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "case ") {
			continue
		}
		for _, part := range strings.Split(strings.TrimPrefix(trimmed, "case "), ",") {
			part = strings.TrimSpace(part)
			// Aliases like "-v" and "--version" are spellings of a command, not new commands;
			// the canonical name is the one without a dash.
			if !strings.HasPrefix(part, `"`) {
				continue
			}
			// A case line holds several comma-separated quoted names: split on the quote
			// leaves a trailing fragment on all but the last, so cut between the quotes.
			if q := strings.Index(part[1:], `"`); q >= 0 {
				part = part[:q+2]
			}
			name := strings.Trim(part, `"`)
			if strings.HasPrefix(name, "-") {
				continue
			}
			names = append(names, name)
		}
	}
	return names
}
