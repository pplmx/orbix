package arch

import (
	"go/ast"
	"go/parser"
	"go/token"
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

// TestArchitectureReviewCIJobCountMatchesTheWorkflow checks the job count AGENTS.md states
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
func TestArchitectureReviewCIJobCountMatchesTheWorkflow(t *testing.T) {
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

// TestAgentDocQuantCrossCheckCountMatchesTheCorpus checks the value count in AGENTS.md's
// xcheck-quant recipe line.
//
// That figure is quoted in prose and nothing computed it, so it drifted: the line said 17770 while
// `just xcheck-quant` reports 14750. The corpus is built by qdump_test.go as two arithmetic blocks
// plus eight random ones per format, so the total is a function of the format table and the block
// count -- both in that file -- and can be recomputed here from the same numbers rather than by
// running the whole cross-check.
//
// It reads the format table's block sizes out of qdump_test.go instead of restating them, so
// adding a format or changing a block size fails here rather than leaving the prose behind. What it
// cannot see is the Python side: if scripts/quant_xcheck.py stopped comparing every value, this
// would still pass. The script now prints the total, which is the check for that half.
func TestAgentDocQuantCrossCheckCountMatchesTheCorpus(t *testing.T) {
	doc, err := os.ReadFile(filepath.Clean(filepath.Join("..", "..", "AGENTS.md")))
	if err != nil {
		t.Fatalf("read AGENTS.md: %v", err)
	}
	m := quantCrossCheckCount.FindStringSubmatch(string(doc))
	if m == nil {
		t.Fatalf("AGENTS.md no longer states the xcheck-quant corpus as \"14 种格式、N 个值\"; " +
			"restore that phrasing or update this pattern")
	}

	src, err := os.ReadFile(filepath.Clean(filepath.Join("..", "quant", "qdump_test.go")))
	if err != nil {
		t.Fatalf("read qdump_test.go: %v", err)
	}
	// Each row is {"NAME", gguf.TYPE, blockSize, bytesPerBlock}; the value count contributed is
	// blockSize per block, and every format gets the same number of blocks.
	rows := quantFormatRow.FindAllStringSubmatch(string(src), -1)
	if len(rows) < 14 {
		t.Fatalf("parsed %d format rows from qdump_test.go, too few; the table moved or its "+
			"shape changed", len(rows))
	}
	const blocksPerFormat = 2 + 8 // the arithmetic run plus the eight seeded blocks
	total := 0
	for _, r := range rows {
		total += archAtoi(t, r[1]) * blocksPerFormat
	}
	if got := archAtoi(t, m[1]); got != total {
		t.Errorf("AGENTS.md says the quant cross-check compares %d values; %d formats x %d "+
			"blocks gives %d -- run `just xcheck-quant` and use the total it prints",
			got, len(rows), blocksPerFormat, total)
	}
}

// quantCrossCheckCount captures the value count from the xcheck-quant recipe line.
var quantCrossCheckCount = regexp.MustCompile(`14 种格式、(\d+) 个值`)

// quantFormatRow captures one format's block size from qdump_test.go's table.
var quantFormatRow = regexp.MustCompile(`\{"[A-Z0-9_]+", gguf\.[A-Z0-9_]+, (\d+), \d+\}`)

// TestDocumentedRecipeArgumentsMatchTheRecipe checks the parameter names the documents show
// against the ones `just --list` reports.
//
// TestEveryRecipeIsDocumented only asks whether a recipe is named somewhere, and justMention
// captures the first word after `just` -- so `just mutate FILE PKG` counted as naming mutate
// while the recipe takes positional `file pkg`. Measured: `just mutate FILE=x PKG=y` passes
// "FILE=x" as the file name and fails with "no such file: FILE=x", so a reader following
// CONTRIBUTING.md got an error that looks like a missing file rather than a wrong invocation.
//
// Only the argument names are compared, not their values: `just RANGE=base..HEAD commits` uses
// the variable-assignment form just accepts for that recipe, and checking values would need
// running the recipe.
func TestDocumentedRecipeArgumentsMatchTheRecipe(t *testing.T) {
	listed, err := exec.Command("just", "--list").Output()
	if err != nil {
		t.Fatalf("just is not available: %v", err)
	}
	params := parseRecipeParams(string(listed))
	if len(params) == 0 {
		t.Fatal("parsed no recipes with parameters from just --list; the output format changed")
	}

	for _, doc := range []string{"CONTRIBUTING.md", "AGENTS.md"} {
		body, err := os.ReadFile(filepath.Join("..", "..", doc))
		if err != nil {
			t.Fatalf("read %s: %v", doc, err)
		}
		for _, m := range justArgsMention.FindAllStringSubmatch(string(body), -1) {
			recipe, shown := m[1], strings.Fields(m[2])
			want, ok := params[recipe]
			if !ok {
				// No parameters to compare, or a recipe this parser did not see; the other guard
				// owns whether the recipe exists at all.
				continue
			}
			for i, arg := range shown {
				if i >= len(want) {
					t.Errorf("%s shows `just %s` with %d argument(s); the recipe takes %d (%v)",
						doc, recipe, len(shown), len(want), want)
					break
				}
				// `NAME=value` is the variable form just accepts; a bare word has to match the
				// parameter name exactly, because just takes parameters positionally.
				if strings.Contains(arg, "=") {
					continue
				}
				if arg != want[i] {
					t.Errorf("%s shows `just %s %s`; parameter %d of that recipe is named %q, "+
						"not %q, and just matches them by position",
						doc, recipe, strings.Join(shown, " "), i+1, want[i], arg)
				}
			}
		}
	}
}

// justArgsMention captures a recipe name and the argument text after it, on the same line.
var justArgsMention = regexp.MustCompile("`?just ([a-z][a-z-]*)( [^`#\n]*)")

// parseRecipeParams reads the parameter names `just --list` prints after each recipe name.
//
// A variable is written as NAME='default' and a positional as name; only the NAME part is kept
// because that is what a caller types.
func parseRecipeParams(out string) map[string][]string {
	found := map[string][]string{}
	for _, line := range strings.Split(out, "\n") {
		if !strings.HasPrefix(line, "    ") {
			continue
		}
		text, _, _ := strings.Cut(strings.TrimSpace(line), " #")
		fields := strings.Fields(text)
		if len(fields) < 2 {
			continue
		}
		name := fields[0]
		if strings.ContainsAny(name, "#=") {
			continue
		}
		var ps []string
		for _, f := range fields[1:] {
			if p, _, ok := strings.Cut(f, "="); ok {
				ps = append(ps, p)
			} else {
				ps = append(ps, f)
			}
		}
		found[name] = ps
	}
	return found
}

// TestFuzzRecipeFindsEveryFuzzTarget guards the extraction `just fuzz` depends on.
//
// The recipe scrapes target names out of *_test.go with a regexp and passes each to
// `go test -fuzz`. Its first version was `func Fuzz[A-Za-z0-9_]*` with no anchor, which also
// matched the phrase inside a comment in gatesdoc_test.go, producing a target named "Fuzz".
// `go test` then failed with "matched no fuzz target", `set -e` ended the loop, and six of the
// ten targets never ran while the recipe reported failure. Nothing caught it for two rounds
// because no local gate runs `just fuzz`.
//
// This test holds the two sides together: the pattern the recipe uses, and a parse of the same
// files. A pattern that starts matching prose again makes these two numbers differ.
func TestFuzzRecipeFindsEveryFuzzTarget(t *testing.T) {
	root := repoRoot(t)

	// The exact pattern from the recipe's `targets` assignment. Kept as a literal rather than
	// read out of the justfile so that changing the recipe without changing this fails loudly.
	recipePattern := regexp.MustCompile(`(?m)^func (Fuzz[A-Za-z0-9_]*)\(f \*testing\.F\)`)

	fromRecipe := map[string]bool{}
	for _, path := range fuzzTestFiles(t, root) {
		src, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading %s: %v", path, err)
		}
		for _, m := range recipePattern.FindAllStringSubmatch(string(src), -1) {
			fromRecipe[m[1]] = true
		}
	}

	fromAST := map[string]bool{}
	for _, path := range fuzzTestFiles(t, root) {
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			t.Fatalf("parsing %s: %v", path, err)
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil {
				continue
			}
			if name := fn.Name.Name; strings.HasPrefix(name, "Fuzz") {
				fromAST[name] = true
			}
		}
	}

	if len(fromAST) < 5 {
		t.Fatalf("found %d fuzz declarations, too few to be the real set -- the walk probably "+
			"missed directories", len(fromAST))
	}
	for name := range fromAST {
		if !fromRecipe[name] {
			t.Errorf("the recipe's pattern misses %s, which the tree declares: that target "+
				"would never be fuzzed", name)
		}
	}
	for name := range fromRecipe {
		if !fromAST[name] {
			t.Errorf("the recipe's pattern invents %s, which no declaration defines: "+
				"go test -fuzz would fail on it and set -e would stop the loop", name)
		}
	}
}

// fuzzTestFiles lists every _test.go file under internal/ and cmd/.
func fuzzTestFiles(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	for _, pattern := range []string{
		filepath.Join(root, "internal", "*", "*_test.go"),
		filepath.Join(root, "cmd", "*", "*_test.go"),
	} {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			t.Fatalf("globbing %s: %v", pattern, err)
		}
		out = append(out, matches...)
	}
	if len(out) == 0 {
		t.Fatal("found no test files")
	}
	return out
}
