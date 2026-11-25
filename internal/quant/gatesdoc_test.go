package quant

import (
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// TestModelsDocGateTableMatchesTheCode keeps the checkpoint-gate table honest.
//
// docs/models.md lists fourteen environment variables a reader must set to run every
// checkpoint-gated test. That table drifted once already: it said 25 skipped with no variables
// when the tree skipped 31, and it described the split as "nine at the call site plus four table
// rows" when ten were at the call site. Nobody notices either kind of error by eye -- the
// document reads fine either way -- so it is checked here instead.
//
// The check runs the other way too. A gate added to the table but never read by a test promises
// a checkpoint that would check nothing, and a variable the tests read but the table omits leaves
// a reader unable to reach the documented zero.
func TestModelsDocGateTableMatchesTheCode(t *testing.T) {
	doc := readDoc(t, "models.md")
	documented, documentedCount := documentedGates(t, doc)

	// The count the prose states, which is a second place to get it wrong.
	if want := "There are fourteen gates"; !strings.Contains(doc, want) {
		t.Errorf("docs/models.md no longer says %q; update this test with the new sentence", want)
	}

	// Every env var the gate tests actually read, from the tree that owns them.
	used := modelGateEnvVars(t)

	for name := range documented {
		if !used[name] {
			t.Errorf("docs/models.md lists %s as a gate, but no test reads it -- a checkpoint for "+
				"it would check nothing", name)
		}
	}
	for name := range used {
		if !documented[name] {
			t.Errorf("tests read %s, which docs/models.md does not list as a gate", name)
		}
	}
	if len(documented) != documentedCount {
		t.Errorf("parsed %d gates from the table but the map holds %d", documentedCount, len(documented))
	}
}

// documentedGates reads the env-var names out of the gate table, and returns how many rows it saw.
//
// The table is located by its own header rather than by "the first table in the file": models.md
// holds several tables, and starting at the first one read zero gate rows. The header row is the
// one whose first column is literally `Variable`, which is specific to this table.
//
// Counting rows separately from the map is what catches a duplicate name, which would otherwise
// silently shrink the set.
func documentedGates(t *testing.T, doc string) (map[string]bool, int) {
	t.Helper()
	out := make(map[string]bool)
	rows := 0
	seenHeader := false
	for line := range strings.SplitSeq(doc, "\n") {
		if !strings.HasPrefix(line, "|") {
			if seenHeader && rows > 0 {
				break // the table ended
			}
			continue
		}
		if !seenHeader {
			if strings.Contains(line, "Variable") {
				seenHeader = true
			}
			continue
		}
		// The separator row carries no name.
		if strings.Contains(line, "---") {
			continue
		}
		cell, _, ok := strings.Cut(strings.TrimPrefix(line, "|"), "|")
		if !ok {
			continue
		}
		name := strings.Trim(strings.TrimSpace(cell), "`")
		if !isEnvVarName(name) {
			continue
		}
		rows++
		out[name] = true
	}
	if rows < 10 {
		t.Fatalf("read %d gate rows from docs/models.md, too few to be the real table -- the "+
			"parser probably stopped matching", rows)
	}
	return out, rows
}

// isEnvVarName reports whether s looks like an environment variable name: upper case, digits and
// underscores, and not all digits. That is what separates a gate row from the header.
func isEnvVarName(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if (r < 'A' || r > 'Z') && (r < '0' || r > '9') && r != '_' {
			return false
		}
	}
	return strings.ContainsAny(s, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
}

// modelGateEnvVars returns every checkpoint-gate variable the tree's tests read.
//
// It walks the whole module rather than this package, because the gates are spread across
// llama, tokenize, quant and server: a per-package walk would see the four table rows in
// formats_real_test.go and none of the rest, and would report thirteen missing from the document
// that in fact documents them.
//
// ORBIX_* variables are excluded deliberately. ROPEDUMP, PREDUMP, QUANTDUMP and REFERENCE_DIR
// are switches that make cross-implementation scripts dump Go-side output; no checkpoint is
// needed for them, and the document says so where it explains the distinction.
func modelGateEnvVars(t *testing.T) map[string]bool {
	t.Helper()
	root := filepath.Join("..", "..")
	out := make(map[string]bool)
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// .git holds no Go tests, and walking it on a large checkout is slow.
			if d.Name() == ".git" {
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, "_test.go") {
			return nil
		}
		b, rerr := os.ReadFile(filepath.Clean(path))
		if rerr != nil {
			return rerr
		}
		// Comments are stripped before matching. formats_real_test.go explains that it has "a
		// variable per format rather than one MODEL_32WEIGHT", and a name in that sentence is
		// not a gate -- matching it made this check demand a row for a variable the file says
		// does not exist.
		src := stripGoComments(string(b))
		for _, m := range getenvCall.FindAllStringSubmatch(src, -1) {
			out[m[1]] = true
		}
		for _, m := range gateLiteral.FindAllStringSubmatch(src, -1) {
			out[m[1]] = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking %s: %v", root, err)
	}
	if len(out) < 10 {
		t.Fatalf("found %d gate variables across the module (%v), too few to be real",
			len(out), slices.Sorted(maps.Keys(out)))
	}
	return out
}

// getenvCall matches os.Getenv("NAME") with a literal name.
var getenvCall = regexp.MustCompile(`Getenv\("([A-Z][A-Z0-9_]*_MODEL|MODEL[A-Z0-9_]*)"\)`)

// gateLiteral matches a gate name written as a bare string literal, which is how the four
// quant-format gates appear: formats_real_test.go holds them in a table of
// {type, envVar, label} rows and reads each through os.Getenv(g.env), so the call site names a
// variable rather than a literal. Matching the literals as well as the calls is what lets this
// check see those four -- the same blindness the document describes when it says a search for
// Getenv( finds ten gates and those four are invisible.
var gateLiteral = regexp.MustCompile(`"(MODEL_[A-Z0-9_]+)"`)

// stripGoComments removes // and /* */ comments, leaving string literals intact.
//
// This is not a Go parser and does not need to be: the input is a test file this repository
// wrote, and the only thing being asked is "does this name appear in code". Comment markers
// inside a string literal would be rare enough to look wrong here rather than to be silently
// mis-handled, so the scan tracks string state and leaves quoted text alone.
func stripGoComments(src string) string {
	var b strings.Builder
	b.Grow(len(src))
	inString := false
	inRaw := false
	inLineComment := false
	inBlockComment := false
	rs := []rune(src)
	for i := 0; i < len(rs); i++ {
		c := rs[i]
		switch {
		case inLineComment:
			if c == '\n' {
				inLineComment = false
				b.WriteRune(c)
			}
		case inBlockComment:
			if c == '*' && i+1 < len(rs) && rs[i+1] == '/' {
				inBlockComment = false
				i++
			}
		case inString:
			b.WriteRune(c)
			if c == '\\' && i+1 < len(rs) {
				// Skip the escaped byte so an escaped quote does not end the string.
				i++
				b.WriteRune(rs[i])
			} else if c == '"' {
				inString = false
			}
		case inRaw:
			b.WriteRune(c)
			if c == '`' {
				inRaw = false
			}
		default:
			switch {
			case c == '/' && i+1 < len(rs) && rs[i+1] == '/':
				inLineComment = true
				i++
			case c == '/' && i+1 < len(rs) && rs[i+1] == '*':
				inBlockComment = true
				i++
			case c == '"':
				inString = true
				b.WriteRune(c)
			case c == '`':
				inRaw = true
				b.WriteRune(c)
			default:
				b.WriteRune(c)
			}
		}
	}
	return b.String()
}

func readDoc(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Clean(filepath.Join("..", "..", "docs", name)))
	if err != nil {
		t.Fatalf("reading docs/%s: %v", name, err)
	}
	return string(b)
}

// TestArchitectureReviewStatesTheSameGateCount keeps the second document honest about a
// number the first one is already checked on.
//
// docs/models.md's gate table has had a go-here since the count last drifted. The same fact
// is restated in docs/architecture-review.md, with no guard, and it drifted again in the
// other direction: that page said "11 个环境变量" and named eleven, four of them missing --
// MODEL_Q4_1, MODEL_Q5_0, MODEL_Q5_1 and MODEL_IQ4_NL, which live in formats_real_test.go's
// table and are read through os.Getenv(g.env), so a grep for Getenv never saw them.
//
// The lesson is in the document itself now, and this is the check that makes it stick: a
// fact written in two places needs a guard on both, or the unguarded copy is the one that
// rots. The review is the unguarded copy here.
func TestArchitectureReviewStatesTheSameGateCount(t *testing.T) {
	review, err := os.ReadFile(filepath.Clean(filepath.Join("..", "..", "docs", "architecture-review.md")))
	if err != nil {
		t.Fatalf("read architecture-review.md: %v", err)
	}
	gateCount := len(modelGateEnvVars(t))

	// The sentence's own wording, so a rewrite is reported as "update this test with the new
	// sentence" rather than as a wrong number.
	claim := regexp.MustCompile(`由 \*\*(\d+)\*\* 个环境变量控制`)
	m := claim.FindStringSubmatch(string(review))
	if m == nil {
		t.Fatalf("architecture-review.md no longer states the gate count as " +
			`"由 **N** 个环境变量控制"; restore that phrasing or update this pattern`)
	}
	if got := m[1]; got != strconv.Itoa(gateCount) {
		t.Errorf("architecture-review.md says %s environment variables gate the skipped tests; "+
			"the module reads %d", got, gateCount)
	}

	// The names as well as the number: a count that matches while a name is missing is how
	// the previous revision failed.
	sentence := regexp.MustCompile(`由 \*\*\d+\*\* 个环境变量控制[^。]*。`).FindString(string(review))
	if sentence == "" {
		t.Fatal("cannot locate the gate sentence to check its names")
	}
	for name := range modelGateEnvVars(t) {
		// ORBIX_* switches are excluded by modelGateEnvVars already; anything else it reports
		// is a checkpoint gate and belongs in the sentence.
		if !strings.Contains(sentence, name) {
			t.Errorf("architecture-review.md's gate sentence omits %s, which the tests read", name)
		}
	}
}

// TestArchitectureReviewStatesTheSkipCounts checks the other two figures in that sentence.
//
// "74 处 t.Skip 分布在 29 个测试文件里" is the same kind of claim as the gate count beside it,
// and it has drifted twice already by the document's own admission ("上一版写 73 和 28, 写下时
// 是 65 和 24"). A number that a person re-measures every release is a number that is wrong
// between releases; the sentence now sits in a test that counts both.
//
// The comparison uses the written skip calls rather than `just skips`' output. That recipe runs
// the whole suite, which is the slowest thing in CI and this check would then be measuring the
// suite rather than the tree. `t.Skip` in a `_test.go` file is the definition the document's own
// number comes from, and the two agreed at 74 and 29 when this was written.
func TestArchitectureReviewStatesTheSkipCounts(t *testing.T) {
	review, err := os.ReadFile(filepath.Clean(filepath.Join("..", "..", "docs", "architecture-review.md")))
	if err != nil {
		t.Fatalf("read architecture-review.md: %v", err)
	}
	claim := regexp.MustCompile(`(\d+) 处 ` + "`t.Skip`" + ` 分布在 (\d+) 个测试文件里`)
	m := claim.FindStringSubmatch(string(review))
	if m == nil {
		t.Fatalf(`architecture-review.md no longer states the skip counts as "N 处 t.Skip 分布在 ` +
			`M 个测试文件里"; restore that phrasing or update this pattern`)
	}

	root := filepath.Join("..", "..")
	sites, files := 0, 0
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" || d.Name() == ".cache" {
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, "_test.go") {
			return nil
		}
		src, rerr := os.ReadFile(filepath.Clean(path))
		if rerr != nil {
			return rerr
		}
		// Comments are stripped for the same reason the gate walk strips them: a test that
		// explains "the old code called t.Skip here" is not a skip site.
		n := len(skipCall.FindAllString(stripGoComments(string(src)), -1))
		if n > 0 {
			sites += n
			files++
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking %s: %v", root, err)
	}
	if sites < 20 {
		t.Fatalf("found %d skip sites in the tree, too few to be real -- the walk probably "+
			"stopped early", sites)
	}

	if want := m[1]; want != strconv.Itoa(sites) {
		t.Errorf("architecture-review.md says %s t.Skip sites; the tree has %d", want, sites)
	}
	if want := m[2]; want != strconv.Itoa(files) {
		t.Errorf("architecture-review.md says the skips are in %s test files; the tree has %d",
			want, files)
	}
}

// skipCall matches the call itself. `t.Skipf` and `t.SkipNow` count, because both are a test
// declining to run and the document's figure is about that decision rather than about one
// spelling of it.
var skipCall = regexp.MustCompile(`\bt\.Skip(f|Now)?\(`)
