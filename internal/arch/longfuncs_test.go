package arch

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestArchitectureReviewLongFunctionCountsMatchTheCode checks the three figures
// docs/architecture-review.md quotes about function sizes against what `count.go -long`
// prints today.
//
// This document has now gone stale twice. It said "seventeen" while the script printed a
// truncated list; it said "nineteen, 751 functions" while the code held twenty and 759; and
// nothing noticed either time, because a number in a document is not a broken link and no
// other test reads it. The prose around those numbers is a judgement call and stays
// hand-written -- what this checks is only what the tool can answer.
//
// `chat.parsePostfix` is the one worth remembering: it was 79 lines, then `2605ec3` grew it
// to 113 while adding the constructs Qwen3's template needs. Nobody edited it badly; a
// feature landed in it.
//
// The script is run rather than reimplemented so the test cannot drift from the document's
// own recompute command.
func TestArchitectureReviewLongFunctionCountsMatchTheCode(t *testing.T) {
	root := "../.."

	cmd := exec.Command("go", "run", "./scripts/count.go", "-long")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go run ./scripts/count.go -long: %v", err)
	}
	total, atOrAbove, got := parseLongReport(t, string(out))

	body, err := os.ReadFile(filepath.Join(root, "docs", "architecture-review.md"))
	if err != nil {
		t.Fatalf("read architecture-review.md: %v", err)
	}
	// A digit run that is not part of a longer number, so "759" cannot be matched inside
	// "1759" and the counts are compared as the document wrote them.
	wantAtOrAbove, wantFuncs, _ := longCountClaims(t, string(body))

	if wantFuncs != total {
		t.Errorf("the document says %d functions; count.go reports %d", wantFuncs, total)
	}
	if wantAtOrAbove != atOrAbove {
		t.Errorf("the document says %d functions at or above 100 lines; count.go reports %d",
			wantAtOrAbove, atOrAbove)
	}

	// The named list has to name exactly the functions that are over the threshold: a count
	// alone passes while the list names the wrong ones, which is the failure the previous
	// revision of the document had -- nineteen names, one of them wrong, and one missing.
	// Only the list paragraph counts. Scanning the whole document picked up every
	// `pkg.Symbol` in the prose -- `cli.cmdRun`, `sync.Mutex`, `engine.Workers` -- and
	// reported each as a name that should be in the list. The list starts at the
	// ">=100 行函数" sentence and runs until the blank line after it.
	listStart := docFuncsList.FindStringIndex(string(body))
	if listStart == nil {
		t.Fatal("cannot find the long-function list paragraph")
	}
	listEnd := strings.Index(string(body)[listStart[1]:], "\n\n")
	if listEnd < 0 {
		t.Fatal("the long-function list paragraph has no end")
	}
	list := string(body)[listStart[1] : listStart[1]+listEnd]

	// The document writes `pkg.func` and count.go prints the bare function name, so the
	// package prefix is dropped on the document side. Keeping the prefix would make the two
	// sets disjoint and the comparison vacuous, which is the failure this test exists to
	// avoid in the document.
	documented := map[string]bool{}
	for _, m := range docFuncName.FindAllStringSubmatch(list, -1) {
		name := m[1]
		if i := strings.LastIndex(name, "."); i >= 0 {
			name = name[i+1:]
		}
		documented[name] = true
	}
	if len(documented) == 0 {
		t.Fatal("parsed no function names from the list paragraph; the format changed")
	}
	actual := map[string]bool{}
	for _, f := range got {
		actual[f.name] = true
	}
	for name := range actual {
		if !documented[name] {
			t.Errorf("%s is %d lines or more but the document's list does not name it",
				name, lenLong)
		}
	}
	for name := range documented {
		if !actual[name] {
			t.Errorf("the document's list names %s, which count.go does not report as long", name)
		}
	}
}

// lenLong mirrors the threshold printLong uses. It is a literal rather than a reference to
// the script's constant: the script is a separate program and the document is what is being
// checked, so reading the number from either would let both move together.
const lenLong = 100

// parseLongReport reads the two figures and the list out of `count.go -long`'s output.
//
// The first line is "759 functions, 20 at or above 100 lines" and each following non-empty
// line is "  310  evalExpr   internal/chat/eval.go". Parsing the summary rather than counting
// the list is deliberate: if the two ever disagree the script has a bug, and this test should
// then fail on the summary it was written to check rather than silently agree with the list.
func parseLongReport(t *testing.T, out string) (total, atOrAbove int, fns []fnLike) {
	t.Helper()
	summary := longSummary.FindStringSubmatch(out)
	if summary == nil {
		t.Fatalf("count.go -long's summary line is not in the expected form; output:\n%s", out)
	}
	total, err := strconv.Atoi(summary[1])
	if err != nil {
		t.Fatalf("summary total %q: %v", summary[1], err)
	}
	atOrAbove, err = strconv.Atoi(summary[2])
	if err != nil {
		t.Fatalf("summary count %q: %v", summary[2], err)
	}
	for _, m := range longRow.FindAllStringSubmatch(out, -1) {
		n, err := strconv.Atoi(m[1])
		if err != nil {
			t.Fatalf("row size %q: %v", m[1], err)
		}
		fns = append(fns, fnLike{lines: n, name: m[2]})
	}
	if len(fns) != atOrAbove {
		t.Fatalf("the summary says %d functions are long but the list has %d rows",
			atOrAbove, len(fns))
	}
	return total, atOrAbove, fns
}

type fnLike struct {
	lines int
	name  string
}

var (
	longSummary = regexp.MustCompile(`(?m)^(\d+) functions, (\d+) at or above \d+ lines$`)
	longRow     = regexp.MustCompile(`(?m)^\s+(\d+)  (\S+)\s+\S+\.go$`)

	// The document states the figures twice: once as "N 个函数在 100 行以上 ... M 个函数" in
	// the section intro, and once as "N 个 >=100 行函数（... M 个函数）" above the list. Both
	// are read, so a revision that updates one and not the other fails here.
	docAtOrAboveIntro = regexp.MustCompile(`\*\*(\d+) 个函数在 100 行以上\*\*`)
	docFuncsIntro     = regexp.MustCompile(`统计骗成几千行；(\d+) 个函数`)
	// Either numeral form is accepted, because writing the count as 二十 is a reasonable
	// choice for this document and a pattern that only takes digits would reject the correct
	// document for a reason that has nothing to do with the count. A numeral the map below
	// does not know is reported as such rather than silently skipped.
	docAtOrAboveList = regexp.MustCompile(`\*\*现存的?\s*([0-9]+|[一二三四五六七八九十]{1,3})\s*个 >=100 行函数\*\*`)
	docFuncsList     = regexp.MustCompile(`个 >=100 行函数\*\*（[^，]*，(\d+) 个函数）`)

	// A function the document names in its list, as `pkg.name` in backticks.
	docFuncName = regexp.MustCompile("`(\\w+\\.\\w+)`")
)

// longCountClaims extracts the two figures from each of the two places the document states
// them, and requires the four to agree before comparing any against the code. A document that
// contradicts itself is a different defect from one that lags the code, and the message should
// say which.
func longCountClaims(t *testing.T, body string) (atOrAbove, funcs, wantAtOrAbove int) {
	t.Helper()
	introA := oneMatch(t, docAtOrAboveIntro, body, "the '**N 个函数在 100 行以上**' figure")
	introF := oneMatch(t, docFuncsIntro, body, "the intro's function total")
	listA := oneMatch(t, docAtOrAboveList, body, "the '**现存的N 个 >=100 行函数**' figure")
	listF := oneMatch(t, docFuncsList, body, "the list's function total")
	if introA != listA || introF != listF {
		t.Fatalf("the document disagrees with itself: the intro says %d long of %d functions, "+
			"the list says %d long of %d; fix that before comparing with the code",
			introA, introF, listA, listF)
	}
	return introA, introF, listA
}

func oneMatch(t *testing.T, re *regexp.Regexp, body, what string) int {
	t.Helper()
	m := re.FindStringSubmatch(body)
	if m == nil {
		t.Fatalf("docs/architecture-review.md no longer states %s in the form this test reads; "+
			"either restore it or update the pattern with the sentence", what)
	}
	if n, ok := chineseNumerals[m[1]]; ok {
		return n
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		t.Fatalf("%s %q is neither a number nor a numeral this test knows: %v", what, m[1], err)
	}
	return n
}

// chineseNumerals covers the counts a document about this repository plausibly writes out.
// It stops at 29 because that is above anything this list has reached; a larger count has to
// be written as digits, which is the form the tool prints and the form the check prefers.
var chineseNumerals = map[string]int{
	"十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
	"十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
	"二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24, "二十五": 25,
	"二十六": 26, "二十七": 27, "二十八": 28, "二十九": 29,
}

// TestArchitectureReviewSizeTableMatchesTheCode checks the twelve-row size table.
//
// It is the most numeric thing in the repository -- five figures per package -- and the only
// part of that document with no guard. Measured before writing this: four of the sixty cells
// were wrong, all from this session's own edits (chat's lines, functions and top-level count,
// and the test-line columns for chat, llama and quant). None of them were noticed by anyone;
// they were found by comparing against `go run ./scripts/count.go`, which is the command the
// document itself gives for reproducing the table.
//
// Lines, functions and top-level functions come from count.go rather than from a copy here, for
// the reason the long-function guard records: the document's reproduce command and the guard
// should be the same computation. Test lines and the ratio are measured here, because count.go
// does not cover them.
func TestArchitectureReviewSizeTableMatchesTheCode(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	doc, err := os.ReadFile(filepath.Join(root, "docs", "architecture-review.md"))
	if err != nil {
		t.Fatalf("read architecture-review.md: %v", err)
	}

	rows := sizeTableRows(t, string(doc))
	if len(rows) < 12 {
		t.Fatalf("parsed %d size-table rows, too few; the table moved or its shape changed", len(rows))
	}

	counts := countGoOutput(t, root)
	for _, r := range rows {
		short := r.pkg[strings.LastIndex(r.pkg, "/")+1:]
		got, ok := counts[short]
		if !ok {
			t.Errorf("the table names %q but count.go reports nothing for %q", r.pkg, short)
			continue
		}
		if r.lines != got.lines {
			t.Errorf("%s: the document says %d code lines, count.go says %d", r.pkg, r.lines, got.lines)
		}
		if r.funcs != got.funcs {
			t.Errorf("%s: the document says %d functions, count.go says %d", r.pkg, r.funcs, got.funcs)
		}
		if r.top != got.top {
			t.Errorf("%s: the document says %d top-level functions, count.go says %d",
				r.pkg, r.top, got.top)
		}

		// Test lines, which count.go does not measure.
		testLines := packageTestLines(t, filepath.Join(root, r.pkg))
		if r.testLines != testLines {
			t.Errorf("%s: the document says %d test lines, measured %d", r.pkg, r.testLines, testLines)
		}
		// And the ratio the last column prints, recomputed from the two figures beside it.
		if want := float64(testLines) / float64(r.lines); math.Abs(want-r.ratio) > 0.005 {
			t.Errorf("%s: the document says the test/code ratio is %.2f, %d/%d is %.2f",
				r.pkg, r.ratio, testLines, r.lines, want)
		}
	}
}

// sizeRow is one line of the document's table.
type sizeRow struct {
	pkg       string
	lines     int
	funcs     int
	top       int
	testLines int
	ratio     float64
}

// sizeTableRows parses the twelve-row table that opens the document.
//
// Anchored on the header rather than on position: the document holds many tables, and the
// long-function section has one with similar-looking numeric cells.
func sizeTableRows(t *testing.T, doc string) []sizeRow {
	t.Helper()
	var out []sizeRow
	inTable := false
	for line := range strings.SplitSeq(doc, "\n") {
		if !strings.HasPrefix(line, "|") {
			if inTable && len(out) > 0 {
				break
			}
			continue
		}
		if strings.Contains(line, "测试/代码") {
			inTable = true
			continue
		}
		if !inTable || strings.Contains(line, "---") {
			continue
		}
		m := sizeRowPattern.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		out = append(out, sizeRow{
			pkg:       m[1],
			lines:     archAtoi(t, m[2]),
			funcs:     archAtoi(t, m[3]),
			top:       archAtoi(t, m[4]),
			testLines: archAtoi(t, m[5]),
			ratio:     archAtof(t, m[6]),
		})
	}
	return out
}

var sizeRowPattern = regexp.MustCompile(
	`^\|\s*` + "`" + `([\w/]+)` + "`" + `\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|`)

// sizeCounts is one package's figures from count.go.
type sizeCounts struct{ lines, funcs, top int }

// countGoOutput runs the document's own reproduce command and parses it.
func countGoOutput(t *testing.T, root string) map[string]sizeCounts {
	t.Helper()
	cmd := exec.Command("go", "run", "./scripts/count.go")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go run ./scripts/count.go: %v", err)
	}
	got := map[string]sizeCounts{}
	for line := range strings.SplitSeq(string(out), "\n") {
		m := countGoPattern.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		got[m[1]] = sizeCounts{
			lines: archAtoi(t, m[4]),
			funcs: archAtoi(t, m[2]),
			top:   archAtoi(t, m[3]),
		}
	}
	if len(got) == 0 {
		t.Fatal("count.go printed no rows this test could parse; its format changed")
	}
	return got
}

var countGoPattern = regexp.MustCompile(
	`^\s+\d+\.\s+(\S+)\s+funcs=\s*(\d+) toplevel=\s*(\d+) lines=\s*(\d+)`)

// packageTestLines counts the lines in a package's _test.go files.
func packageTestLines(t *testing.T, dir string) int {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	total := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		total += strings.Count(string(raw), "\n")
	}
	return total
}

// archAtof parses a ratio cell.
func archAtof(t *testing.T, s string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("%q is not a number: %v", s, err)
	}
	return v
}
