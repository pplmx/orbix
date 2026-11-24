package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type row struct {
	name      string
	lines     int
	funcs     int // all FuncDecl, methods included
	topfn     int
	testLines int
}

// fn is one function's size, for the -long report.
type fn struct {
	name  string
	lines int
	file  string
}

func main() {
	var rows []row
	var fns []fn
	base, _ := os.Getwd()
	for _, pat := range []string{"internal/*", "cmd/orbix"} {
		paths, _ := filepath.Glob(filepath.Join(base, pat))
		for _, dir := range paths {
			st, err := os.Stat(dir)
			if err != nil || !st.IsDir() {
				continue
			}
			files, _ := filepath.Glob(filepath.Join(dir, "*.go"))
			var lines, nfuncs, top, testLines int
			for _, f := range files {
				if strings.HasSuffix(filepath.Base(f), "_test.go") {
					if data, err := os.ReadFile(f); err == nil {
						testLines += strings.Count(string(data), "\n")
					}
					continue
				}
				data, err := os.ReadFile(f)
				if err != nil {
					continue
				}
				lines += strings.Count(string(data), "\n")
				fs := token.NewFileSet()
				af, err := parser.ParseFile(fs, f, data, 0)
				if err != nil {
					continue
				}
				for _, d := range af.Decls {
					fd, ok := d.(*ast.FuncDecl)
					if !ok {
						continue
					}
					nfuncs++
					if fd.Recv == nil {
						top++
					}
					// A body-less declaration has no lines to count.
					if fd.Body == nil {
						continue
					}
					fns = append(fns, fn{
						name:  fd.Name.Name,
						lines: fs.Position(fd.Body.Rbrace).Line - fs.Position(fd.Pos()).Line + 1,
						file:  filepath.ToSlash(relTo(base, f)),
					})
				}
			}
			rows = append(rows, row{filepath.Base(dir), lines, nfuncs, top, testLines})
		}
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].funcs > rows[j].funcs })
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "-table":
			printTable(rows)
			return
		case "-long":
			printLong(fns)
			return
		}
	}

	fmt.Println("rank by ALL funcs (methods included):")
	for i, r := range rows {
		fmt.Printf("  %d. %-12s funcs=%4d toplevel=%3d lines=%5d\n", i+1, r.name, r.funcs, r.topfn, r.lines)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].lines > rows[j].lines })
	fmt.Println("rank by LINES:")
	for i, r := range rows {
		fmt.Printf("  %d. %-12s lines=%5d funcs=%4d\n", i+1, r.name, r.lines, r.funcs)
	}
}

// printTable writes the table docs/architecture-review.md carries, so the document can be
// regenerated rather than retyped.
//
// The ranking output above is what the analysis was originally run for; this mode exists because a
// reader comparing the document against the code had to translate between two layouts before they
// could tell whether a figure had drifted.
func printTable(rows []row) {
	var totalLines, totalFuncs, totalTop, totalTestLines int
	sorted := make([]row, len(rows))
	copy(sorted, rows)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].lines > sorted[j].lines })

	fmt.Println("| 包                  | 代码行    | 函数（含方法） | 顶层函数 | 测试行    | 测试/代码 |")
	fmt.Println("| ------------------- | --------- | -------------- | -------- | --------- | --------- |")
	for _, r := range sorted {
		name := r.name
		if name != "orbix" {
			name = "internal/" + name
		} else {
			name = "cmd/orbix"
		}
		ratio := 0.0
		if r.lines > 0 {
			ratio = float64(r.testLines) / float64(r.lines)
		}
		fmt.Printf("| %-19s | %-9d | %-14d | %-8d | %-9d | %-9.2f |\n",
			"`"+name+"`", r.lines, r.funcs, r.topfn, r.testLines, ratio)
		totalLines += r.lines
		totalFuncs += r.funcs
		totalTop += r.topfn
		totalTestLines += r.testLines
	}
	totalRatio := 0.0
	if totalLines > 0 {
		totalRatio = float64(totalTestLines) / float64(totalLines)
	}
	fmt.Printf("| %-19s | %-9s | %-14s | %-8s | %-9s | %-9s |\n", "**合计**",
		fmt.Sprintf("**%d**", totalLines), fmt.Sprintf("**%d**", totalFuncs),
		fmt.Sprintf("**%d**", totalTop), fmt.Sprintf("**%d**", totalTestLines),
		fmt.Sprintf("**%.2f**", totalRatio))
}

// printLong lists every function at or above the threshold, longest first.
//
// This replaces a scratch tool that printed only the first twelve and a total that said
// seventeen; the twelve reached a document and the five it hid went unnoticed for weeks. The
// limit that was useful as a terminal view is the wrong thing to hand a reader who is asking
// "what is still long".
//
// The count line and the list are printed from the same slice, so they cannot disagree.
func printLong(fns []fn) {
	const threshold = 100
	sorted := make([]fn, len(fns))
	copy(sorted, fns)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].lines != sorted[j].lines {
			return sorted[i].lines > sorted[j].lines
		}
		return sorted[i].name < sorted[j].name
	})
	var atOrAbove []fn
	for _, f := range sorted {
		if f.lines >= threshold {
			atOrAbove = append(atOrAbove, f)
		}
	}
	fmt.Printf("%d functions, %d at or above %d lines\n\n", len(fns), len(atOrAbove), threshold)
	for _, f := range atOrAbove {
		fmt.Printf("  %4d  %-32s %s\n", f.lines, f.name, f.file)
	}
}

// relTo returns target relative to base, or target unchanged when it is not under base.
func relTo(base, target string) string {
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return target
	}
	return rel
}
