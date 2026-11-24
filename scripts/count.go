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

func main() {
	var rows []row
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
					if fd, ok := d.(*ast.FuncDecl); ok {
						nfuncs++
						if fd.Recv == nil {
							top++
						}
					}
				}
			}
			rows = append(rows, row{filepath.Base(dir), lines, nfuncs, top, testLines})
		}
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].funcs > rows[j].funcs })
	if len(os.Args) > 1 && os.Args[1] == "-table" {
		printTable(rows)
		return
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
