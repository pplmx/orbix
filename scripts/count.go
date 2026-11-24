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
	name  string
	lines int
	funcs int // all FuncDecl, methods included
	topfn int
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
			var lines, nfuncs, top int
			for _, f := range files {
				if strings.HasSuffix(filepath.Base(f), "_test.go") {
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
			rows = append(rows, row{filepath.Base(dir), lines, nfuncs, top})
		}
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].funcs > rows[j].funcs })
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
