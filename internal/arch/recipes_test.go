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
		t.Skipf("just is not available: %v", err)
	}
	recipes := parseRecipes(string(listed))
	if len(recipes) == 0 {
		t.Fatal("parsed no recipes from just --list; the output format changed")
	}

	// CONTRIBUTING.md is the one that has to be complete. It is the list a person reads when they
	// clone the repository; AGENTS.md is for agents and may group a recipe under a longer
	// explanation instead of a line per recipe. Checking the union would pass with CONTRIBUTING.md
	// missing entries, which is exactly the drift this is here to catch.
	body, err := os.ReadFile(filepath.Join(root, "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
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
		t.Errorf("recipe %q is in the justfile but not named in CONTRIBUTING.md;"+
			" add a line for it where the other recipes are listed", r)
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
