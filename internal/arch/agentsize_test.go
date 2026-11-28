package arch

import (
	"os"
	"path/filepath"
	"testing"
)

// agentDocBudget is the largest AGENTS.md a harness will load whole.
//
// The file exists to be read by an agent at the start of a session, not only by a person opening
// it in an editor, and the loaders put a cap on how much of it they will take. Past that cap the
// tail is dropped without an error anywhere a maintainer would see it: the file on disk is
// complete, `just docs` is green, and the only symptom is that a convention written near the end
// stops having any effect.
//
// Measured 2026-11-28: AGENTS.md reached 66155 bytes against a 65536-byte budget, so the last
// section -- the CI verification record -- was being truncated for every new session. Nothing
// failed. It was found by noticing the truncation notice in a tool result, not by any check in
// this repository.
//
// 65536 is not a number this repository controls, so it is written here rather than derived. If a
// loader's cap changes, this constant changes with it; the point is that exceeding the current one
// fails a test instead of quietly shortening the file for its readers.
const agentDocBudget = 65536

// TestAgentsDocFitsTheLoadBudget refuses an AGENTS.md that would be read in part.
//
// The rule is deliberately about size and not about content: which conventions belong in the file
// is a judgement, and the earlier sections say what they are for. What is mechanical is that the
// file must stay within what a loader will take whole, and that is what this checks.
//
// When it fails, the fix is to move a record to docs/ -- the CI verification section went to
// docs/architecture-review.md's 复算清单 that way, which is where the other re-measurement records
// already lived -- rather than to compress an argument. The three largest sections were measured
// during that move (测试 23KB, 入口 14.5KB, 判断一条测量 8KB) and all three were real content.
func TestAgentsDocFitsTheLoadBudget(t *testing.T) {
	path := filepath.Join("..", "..", "AGENTS.md")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(body) > agentDocBudget {
		t.Errorf("AGENTS.md is %d bytes, over the %d-byte budget by %d; "+
			"a loader will drop the last %d bytes, which is roughly the final section. "+
			"Move a record to docs/ instead of compressing an argument.",
			len(body), agentDocBudget, len(body)-agentDocBudget, len(body)-agentDocBudget)
	}

	// A budget test that passes on an empty file is not checking anything, and the file being
	// emptied is a plausible accident (a bad redirect, a truncated checkout).
	if len(body) < agentDocBudget/2 {
		t.Errorf("AGENTS.md is %d bytes, less than half the budget; "+
			"that is more likely a truncated file than a rewrite", len(body))
	}
}
