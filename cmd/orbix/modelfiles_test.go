package main

import (
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestChatFormatNamesWhichTranscriptIsInUse covers both branches of the banner line.
//
// The distinction is the point of the line: a model whose template could not be applied answers
// in a shape the caller did not expect, and saying so at startup is the difference between a bug
// report and a mystery. The model-template branch had no test -- every fixture in this package
// declares no template, so only the fallback path was ever taken.
func TestChatFormatNamesWhichTranscriptIsInUse(t *testing.T) {
	t.Run("a model with a template", func(t *testing.T) {
		// A template this package's evaluator accepts, so the probe succeeds and the renderer
		// keeps it.
		const tmpl = `{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}`
		path := gguftest.Model{NoEOS: true, ContextLen: 256, ChatTemplate: tmpl}.Write(t, t.TempDir())
		m, err := openModel(path)
		if err != nil {
			t.Fatalf("openModel: %v", err)
		}
		if !m.renderer.UsesModelTemplate() {
			t.Fatalf("the fixture's template was not adopted, so this case cannot check the "+
				"branch it names: %s", m.renderer.FallbackReason())
		}
		got := m.ChatFormat()
		if !strings.Contains(got, "model's own chat template") {
			t.Errorf("ChatFormat = %q, want it to name the model's template", got)
		}
		// The fallback reason must not leak into this branch: naming a reason implies a
		// fallback, which would be the opposite of what happened.
		if strings.Contains(got, "neutral") {
			t.Errorf("ChatFormat = %q mentions the neutral transcript for a model whose own "+
				"template is in use", got)
		}
	})

	t.Run("a model without one", func(t *testing.T) {
		path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
		m, err := openModel(path)
		if err != nil {
			t.Fatalf("openModel: %v", err)
		}
		if m.renderer.UsesModelTemplate() {
			t.Fatal("a model with no template reported using one")
		}
		got := m.ChatFormat()
		if !strings.Contains(got, "neutral transcript") {
			t.Errorf("ChatFormat = %q, want it to name the neutral transcript", got)
		}
		// The reason comes along, because "neutral transcript" alone does not say whether the
		// file declared no template or declared one this package cannot evaluate.
		if reason := m.renderer.FallbackReason(); reason != "" &&
			!strings.Contains(got, reason) {
			t.Errorf("ChatFormat = %q does not include the reason %q it says it has", got, reason)
		}
	})
}

// TestChatFormatDiffersBetweenTheTwoModels is the property the banner depends on.
//
// If both branches produced the same string, the line would be useless and neither test above
// would mean anything. Comparing them directly makes that impossible to miss.
func TestChatFormatDiffersBetweenTheTwoModels(t *testing.T) {
	withTemplate := gguftest.Model{
		NoEOS: true, ContextLen: 256,
		ChatTemplate: `{% for m in messages %}{{ m.content }}{% endfor %}`,
	}.Write(t, t.TempDir())
	withoutTemplate := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())

	a, err := openModel(withTemplate)
	if err != nil {
		t.Fatalf("openModel(with template): %v", err)
	}
	b, err := openModel(withoutTemplate)
	if err != nil {
		t.Fatalf("openModel(without template): %v", err)
	}
	if a.ChatFormat() == b.ChatFormat() {
		t.Errorf("both models report %q, so the banner cannot tell an operator which "+
			"transcript is in use", a.ChatFormat())
	}
}
