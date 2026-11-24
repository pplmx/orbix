package engine

import (
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
)

// TestRenderWithRefusesAVariableTheTemplateNeverReads covers the second refusal RenderWith makes.
//
// The first -- variables on a model with no template -- is reached through the HTTP layer and
// fails there. This one needs a model that *has* a template and a variable that template does not
// mention, and nothing exercised it: it is 0% both in this package and across the repository.
//
// The refusal is what makes chat_template_kwargs safe to accept from a caller. Without it a
// request setting enable_thinking against a template that never reads it would render the default
// branch and answer normally, with nothing saying the switch was ignored.
func TestRenderWithRefusesAVariableTheTemplateNeverReads(t *testing.T) {
	// A template that reads messages and nothing else.
	path := gguftest.Model{
		NoEOS:      true,
		ContextLen: 128,
		ChatTemplate: `{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}
{% endfor %}`,
	}.Write(t, t.TempDir())

	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	turns := []Turn{{Role: "user", Text: "hi"}}

	// A variable the template does read is fine to set.
	if _, err := r.RenderWith(turns, map[string]any{"enable_thinking": false}); err == nil {
		t.Fatal("setting a variable no template reads was accepted")
	} else if !strings.Contains(err.Error(), "enable_thinking") {
		t.Errorf("error %q does not name the variable", err)
	}

	// And with no variables it renders, so the refusal above is about the variable rather than
	// about the template being unusable.
	if out, err := r.RenderWith(turns, nil); err != nil {
		t.Fatalf("RenderWith with no variables: %v", err)
	} else if out == "" {
		t.Error("rendered nothing")
	}
}
