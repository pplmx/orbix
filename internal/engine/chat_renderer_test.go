package engine

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// kvPair is one metadata entry for buildFileWithMeta.
type kvPair struct {
	name string
	kind gguf.ValueType
	str  string
	num  int
	strs []string
}

// buildFileWithMeta assembles a GGUF carrying only metadata, which is all a
// ChatRenderer reads.
func buildFileWithMeta(kvs []kvPair) *gguf.File {
	var b bytes.Buffer
	b.WriteString(gguf.Magic)
	putU32 := func(v uint32) {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], v)
		b.Write(tmp[:])
	}
	putU64 := func(v uint64) {
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], v)
		b.Write(tmp[:])
	}
	str := func(s string) {
		putU64(uint64(len(s)))
		b.WriteString(s)
	}
	putU32(gguf.Version)
	putU64(0) // tensors
	putU64(uint64(len(kvs)))
	for _, kv := range kvs {
		str(kv.name)
		putU32(uint32(kv.kind))
		switch kv.kind {
		case gguf.TypeString:
			str(kv.str)
		case gguf.TypeInt32, gguf.TypeUint32:
			putU32(uint32(kv.num))
		case gguf.TypeArray:
			putU32(uint32(gguf.TypeString))
			putU64(uint64(len(kv.strs)))
			for _, s := range kv.strs {
				str(s)
			}
		}
	}
	f, err := gguf.Read(bytes.NewReader(b.Bytes()))
	if err != nil {
		panic(err)
	}
	return f
}

// TestChatRendererFallsBackWithoutTemplate checks the common case: a base model has no
// chat format, so the renderer must produce the neutral transcript and say why.
func TestChatRendererFallsBackWithoutTemplate(t *testing.T) {
	f := buildFileWithMeta(nil)
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	if r.UsesModelTemplate() {
		t.Error("UsesModelTemplate = true for a file with no template")
	}
	if r.FallbackReason() == "" {
		t.Error("FallbackReason is empty; a fallback must say why")
	}
	got := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if !strings.Contains(got, "User: hi") {
		t.Errorf("neutral transcript = %q", got)
	}
}

// TestChatRendererFallsBackOnUnsupportedTemplate is the property that keeps an
// unsupported checkpoint usable: an unparseable template does not fail, it degrades to
// the neutral transcript with the reason recorded.
func TestChatRendererFallsBackOnUnsupportedTemplate(t *testing.T) {
	f := buildFileWithMeta([]kvPair{
		{name: "tokenizer.chat_template", kind: gguf.TypeString, str: "{% macro x() %}{% endmacro %}"},
	})
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	if r.UsesModelTemplate() {
		t.Error("UsesModelTemplate = true for an unparseable template")
	}
	if !strings.Contains(r.FallbackReason(), "macro") {
		t.Errorf("FallbackReason = %q, want it to name the unsupported construct", r.FallbackReason())
	}
}

// TestChatRendererUsesModelTemplate checks the happy path end to end, including that the
// BOS piece is interpolated as text rather than as an id.
func TestChatRendererUsesModelTemplate(t *testing.T) {
	tmpl := `{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}{{ bos_token }}`
	f := buildFileWithMeta([]kvPair{
		{name: "tokenizer.chat_template", kind: gguf.TypeString, str: tmpl},
		{name: "tokenizer.ggml.bos_token_id", kind: gguf.TypeUint32, num: 1},
		{name: "tokenizer.ggml.tokens", kind: gguf.TypeArray, strs: []string{"<unk>", "<s>"}},
	})
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	if !r.UsesModelTemplate() {
		t.Fatalf("UsesModelTemplate = false: %s", r.FallbackReason())
	}
	got := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if want := "[user:hi]<s>"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestChatRendererRenderFailureFallsBack checks that a template which compiles but fails
// on this conversation still yields a prompt, so a request never stalls on a template
// bug.
func TestChatRendererRenderFailureFallsBack(t *testing.T) {
	// messages[1] does not exist for a one-turn conversation.
	f := buildFileWithMeta([]kvPair{
		{name: "tokenizer.chat_template", kind: gguf.TypeString, str: `{{ messages[1].content }}`},
	})
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	got := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if !strings.Contains(got, "User: hi") {
		t.Errorf("got %q, want the neutral transcript", got)
	}
}

// TestNilChatRendererUsesNeutral makes the nil case explicit, since the server's Config
// defaults to nil and must keep working.
func TestNilChatRendererUsesNeutral(t *testing.T) {
	var r *ChatRenderer
	got := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if got != RenderChatPrompt([]Turn{{Role: "user", Text: "hi"}}) {
		t.Errorf("nil renderer produced %q", got)
	}
}

// TestChatRendererReportsARenderFailure checks the case between the two the file already
// covers: a template that compiles but cannot render a particular conversation.
//
// Falling back is right -- the session stays usable -- but the renderer used to fall back
// silently, so UsesModelTemplate stayed true and the caller reported the model's own format
// while every prompt was the neutral transcript. Qwen2's template did exactly this: it reads
// message.tool_calls, the field was missing, and a two-turn conversation came out in the
// neutral format with "Human:" in it and nothing to suggest why.
func TestChatRendererReportsARenderFailure(t *testing.T) {
	// Parses, but fails to render: for over a name that is not a sequence. A missing
	// map field is empty rather than an error, so it would not do here.
	const src = `{%- for x in bos_token %}{{ x }}{%- endfor %}`
	f := buildFileWithMeta([]kvPair{
		{name: "tokenizer.chat_template", kind: gguf.TypeString, str: src},
	})
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	if !r.UsesModelTemplate() {
		t.Fatalf("the template compiled, so it should be in use before rendering: %s",
			r.FallbackReason())
	}

	out := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if !strings.Contains(out, "User: hi") {
		t.Errorf("the fallback did not produce a neutral transcript: %q", out)
	}

	if r.UsesModelTemplate() {
		t.Error("after a render failure the template is still reported as in use")
	}
	if r.FallbackReason() == "" {
		t.Error("no reason was recorded for the render failure")
	}
}

// TestChatRendererHasTemplateSeparatesTheTwoFallbacks pins the pair the REPL needs.
//
// UsesModelTemplate answers "is the template in use", which is false both for a model that
// declared no template and for one whose template failed on this conversation. HasTemplate
// answers "was there a template at all", which distinguishes them. The REPL needs the
// difference to decide whether a fallback is news: a model with no template already said so in
// its banner, so announcing a render-time failure on every turn of such a session describes
// something that never happened.
func TestChatRendererHasTemplateSeparatesTheTwoFallbacks(t *testing.T) {
	t.Run("no template declared", func(t *testing.T) {
		f := buildFileWithMeta(nil)
		r, err := NewChatRenderer(f)
		if err != nil {
			t.Fatalf("NewChatRenderer: %v", err)
		}
		if r.HasTemplate() {
			t.Error("HasTemplate = true for a file that declares no template")
		}
		if r.UsesModelTemplate() {
			t.Error("UsesModelTemplate = true for a file that declares no template")
		}
		// Both false: nothing to report, which is what makes this case quiet.
	})

	t.Run("template fails to render", func(t *testing.T) {
		// Parses, then fails: iterating bos_token, which is not a sequence.
		const src = `{%- for x in bos_token %}{{ x }}{%- endfor %}`
		f := buildFileWithMeta([]kvPair{
			{name: "tokenizer.chat_template", kind: gguf.TypeString, str: src},
		})
		r, err := NewChatRenderer(f)
		if err != nil {
			t.Fatalf("NewChatRenderer: %v", err)
		}
		r.Render([]Turn{{Role: "user", Text: "hi"}})
		if !r.HasTemplate() {
			t.Error("HasTemplate = false, but the file did declare a template that parsed")
		}
		if r.UsesModelTemplate() {
			t.Error("UsesModelTemplate = true after the render failed")
		}
		// HasTemplate true and UsesModelTemplate false: this is the pair the REPL reports.
	})
}
