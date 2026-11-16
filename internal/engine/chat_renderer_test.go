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
	got, err := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
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
	got, err := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
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
	got, err := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if err != nil {
		t.Fatalf("Render returned an error instead of falling back: %v", err)
	}
	if !strings.Contains(got, "User: hi") {
		t.Errorf("got %q, want the neutral transcript", got)
	}
}

// TestNilChatRendererUsesNeutral makes the nil case explicit, since the server's Config
// defaults to nil and must keep working.
func TestNilChatRendererUsesNeutral(t *testing.T) {
	var r *ChatRenderer
	got, err := r.Render([]Turn{{Role: "user", Text: "hi"}})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != RenderChatPrompt([]Turn{{Role: "user", Text: "hi"}}) {
		t.Errorf("nil renderer produced %q", got)
	}
}
