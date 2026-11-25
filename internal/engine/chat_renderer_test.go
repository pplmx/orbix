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
	// And it has to say what the fallback costs, not only which construct was at fault. The
	// parser's message is about its own grammar; an operator reading it in the startup banner
	// needs to know the server still answers, in a format the model was not trained on.
	//
	// Measured on Qwen3-0.6B's template, which uses namespace(...): under the neutral
	// transcript the model answered a question correctly and then went on inventing turns.
	if !strings.Contains(r.FallbackReason(), "neutral transcript") {
		t.Errorf("FallbackReason = %q, want it to say what the fallback means for the prompt",
			r.FallbackReason())
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

// TestChatRendererDefaultsAnEmptyRoleToUser pins the substitution Render makes before the
// template sees the turn.
//
// Both renderers map "" to "user" -- RenderChatPrompt does it inline and this one does it
// while building the message list -- so the template never sees an empty role. Worth pinning
// because a template that branches on the role would otherwise fall into its else arm and
// render an unattributed turn as something other than the user's, which is a difference no
// percentage of ordinary conversations would show.
func TestChatRendererDefaultsAnEmptyRoleToUser(t *testing.T) {
	// The else arm is what an empty role would hit if the substitution did not happen.
	const src = `{% for m in messages %}{% if m.role == "user" %}U:{% else %}O:{% endif %}{{ m.content }}{% endfor %}`
	f := buildFileWithMeta([]kvPair{
		{name: "tokenizer.chat_template", kind: gguf.TypeString, str: src},
	})
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	if got, want := r.Render([]Turn{{Role: "", Text: "hi"}}), "U:hi"; got != want {
		t.Errorf("Render with an empty role = %q, want %q: the role must reach the template "+
			"as \"user\"", got, want)
	}
	// The neutral transcript makes the same substitution, so the two agree.
	if got, want := RenderChatPrompt([]Turn{{Role: "", Text: "hi"}}), "User: hi\nAssistant:"; got != want {
		t.Errorf("RenderChatPrompt with an empty role = %q, want %q", got, want)
	}
}

// TestSpecialPieceTextEmptyPaths covers the two guards in specialPieceText.
//
// It looks a token id up in the vocabulary because a template interpolates bos_token as
// text, and a file can name an id that its own vocabulary does not contain -- a truncated
// file, or one converted with a different tokenizer. Returning "" is what keeps that from
// panicking during a render, and "" is also what a template sees when the key is absent
// entirely, so the two paths have to agree.
func TestSpecialPieceTextEmptyPaths(t *testing.T) {
	t.Run("key absent", func(t *testing.T) {
		f := buildFileWithMeta([]kvPair{
			{name: "tokenizer.ggml.tokens", kind: gguf.TypeArray, strs: []string{"<s>", "</s>"}},
		})
		if got := specialPieceText(f, "tokenizer.ggml.bos_token_id"); got != "" {
			t.Errorf("specialPieceText without the key = %q, want empty", got)
		}
	})

	t.Run("id past the vocabulary", func(t *testing.T) {
		f := buildFileWithMeta([]kvPair{
			{name: "tokenizer.ggml.tokens", kind: gguf.TypeArray, strs: []string{"<s>", "</s>"}},
			{name: "tokenizer.ggml.bos_token_id", kind: gguf.TypeUint32, num: 99},
		})
		if got := specialPieceText(f, "tokenizer.ggml.bos_token_id"); got != "" {
			t.Errorf("specialPieceText for id 99 of 2 = %q, want empty", got)
		}
	})

	t.Run("vocabulary absent", func(t *testing.T) {
		f := buildFileWithMeta([]kvPair{
			{name: "tokenizer.ggml.bos_token_id", kind: gguf.TypeUint32, num: 0},
		})
		if got := specialPieceText(f, "tokenizer.ggml.bos_token_id"); got != "" {
			t.Errorf("specialPieceText without a vocabulary = %q, want empty", got)
		}
	})

	t.Run("normal", func(t *testing.T) {
		f := buildFileWithMeta([]kvPair{
			{name: "tokenizer.ggml.tokens", kind: gguf.TypeArray, strs: []string{"<s>", "</s>"}},
			{name: "tokenizer.ggml.bos_token_id", kind: gguf.TypeUint32, num: 0},
		})
		if got := specialPieceText(f, "tokenizer.ggml.bos_token_id"); got != "<s>" {
			t.Errorf("specialPieceText for id 0 = %q, want %q", got, "<s>")
		}
	})
}

// TestNilChatRendererPredicates covers the three predicates on a nil renderer.
//
// The server's Config defaults to a nil ChatRenderer and must keep working, which is why
// Render has a nil test. The predicates matter for the same reason and on a different path:
// the REPL asks HasTemplate and UsesModelTemplate on every turn to decide whether a fallback
// is worth announcing, and a nil dereference there would be a panic on a configuration the
// package documents as supported.
func TestNilChatRendererPredicates(t *testing.T) {
	var r *ChatRenderer
	if got := r.FallbackReason(); got != "" {
		t.Errorf("FallbackReason on a nil renderer = %q, want empty", got)
	}
	if r.HasTemplate() {
		t.Error("HasTemplate on a nil renderer = true, want false")
	}
	if r.UsesModelTemplate() {
		t.Error("UsesModelTemplate on a nil renderer = true, want false")
	}
	// Empty rather than a message: a nil renderer is not in a fallback, it is absent, and
	// the REPL only prints a reason when HasTemplate says a template exists.
}

// TestRenderConversationCarriesToolsAndCalls covers what RenderWith cannot express: a request
// that declares tools, and a conversation whose assistant turn carries a call.
//
// The template is a miniature of Qwen2.5's, whose tool branch prints the signatures and whose
// assistant branch prints the call back, including the `{% set %}` rebind that distinguishes this
// package's scoping from Jinja2's until the previous commit. What matters here is the plumbing:
// Parameters reaches the template verbatim, and a call's Arguments are not trimmed or re-encoded
// on the way in.
func TestRenderConversationCarriesToolsAndCalls(t *testing.T) {
	const tmpl = `{%- if tools %}<tools>{% for t in tools %}{{ t | tojson }}{% endfor %}</tools>{% endif %}` +
		`{%- for m in messages %}[{{ m.role }}{% if m.tool_calls %}` +
		`{%- for tc in m.tool_calls %}{% if tc.function is defined %}{% set tc = tc.function %}{% endif %}` +
		`{name:{{ tc.name }},args:{{ tc.arguments | tojson }}}{% endfor %}{% endif %}]{% endfor %}`
	f := buildFileWithMeta([]kvPair{
		{name: "tokenizer.chat_template", kind: gguf.TypeString, str: tmpl},
	})
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	const params = `{"type": "object"}`
	got, err := r.RenderConversation(
		[]Turn{
			{Role: "user", Text: "weather?"},
			{Role: "assistant", Text: "  ", ToolCalls: []ToolCall{{
				ID: "call_1", Name: "get_weather", Arguments: `{"city": "Paris"}`,
			}}},
		},
		[]ToolSpec{{Name: "get_weather", Description: "look it up", Parameters: params}},
		nil)
	if err != nil {
		t.Fatalf("RenderConversation: %v", err)
	}
	// The assistant turn's Text is trimmed to empty, which is why the call follows its role
	// directly. The call itself survives with its arguments intact.
	want := `<tools>{"function": {"description": "look it up", "name": "get_weather", "parameters": ` +
		params + `}, "type": "function"}</tools>` +
		`[user]` +
		`[assistant{name:get_weather,args:{"city": "Paris"}}]`
	if got != want {
		t.Errorf("prompt =\n%q\nwant\n%q", got, want)
	}
}

// TestRenderConversationRefusesToolsWithoutATemplate is the refusal that keeps a declared tool
// from being silently dropped: a model whose template this project cannot evaluate has nowhere to
// print the signatures, and answering anyway would be the failure the server refuses everywhere
// else -- a reply that looks like an answer to a request that was never made.
func TestRenderConversationRefusesToolsWithoutATemplate(t *testing.T) {
	f := buildFileWithMeta([]kvPair{
		// A construct outside the supported subset, so the renderer falls back.
		{name: "tokenizer.chat_template", kind: gguf.TypeString, str: `{% macro m() %}x{% endmacro %}`},
	})
	r, err := NewChatRenderer(f)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	_, err = r.RenderConversation([]Turn{{Role: "user", Text: "hi"}},
		[]ToolSpec{{Name: "f"}}, nil)
	if err == nil {
		t.Fatal("tools were accepted by a model with no usable template")
	}
	if !strings.Contains(err.Error(), "no chat template") {
		t.Errorf("error %q does not say the model has no usable template", err)
	}
	// Without tools the same renderer still answers, so the refusal is about the tools.
	if _, err := r.RenderConversation([]Turn{{Role: "user", Text: "hi"}}, nil, nil); err != nil {
		t.Errorf("a conversation without tools was refused: %v", err)
	}
}
