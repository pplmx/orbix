package chat

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These are the templates of two widely used checkpoints, copied from their
// tokenizer_config.json. They are the reason the supported subset is what it is: each
// construct here was added because one of these needed it, and a change that breaks
// either is caught without downloading a model.

// qwen2Template is Qwen2.5-Instruct's template, minus the tool-calling branch's
// selectattr/map usage, which this package does not implement. The tools branch itself
// is kept because it exercises tojson and a nested for.
const qwen2Template = `{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\n" }}
    {%- for tool in tools %}
        {{- tool | tojson }}
    {%- endfor %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}`

// llama3Template is Llama 3.1-Instruct's template in full.
const llama3Template = `{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}`

// TestQwen2TemplateRenders checks the exact prompt Qwen2 expects, including the default
// system turn it inserts when the conversation has none.
func TestQwen2TemplateRenders(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "What is 2+2?"}}
	got := render(t, qwen2Template, msgs, Inputs{AddGenerationPrompt: true})
	want := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nWhat is 2+2?<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
}

// TestQwen2TemplateUsesSystemMessage checks the branch that matters when a caller does
// supply a system prompt: the model's own default must not appear as well.
func TestQwen2TemplateUsesSystemMessage(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "hi"},
	}
	got := render(t, qwen2Template, msgs, Inputs{AddGenerationPrompt: false})
	want := "<|im_start|>system\nYou are terse.<|im_end|>\n" +
		"<|im_start|>user\nhi<|im_end|>\n"
	if got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
}

// TestLlama3TemplateRenders checks Llama 3's format, which uses a different marker set
// and a trim filter on the content.
func TestLlama3TemplateRenders(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "  hi  "}}
	got := render(t, llama3Template, msgs, Inputs{BOSToken: "<|begin_of_text|>"})
	want := "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\n"
	if got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
}

// TestRealTemplatesRejectNothing pins the parse step separately: a template that
// compiles cannot later surprise the renderer with an unsupported construct.
func TestRealTemplatesRejectNothing(t *testing.T) {
	for name, src := range map[string]string{"qwen2": qwen2Template, "llama3": llama3Template} {
		if _, err := Parse(src); err != nil {
			t.Errorf("%s template failed to parse: %v", name, err)
		}
	}
}

// TestQwen2TemplateWithTools exercises the tools branch, which renders a JSON array and
// uses a nested loop.
func TestQwen2TemplateWithTools(t *testing.T) {
	msgs := []Message{{Role: "system", Content: "you can call tools"}, {Role: "user", Content: "go"}}
	got := render(t, qwen2Template, msgs, Inputs{AddGenerationPrompt: false, Tools: []Tool{
		{Name: "get", Description: "fetch", Parameters: `{}`},
	}})
	if !strings.Contains(got, "# Tools") {
		t.Errorf("prompt does not contain the tools heading: %q", got)
	}
	if !strings.Contains(got, `{"function"`) {
		t.Errorf("prompt does not contain the serialized tool: %q", got)
	}
}

// loadTestTemplate reads a chat template from testdata.
//
// The full Llama 3.2 template is 93 lines. Inlining it would make this file mostly template
// text, which is harder to review than the code around it and easy to lose a line from when
// editing. The file keeps its provenance in the name.
// loadTestTemplate reads a chat-template asset exactly as stored.
//
// The assets are byte copies of what a checkpoint holds in tokenizer.chat_template, and that
// exactness is the point: text outside a {% %} tag is rendered, so a byte the file gained on the
// way into the repository becomes part of the prompt. Nothing trims, normalises or appends here.
//
// Two places keep the files honest. .gitattributes marks them `-text` so a checkout does not
// rewrite line endings, and .pre-commit-config.yaml excludes them from end-of-file-fixer,
// trailing-whitespace and mixed-line-ending. Both were needed: restoring the gemma 2 asset to the
// checkpoint's 591 bytes and running `just hooks` rewrote it to 592.
func loadTestTemplate(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read testdata/%s: %v", name, err)
	}
	return string(raw)
}

// TestTemplateAssetsAreExact guards the convention above with a hash.
//
// A hook that starts touching these files again fails here rather than in a rendered prompt: the
// hash changes the moment a byte does, and the failure says which asset and why it matters. The
// hashes are of the bytes as committed, so updating an asset deliberately means updating its
// entry -- which is the moment to check the new content against the checkpoint.
//
// llama32-instruct.jinja ends with a newline that is part of the asset as it was first committed.
// Whether the checkpoint's own template ends that way is not verifiable here, unlike the gemma 2
// one, which was extracted from a GGUF file that is available locally. The entry records the bytes
// rather than a claim about their provenance.
func TestTemplateAssetsAreExact(t *testing.T) {
	want := map[string]struct {
		size int
		sha  string
	}{
		"gemma2-it.jinja":        {591, "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6"},
		"gemma3-it.jinja":        {1654, "af95fbef33b76a50e5f463ff9766f85ce84f5849d915c4bd6c1619d852ac3231"},
		"llama32-instruct.jinja": {3827, "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"},
		// Extracted byte-for-byte from /models/Qwen2.5-0.5B-Instruct/tokenizer_config.json
		// (tokenizer.chat_template), so unlike llama32 this one's provenance is checkable: the
		// source directory is on the machine the asset was taken from.
		"qwen2-instruct.jinja": {2507, "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"},
	}
	for name, w := range want {
		raw, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Errorf("read %s: %v", name, err)
			continue
		}
		if len(raw) != w.size {
			t.Errorf("%s is %d bytes, want the %d that were committed; a whitespace hook may have"+
				" rewritten it, and for a chat template that is a change to the prompt",
				name, len(raw), w.size)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != w.sha {
			t.Errorf("%s hashes to %s, want %s", name, got, w.sha)
		}
	}
	// The gemma 2 asset must not end with whitespace: the checkpoint's template ends with
	// "{% endif %}", and a newline after it would be rendered. This is the assertion that names
	// the specific mistake the hooks used to make.
	raw, err := os.ReadFile(filepath.Join("testdata", "gemma2-it.jinja"))
	if err == nil && len(raw) > 0 && raw[len(raw)-1] == '\n' {
		t.Error("gemma2-it.jinja ends with a newline; the checkpoint's template does not, and the" +
			" newline would appear in every rendered prompt")
	}
}

// TestLlama32TemplateWithTools covers the tool-calling branch of the full Llama 3.2 template,
// which the abbreviated llama3Template above does not have.
//
// This is the path the previous two rounds' fixes were for: before them the branch was taken
// even with no tools, because Render bound an empty tools list and the template's own
// "{% if not tools is defined %}" guard could never fire. With a tool supplied the branch is
// now taken for the right reason, and what it emits has to be JSON the model can act on.
func TestLlama32TemplateWithTools(t *testing.T) {
	tmpl := loadTestTemplate(t, "llama32-instruct.jinja")
	parsed, err := Parse(tmpl)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	const params = `{"type":"object","properties":{"city":{"type":"string"}}}`
	got, err := parsed.Render(
		[]Message{{Role: "user", Content: "What is the weather in Paris?"}},
		Inputs{
			BOSToken:            "<|begin_of_text|>",
			EOSToken:            "<|eot_id|>",
			AddGenerationPrompt: true,
			Tools: []Tool{{
				Name:        "get_weather",
				Description: "Current weather for a city",
				Parameters:  params,
			}},
		})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}

	// The tool list is JSON produced by the template's tojson(indent=4), so it is a
	// multi-line block. Finding and parsing it checks the shape rather than the presence of
	// a substring: a missing brace or a doubled key would still contain the tool's name.
	start := strings.Index(got, `{
    "function": {`)
	if start < 0 {
		t.Fatalf("no indented tool JSON block in the prompt:\n%s", got)
	}
	// The block ends at the closing brace that matches the opening one.
	depth, end := 0, -1
	for i := start; i < len(got); i++ {
		switch got[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				end = i + 1
			}
		}
		if end >= 0 {
			break
		}
	}
	if end < 0 {
		t.Fatalf("the tool JSON block is not closed:\n%s", got[start:])
	}
	var v struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal([]byte(got[start:end]), &v); err != nil {
		t.Fatalf("the tool block is not valid JSON: %v\n%s", err, got[start:end])
	}
	if v.Type != "function" || v.Function.Name != "get_weather" {
		t.Errorf("tool JSON = %+v, want the function get_weather", v)
	}
	// parameters is passed through verbatim, so it must still be the caller's document
	// and not a re-encoding that dropped a field.
	if string(v.Function.Parameters) != params {
		t.Errorf("parameters = %s, want %s", v.Function.Parameters, params)
	}

	// The no-tools guidance must not appear when tools were supplied.
	if strings.Contains(got, "Cutting Knowledge Date") && strings.Contains(got, "No tools") {
		t.Errorf("prompt mixes tool and no-tool guidance:\n%s", got)
	}
}

// TestLlama32TemplateWithoutTools is the other side: the same template and no tools must not
// produce the tool-calling instructions.
func TestLlama32TemplateWithoutTools(t *testing.T) {
	tmpl := loadTestTemplate(t, "llama32-instruct.jinja")
	parsed, err := Parse(tmpl)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := parsed.Render(
		[]Message{{Role: "user", Content: "What is the capital of France?"}},
		Inputs{BOSToken: "<|begin_of_text|>", EOSToken: "<|eot_id|>", AddGenerationPrompt: true})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	for _, unwanted := range []string{"ipython", "respond with a JSON", `"function"`} {
		if strings.Contains(got, unwanted) {
			t.Errorf("a no-tools prompt contains %q:\n%s", unwanted, got)
		}
	}
	if !strings.Contains(got, "What is the capital of France?") {
		t.Errorf("the user's message is missing:\n%s", got)
	}
}

// TestLlama32ToolMessageContentIsNotQuoted covers the branch the tools test above never reaches.
//
// The template guards a tool message with `message.content is mapping or message.content is
// iterable` before choosing between {{ content | tojson }} and {{ content }}. message.content
// is always a string, and this package treats only sequences as iterable, so the guard is false
// where Jinja2 would make it true -- which is the difference doc.go records under the is-tests.
// Either way the content is printed as it stands rather than as a quoted JSON string, and that
// is what a tool result carrying JSON needs: the model has to read the object, not a string
// holding one.
//
// Nothing tested this before: TestLlama32TemplateWithTools submits a user turn only, so the
// tool-message branch was parsed but never rendered.
func TestLlama32ToolMessageContentIsNotQuoted(t *testing.T) {
	parsed, err := Parse(loadTestTemplate(t, "llama32-instruct.jinja"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	// A tool turn whose content is JSON: the shape the branch exists for.
	const payload = `{"temp_c": 21}`
	got, err := parsed.Render(
		[]Message{
			{Role: "user", Content: "What is the weather in Paris?"},
			{Role: "tool", Content: payload},
		},
		Inputs{BOSToken: "<|begin_of_text|>", EOSToken: "<|eot_id|>"},
	)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	// The payload has to appear verbatim. A quoted rendering would show \"temp_c\".
	if !strings.Contains(got, payload) {
		t.Errorf("the tool payload was not emitted as it stands; got:\n%s", got)
	}
	if strings.Contains(got, `\"temp_c\"`) {
		t.Errorf("the tool payload was JSON-encoded into a string; got:\n%s", got)
	}
	// The ipython header is that branch's marker, so the run actually took it.
	if !strings.Contains(got, "ipython") {
		t.Errorf("the tool branch was not taken; got:\n%s", got)
	}
}

// TestGemma2Template covers Gemma 2's own template, which is the reason
// the modulo operator exists.
//
// The alternation guard is "(message['role'] == 'user') != (loop.index0 % 2 == 0)", so without
// '%' the whole template fails to compile and the server falls back to a neutral transcript --
// visible as a startup warning, but for an instruct model it means the turns are wrapped in
// markers the checkpoint was not trained on.
func TestGemma2Template(t *testing.T) {
	tmpl := loadTestTemplate(t, "gemma2-it.jinja")
	parsed, err := Parse(tmpl)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	// A well-formed alternating conversation renders the turn markers the checkpoint expects.
	got, err := parsed.Render(
		[]Message{
			{Role: "user", Content: "What is the capital of France?"},
			{Role: "assistant", Content: "Paris."},
			{Role: "user", Content: "And of Italy?"},
		},
		Inputs{BOSToken: "<bos>", AddGenerationPrompt: true})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	want := "<bos><start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n" +
		"<start_of_turn>model\nParis.<end_of_turn>\n" +
		"<start_of_turn>user\nAnd of Italy?<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if got != want {
		t.Errorf("Render =\n%q\nwant\n%q", got, want)
	}

	// A conversation that does not alternate reaches the template's own raise_exception, which
	// is what proves the modulo comparison actually compared: with '%' reading as something
	// else the guard would pass everything and this would render instead of failing.
	_, err = parsed.Render(
		[]Message{
			{Role: "user", Content: "one"},
			{Role: "user", Content: "two"},
		},
		Inputs{BOSToken: "<bos>", AddGenerationPrompt: true})
	if err == nil {
		t.Fatal("two user turns in a row rendered; the alternation guard did not fire")
	}
	// The template's own sentence, not the name of the function it called with. An operator
	// reading a fallback needs to know the conversation was rejected and why; "call to
	// raise_exception: no function is implemented" said neither.
	if !strings.Contains(err.Error(), "roles must alternate") {
		t.Errorf("error = %v, want it to carry the template's own reason", err)
	}
}

// TestGemma3TemplateRendersTurnMarkers covers the third checkpoint template this package carries.
//
// The template is the reason three constructs exist: it asks "is string" to tell a plain content
// string from a list of multimodal parts, writes "x if c else y" to emit its system prefix only on
// the first turn, and needs a parenthesised conditional to concatenate the result. Each was refused
// before, so the whole template fell back to the neutral transcript and every Gemma 3 conversation
// lost the <start_of_turn> markers the model was trained on -- visible as "using a neutral
// transcript" at startup and as worse answers, not as an error.
//
// The expectations come from the template's own text rather than from a reference implementation:
// Gemma 3's format is literal markers around the turn, so what the render produces can be read
// against the template directly.
func TestGemma3TemplateRendersTurnMarkers(t *testing.T) {
	tmpl, err := Parse(loadTestTemplate(t, "gemma3-it.jinja"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	const addGen = true

	got, rerr := tmpl.Render([]Message{{Role: "user", Content: "hello"}},
		Inputs{BOSToken: "<bos>", AddGenerationPrompt: addGen})
	if rerr != nil {
		t.Fatalf("Render: %v", rerr)
	}
	// The leading newline is the asset's: the template's first line is {{ bos_token }} and the
	// line before it is blank in the checkpoint. Trimming it would emit a prompt the model was
	// not trained on, and it is exactly the kind of byte this package's assets are pinned for.
	want := "\n<bos><start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"
	if got != want {
		t.Errorf("Render =\n%q\nwant\n%q", got, want)
	}

	// A system turn becomes the prefix on the first user turn, which is the branch the inline
	// conditional guards: emitting it every turn would repeat the system prompt in the history.
	got, err = tmpl.Render([]Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "more"},
	}, Inputs{BOSToken: "<bos>", AddGenerationPrompt: addGen})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	want = "\n<bos><start_of_turn>user\nbe brief\n\nhi<end_of_turn>\n" +
		"<start_of_turn>model\nok<end_of_turn>\n" +
		"<start_of_turn>user\nmore<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if got != want {
		t.Errorf("Render =\n%q\nwant\n%q", got, want)
	}
	if strings.Count(got, "be brief") != 1 {
		t.Errorf("the system prompt appears %d times; it belongs on the first user turn only",
			strings.Count(got, "be brief"))
	}

	// A non-alternating history reaches the template's raise_exception, and the reason it gives is
	// the template's own sentence.
	_, err = tmpl.Render([]Message{
		{Role: "user", Content: "one"},
		{Role: "user", Content: "two"},
	}, Inputs{BOSToken: "<bos>"})
	if err == nil {
		t.Fatal("a history with two user turns rendered instead of raising")
	}
	if !strings.Contains(err.Error(), "roles must alternate") {
		t.Errorf("error = %v, want the template's own reason", err)
	}
}

// TestQwen2ToolConversationMatchesJinja2 renders a whole tool-calling exchange through the real
// Qwen2.5 template and compares against what Jinja2 produced for the same conversation.
//
// The expected string is not hand-written from reading the template. It is Jinja2 3.1.6's own
// output for this exact input, captured with
//
//	uv run --with jinja2 python3 -c "…open('/models/Qwen2.5-0.5B-Instruct/tokenizer_config.json')…"
//
// and pasted here, because the point of the case is the tool round trip and a hand-transcribed
// expectation would only restate what this package already does. Anything that drifts -- the
// arguments JSON's spacing, the tool_response wrapper, the blank line before the generation
// prompt -- changes the prompt the model is trained to read.
//
// It reaches three constructs the earlier tool tests did not: iterating a message's tool_calls,
// `tool_call.function is defined` on a nested mapping, and `{% set tool_call = tool_call.function %}`
// rebinding a loop variable. The last one is the reason the case exists: this package's `set`
// writes to the outermost frame, where Jinja2's writes to the enclosing block, and the two
// readings disagree on exactly this shape.
func TestQwen2ToolConversationMatchesJinja2(t *testing.T) {
	tmpl, err := Parse(loadTestTemplate(t, "qwen2-instruct.jinja"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tmpl.Render(
		[]Message{
			{Role: "user", Content: "What is the weather in Paris?"},
			{Role: "assistant", ToolCalls: []ToolCall{{
				ID:        "call_1",
				Name:      "get_weather",
				Arguments: `{"city": "Paris"}`,
			}}},
			{Role: "tool", Content: `{"temp_c": 21}`},
		},
		Inputs{
			AddGenerationPrompt: true,
			Tools: []Tool{{
				Name:        "get_weather",
				Description: "Current weather for a city",
				// Sorted keys on purpose: tojson sorts them, and this package emits a
				// Parameters document verbatim, so the two agree only when the caller's key
				// order is already Python's. See the note in the test body.
				Parameters: `{"properties": {"city": {"type": "string"}}, "type": "object"}`,
			}},
		})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	want := "<|im_start|>system\n" +
		"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n" +
		"\n" +
		"# Tools\n" +
		"\n" +
		"You may call one or more functions to assist with the user query.\n" +
		"\n" +
		"You are provided with function signatures within <tools></tools> XML tags:\n" +
		"<tools>\n" +
		"{\"function\": {\"description\": \"Current weather for a city\", \"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"type\": \"string\"}}, \"type\": \"object\"}}, \"type\": \"function\"}\n" +
		"</tools>\n" +
		"\n" +
		"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n" +
		"<tool_call>\n" +
		"{\"name\": <function-name>, \"arguments\": <args-json-object>}\n" +
		"</tool_call><|im_end|>\n" +
		"<|im_start|>user\n" +
		"What is the weather in Paris?<|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"<tool_call>\n" +
		"{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n" +
		"</tool_call><|im_end|>\n" +
		"<|im_start|>user\n" +
		"<tool_response>\n" +
		"{\"temp_c\": 21}\n" +
		"</tool_response><|im_end|>\n" +
		"<|im_start|>assistant\n" +
		"\n"
	if got != want {
		t.Errorf("Render =\n%q\nwant (Jinja2's own output)\n%q", got, want)
	}
}

// TestQwen2TemplateRangesOverAdjacentToolTurns covers the construct that put integer addition
// into this package's + operator.
//
// The template's tool branch reads `messages[loop.index0 + 1].role` to decide whether the next
// turn is another tool result, which needs two things at once: a numeric +, and a subscript that
// is computed from the loop index rather than from a literal. Under the old concatenating +
// the index became the string "01", the subscript missed, and the whole render failed -- which
// the renderer turned into the neutral transcript, so a multi-turn tool conversation silently
// lost the model's own format while the banner still reported "model template".
//
// The expected string is Jinja2's own output for this conversation, captured the same way as the
// test above. Two adjacent tool turns are what exercises the guarded read: the first one sees a
// following tool turn and stays in the same <|im_start|>user block, the second closes it.
func TestQwen2TemplateRangesOverAdjacentToolTurns(t *testing.T) {
	tmpl, err := Parse(loadTestTemplate(t, "qwen2-instruct.jinja"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tmpl.Render(
		[]Message{
			{Role: "user", Content: "weather in Paris and Lyon?"},
			{Role: "assistant", ToolCalls: []ToolCall{
				{ID: "call_1", Name: "get_weather", Arguments: `{"city": "Paris"}`},
				{ID: "call_2", Name: "get_weather", Arguments: `{"city": "Lyon"}`},
			}},
			{Role: "tool", Content: `{"temp_c": 21}`},
			{Role: "tool", Content: `{"temp_c": 18}`},
			{Role: "user", Content: "compare them"},
		},
		Inputs{
			AddGenerationPrompt: true,
			Tools: []Tool{{
				Name:        "get_weather",
				Description: "Current weather for a city",
				Parameters:  `{"properties": {"city": {"type": "string"}}, "type": "object"}`,
			}},
		})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	// The two results share one user turn: the first <|im_start|>user opens it and the second
	// tool turn does not open another. That is the assertion the guarded read exists for, so it
	// is stated directly rather than left implicit in a long string comparison.
	if n := strings.Count(got, "<|im_start|>user\n<tool_response>"); n != 1 {
		t.Errorf("%d user blocks open a tool response, want 1: adjacent tool results belong to "+
			"one turn\n%s", n, got)
	}
	if n := strings.Count(got, "</tool_response>"); n != 2 {
		t.Errorf("%d tool responses in the prompt, want 2\n%s", n, got)
	}
	// And both calls were rendered, with their arguments.
	for _, city := range []string{"Paris", "Lyon"} {
		if !strings.Contains(got, `"city": "`+city+`"`) {
			t.Errorf("the prompt does not carry the %s call:\n%s", city, got)
		}
	}
}
