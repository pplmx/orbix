package chat

import (
	"encoding/json"
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
func loadTestTemplate(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read testdata/%s: %v", name, err)
	}
	return string(raw)
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

	// The tool list is a line of JSON, which is what the template's tojson produces. Finding
	// and parsing it checks the shape rather than the presence of a substring: a missing
	// brace or a doubled key would still contain the tool's name.
	var found bool
	for _, line := range strings.Split(got, "\n") {
		if !strings.HasPrefix(line, `{"function":`) {
			continue
		}
		found = true
		var v struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		}
		if err := json.Unmarshal([]byte(line), &v); err != nil {
			t.Fatalf("the tool line is not valid JSON: %v\n%s", err, line)
		}
		if v.Type != "function" || v.Function.Name != "get_weather" {
			t.Errorf("tool JSON = %+v, want the function get_weather", v)
		}
		// parameters is passed through verbatim, so it must still be the caller's document
		// and not a re-encoding that dropped a field.
		if string(v.Function.Parameters) != params {
			t.Errorf("parameters = %s, want %s", v.Function.Parameters, params)
		}
	}
	if !found {
		t.Errorf("no tool JSON line in the prompt:\n%s", got)
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
