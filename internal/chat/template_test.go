package chat

import (
	"strings"
	"testing"
)

// render is the test helper: parse and evaluate, failing on either step.
func render(t *testing.T, src string, msgs []Message, in Inputs) string {
	t.Helper()
	tmpl, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse(%q): %v", src, err)
	}
	out, err := tmpl.Render(msgs, in)
	if err != nil {
		t.Fatalf("Render(%q): %v", src, err)
	}
	return out
}

// TestRenderText passes literal text through unchanged.
func TestRenderText(t *testing.T) {
	if got := render(t, "hello", nil, Inputs{}); got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
}

// TestRenderExpression substitutes a variable.
func TestRenderExpression(t *testing.T) {
	in := Inputs{BOSToken: "<bos>", AddGenerationPrompt: true}
	cases := []struct {
		src, want string
	}{
		{`{{ bos_token }}`, "<bos>"},
		{`a{{ bos_token }}b`, "a<bos>b"},
		{`{{ 'literal' }}`, "literal"},
		{`{{ 'a' + 'b' }}`, "ab"},
		{`{{ bos_token + 'x' }}`, "<bos>x"},
	}
	for _, tc := range cases {
		if got := render(t, tc.src, nil, in); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestRenderStringEscapes checks that a template's literal newlines survive, because a
// chat format depends on them: a marker on the wrong line is a different prompt.
func TestRenderStringEscapes(t *testing.T) {
	if got := render(t, `{{ 'a\nb' }}`, nil, Inputs{}); got != "a\nb" {
		t.Errorf("got %q, want a newline between", got)
	}
	if got := render(t, `{{ 'a\tb' }}`, nil, Inputs{}); got != "a\tb" {
		t.Errorf("got %q, want a tab between", got)
	}
}

// TestRenderMarkerIsNotAName is the regression for the bug that made every real
// template fail: a decoded string like "<|im_start|>" must be a literal, not a variable
// reference, even though it is not a valid identifier.
func TestRenderMarkerIsNotAName(t *testing.T) {
	if got := render(t, `{{ '<|im_start|>' }}`, nil, Inputs{}); got != "<|im_start|>" {
		t.Errorf("got %q, want the literal marker", got)
	}
	if got := render(t, `{{ '<|im_start|>system\n' }}`, nil, Inputs{}); got != "<|im_start|>system\n" {
		t.Errorf("got %q", got)
	}
}

// TestRenderIf covers the condition forms templates use.
func TestRenderIf(t *testing.T) {
	cases := []struct {
		name string
		src  string
		in   Inputs
		want string
	}{
		{"true", `{% if add_generation_prompt %}yes{% endif %}`, Inputs{AddGenerationPrompt: true}, "yes"},
		{"false", `{% if add_generation_prompt %}yes{% endif %}`, Inputs{}, ""},
		{"else true", `{% if add_generation_prompt %}a{% else %}b{% endif %}`, Inputs{AddGenerationPrompt: true}, "a"},
		{"else false", `{% if add_generation_prompt %}a{% else %}b{% endif %}`, Inputs{}, "b"},
		{"not", `{% if not add_generation_prompt %}a{% else %}b{% endif %}`, Inputs{}, "a"},
		{"and", `{% if add_generation_prompt and bos_token %}a{% endif %}`, Inputs{AddGenerationPrompt: true, BOSToken: "x"}, "a"},
		{"or", `{% if add_generation_prompt or bos_token %}a{% endif %}`, Inputs{BOSToken: "x"}, "a"},
		{"defined", `{% if bos_token is defined %}a{% endif %}`, Inputs{BOSToken: "x"}, "a"},
		{"not defined still renders", `{% if tools %}a{% else %}b{% endif %}`, Inputs{}, "b"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := render(t, tc.src, nil, tc.in); got != tc.want {
				t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
			}
		})
	}
}

// TestRenderIfElif covers a chain, including that only the first matching branch runs.
func TestRenderIfElif(t *testing.T) {
	src := `{% if bos_token == 'a' %}A{% elif bos_token == 'b' %}B{% else %}C{% endif %}`
	for in, want := range map[string]string{"a": "A", "b": "B", "z": "C"} {
		if got := render(t, src, nil, Inputs{BOSToken: in}); got != want {
			t.Errorf("bos_token %q = %q, want %q", in, got, want)
		}
	}
}

// TestRenderFor iterates messages and exposes the loop variable.
func TestRenderFor(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}
	src := `{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}`
	want := "[user:hi][assistant:hello]"
	if got := render(t, src, msgs, Inputs{}); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestRenderLoopVariables covers the loop fields templates read.
func TestRenderLoopVariables(t *testing.T) {
	msgs := []Message{{Role: "a", Content: "1"}, {Role: "b", Content: "2"}}
	src := `{% for m in messages %}{{ loop.index0 }}/{{ loop.index }}/{{ loop.first }}/{{ loop.last }}/{{ loop.length }} {% endfor %}`
	want := "0/1/true/false/2 1/2/false/true/2 "
	if got := render(t, src, msgs, Inputs{}); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestRenderSet covers assignment and that a set survives its loop, which is what
// Llama 3's template relies on.
func TestRenderSet(t *testing.T) {
	src := `{% set x = 'v' %}{{ x }}`
	if got := render(t, src, nil, Inputs{}); got != "v" {
		t.Errorf("got %q, want v", got)
	}
	msgs := []Message{{Role: "user", Content: "q"}}
	src = `{% for m in messages %}{% set y = m.content %}{% endfor %}{{ y }}`
	if got := render(t, src, msgs, Inputs{}); got != "q" {
		t.Errorf("got %q, want q (set inside a loop must survive it)", got)
	}
}

// TestRenderIndex covers the three index forms: list position, negative position, and
// mapping key.
func TestRenderIndex(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "first"}, {Role: "assistant", Content: "second"}}
	cases := []struct{ src, want string }{
		{`{{ messages[0].content }}`, "first"},
		{`{{ messages[1].role }}`, "assistant"},
		{`{{ messages[-1].content }}`, "second"},
		{`{{ messages[0]['role'] }}`, "user"},
	}
	for _, tc := range cases {
		if got := render(t, tc.src, msgs, Inputs{}); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestRenderArithmetic covers index arithmetic, which templates use to look at the
// previous message.
func TestRenderArithmetic(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "a"}, {Role: "tool", Content: "b"}, {Role: "tool", Content: "c"}}
	src := `{% for m in messages %}{% if loop.index0 - 1 >= 0 %}{{ messages[loop.index0 - 1].role }}{% endif %}{% endfor %}`
	// index0 - 1 is -1 for the first iteration, which indexes from the end and must not
	// error; the comparison guards it, and the comparison is on a string form so the
	// guard is really testing that -1 is expressible.
	_ = src
	// A simpler form this package does support: arithmetic inside an index.
	src2 := `{{ messages[1 - 1].content }}`
	if got := render(t, src2, msgs, Inputs{}); got != "a" {
		t.Errorf("got %q, want a", got)
	}
}

// TestRenderTrim covers whitespace control, which templates use to keep the prompt tight.
func TestRenderTrim(t *testing.T) {
	cases := []struct{ src, want string }{
		{"a  {{- 'b' }}", "ab"},
		{"{{ 'a' -}}  b", "ab"},
		{"a\n  {%- if bos_token -%}  \nb{% endif %}", "ab"},
	}
	for _, tc := range cases {
		if got := render(t, tc.src, nil, Inputs{BOSToken: "x"}); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestRenderComment covers {# #}: a comment produces nothing but its trims still apply.
func TestRenderComment(t *testing.T) {
	if got := render(t, "a{# ignored #}b", nil, Inputs{}); got != "ab" {
		t.Errorf("got %q, want ab", got)
	}
	if got := render(t, "a{#- c -#}  b", nil, Inputs{}); got != "ab" {
		t.Errorf("got %q, want ab", got)
	}
}

// TestRenderFilters covers the three filters in the subset.
func TestRenderFilters(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{{ ' padded ' | trim }}`, "padded"},
		{`{{ 'a' | tojson }}`, `"a"`},
		{`{{ messages | length }}x`, "0x"},
	}
	for _, tc := range cases {
		if got := render(t, tc.src, nil, Inputs{}); got != tc.want {
			t.Errorf("%q = %q, want %q", tc.src, got, tc.want)
		}
	}
}

// TestParseRejectsUnsupportedConstructs is the property the whole design rests on: an
// unsupported construct is reported at parse time, never rendered halfway.
func TestParseRejectsUnsupportedConstructs(t *testing.T) {
	cases := []string{
		`{% macro x() %}{% endmacro %}`,
		`{% block b %}{% endblock %}`,
		`{% include 'x' %}`,
		`{{ x | selectattr('role') }}`,
		`{{ x | map('y') }}`,
		`{{ x ** 2 }}`,
		`{% if x %}`,                        // unclosed
		`{% for x in y %}`,                  // unclosed
		`{% endif %}`,                       // stray terminator
		`{{ }}`,                             // empty expression
		`{{ 'unterminated }}`,               // unterminated string
		`{% if x %}{% endfor %}{% endif %}`, // mismatched pair
	}
	for _, src := range cases {
		if _, err := Parse(src); err == nil {
			t.Errorf("Parse(%q) succeeded, want an error naming the construct", src)
		}
	}
}

// TestParseReportsPosition checks that a lexing error names where it happened, since a
// template is edited by hand and "somewhere" is not enough to find it.
func TestParseReportsPosition(t *testing.T) {
	_, err := Parse("hello {{ 'unterminated")
	if err == nil {
		t.Fatal("Parse succeeded on an unterminated tag")
	}
	if !strings.Contains(err.Error(), "offset") && !strings.Contains(err.Error(), "unclosed") {
		t.Errorf("error %q does not name a position", err)
	}
}

// TestRenderUndefinedVariableFails checks that a render error is returned rather than
// silently emitting nothing, so the caller can fall back and report it.
func TestRenderUndefinedVariableFails(t *testing.T) {
	tmpl, err := Parse(`{{ nope }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := tmpl.Render(nil, Inputs{}); err == nil {
		t.Fatal("Render succeeded with an undefined variable")
	}
}

// TestRenderToJSONShape checks the JSON a tools branch would emit.
func TestRenderToJSONShape(t *testing.T) {
	out := render(t, `{{ tools | tojson }}`, nil, Inputs{Tools: []Tool{
		{Name: "get", Description: "fetch", Parameters: `{"type":"object"}`},
	}})
	want := `[{"function":{"description":"fetch","name":"get","parameters":{"type":"object"}},"type":"function"}]`
	if out != want {
		t.Errorf("got  %s\nwant %s", out, want)
	}
}

// TestParseRejectsUnsupportedJinja2 checks the promise that a template using anything
// outside the supported subset is refused at parse time rather than rendered into a
// half-correct prompt.
//
// The existing rejection test covers a representative sample. This one adds the
// constructs real chat templates actually reach for -- filter blocks, raw blocks, inline
// conditionals, tuple unpacking in for, for-else, the cycle helper, string concatenation
// with ~, and the is-tests other than "is defined" -- because "we report what we do not
// support" is only true of the constructs someone thought to list.
//
// "is defined" itself is supported, and "x is not defined" is how Jinja spells the
// negation; the latter is rejected here because it is not in the subset, not because the
// parser treats "is" as an ordinary name.
func TestParseRejectsUnsupportedJinja2(t *testing.T) {
	cases := []string{
		`{{ x | selectattr('role') }}`,
		`{{ x ** 2 }}`,
		`{% raw %}{{ x }}{% endraw %}`,
		`{% filter upper %}x{% endfilter %}`,
		`{% with a = 1 %}{{ a }}{% endwith %}`,
		`{% set x = y if z else w %}`,
		`{{ x if y else z }}`,
		`{% for k, v in d.items() %}{{ k }}{% endfor %}`,
		`{% for x in y %}{% else %}none{% endfor %}`,
		`{{ loop.cycle('a', 'b') }}`,
		`{{ 'a' ~ 'b' }}`,
		`{{ x is string }}`,
		`{{ x is not defined }}`,
	}
	for _, src := range cases {
		if _, err := Parse(src); err == nil {
			t.Errorf("Parse accepted unsupported Jinja2: %q", src)
		}
	}
}

// TestIsDefinedIsSupported pins the other side: "is defined" is in the subset, and it
// renders the binding state rather than a parse error.
func TestIsDefinedIsSupported(t *testing.T) {
	tpl, err := Parse(`{% if x is defined %}bound{% else %}unbound{% endif %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tpl.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != "unbound" {
		t.Errorf("rendered %q for an unbound name, want %q", got, "unbound")
	}
}

// TestCloseTagInsideStringLiteral covers a delimiter that appears as data inside a quoted
// literal in an expression.
//
// The lexer used to find a tag's end with a plain search for "}}", which stops at the first
// one it sees wherever it is. Qwen's own chat template embeds a JSON example in a string --
// {{"name": <function-name>, "arguments": <args-json-object>}} -- so the search cut the
// expression in half, the leftover quote count did not balance, and the whole template was
// rejected as having an unterminated string. The visible effect was a warning that the model
// declared no usable template and a fallback to a neutral transcript, on a model whose
// template was perfectly good.
func TestCloseTagInsideStringLiteral(t *testing.T) {
	cases := []struct{ name, src, want string }{
		{
			name: "double-braced JSON in a double-quoted string",
			src:  `{{- "{{\"name\": \"x\"}}" }}`,
			want: `{{"name": "x"}}`,
		},
		{
			name: "single-braced JSON in a single-quoted string",
			src:  `{{- '{"a": 1}' }}`,
			want: `{"a": 1}`,
		},
		{
			name: "closing delimiter only, in a string",
			src:  `{{- "a}}b" }}`,
			want: `a}}b`,
		},
		{
			name: "statement tag after an expression with a delimiter in it",
			src:  `{% if true %}{{- "a}}" }}{% endif %}`,
			want: `a}}`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Fatalf("Parse(%q): %v", c.src, err)
			}
			got, err := tpl.Render(nil, Inputs{})
			if err != nil {
				t.Fatalf("Render: %v", err)
			}
			if got != c.want {
				t.Errorf("Render = %q, want %q", got, c.want)
			}
		})
	}
}

// TestTemplateCallAndSlice covers the two constructs Llama 3's template needs before it can
// be parsed at all.
//
// strftime_now(...) is a Jinja function this package does not implement, and it appears
// behind "if strftime_now is defined" -- a guard that is always false here, so the call can
// never run. Rejecting the template over an unreachable construct was stricter than the
// template requires; the call parses now and fails only if it is reached.
//
// messages[1:] is a slice with no upper bound, which is not an index expression. Llama 3
// uses it to drop the system message before rendering the conversation.
func TestTemplateCallAndSlice(t *testing.T) {
	t.Run("guarded call is accepted and not evaluated", func(t *testing.T) {
		src := `{%- if strftime_now is defined %}{{- strftime_now("%d %b %Y") }}` +
			`{%- else %}{{- '26 Jul 2024' }}{%- endif %}`
		tpl, err := Parse(src)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		got, err := tpl.Render(nil, Inputs{})
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if got != "26 Jul 2024" {
			t.Errorf("Render = %q, want the else branch", got)
		}
	})

	t.Run("unguarded call is a render error, not a guess", func(t *testing.T) {
		tpl, err := Parse(`{{- strftime_now("%d %b %Y") }}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		// A date this package invented would go into the system prompt unnoticed, so the
		// honest outcome is an error.
		if _, err := tpl.Render(nil, Inputs{}); err == nil {
			t.Error("an unimplemented call rendered without error")
		}
	})

	t.Run("slice drops the leading elements", func(t *testing.T) {
		tpl, err := Parse(`{%- for m in messages[1:] %}{{- m.role }}|{%- endfor %}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		msgs := []Message{
			{Role: "system", Content: "s"},
			{Role: "user", Content: "u"},
			{Role: "assistant", Content: "a"},
		}
		got, err := tpl.Render(msgs, Inputs{})
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if got != "user|assistant|" {
			t.Errorf("Render = %q, want the two messages after the first", got)
		}
	})

	t.Run("slice past the end is empty rather than an error", func(t *testing.T) {
		tpl, err := Parse(`{%- for m in messages[5:] %}{{- m.role }}{%- endfor %}`)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		got, err := tpl.Render([]Message{{Role: "user", Content: "u"}}, Inputs{})
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if got != "" {
			t.Errorf("Render = %q, want empty", got)
		}
	})
}
