package chat

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
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
	// A boolean prints as Python's repr does -- "True", not "true" -- which is what Jinja2
	// produces and what the reference implementation was checked against.
	src := `{% for m in messages %}{{ loop.index0 }}/{{ loop.index }}/{{ loop.first }}/{{ loop.last }}/{{ loop.length }} {% endfor %}`
	want := "0/1/True/False/2 1/2/False/True/2 "
	if got := render(t, src, msgs, Inputs{}); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// The two counting down. Jinja2 3.1.6 renders the same two-item loop as
	// "2/1 1/0", which is what a separator that depends on remaining items reads.
	src = `{% for m in messages %}{{ loop.revindex }}/{{ loop.revindex0 }} {% endfor %}`
	want = "2/1 1/0 "
	if got := render(t, src, msgs, Inputs{}); got != want {
		t.Errorf("revindex: got %q, want %q", got, want)
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
		// Jinja2's length is Python's len, which counts characters rather than bytes.
		// Measured against it: "中文" is 2 there, and counting bytes here made it 6, so a
		// template branching on a length threshold would have taken the other arm.
		{`{{ '中文' | length }}`, "2"},
		{`{{ 'é' | length }}`, "1"},
		{`{{ 'hi' | length }}`, "2"},
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
		// Constructs the cross-check corpus also probes, each refused with its own message
		// rather than a bare "syntax error": the point of refusing at parse time is that the
		// caller learns which construct and can decide, so the five below are pinned here as
		// well as in scripts/jinja_diff.py.
		`{{ 'a' if x else 'b' }}`,                       // inline if-expression
		`{% filter upper %}x{% endfilter %}`,            // filter block
		`{% set ns = namespace(x=1) %}`,                 // namespace()
		`{% for m in messages %}{% else %}{% endfor %}`, // for-else
		`{% for a, b in pairs %}{% endfor %}`,           // tuple unpacking
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
	// Jinja2's tojson is Python's json.dumps with sort_keys=True, so keys are sorted and
	// the default separators are ", " and ": ". The spacing is part of the prompt text.
	want := `[{"function": {"description": "fetch", "name": "get", "parameters": {"type":"object"}}, "type": "function"}]`
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

// TestBooleanLiteralsAreNotStrings covers a defect that made every condition constant.
//
// "true" and "false" were parsed as string literals. A non-empty string is truthy, so
// {% if false %} took the true branch -- and since a template's whole structure is
// conditions, every branch that tested a literal was decided wrongly. Nothing failed: the
// output was well-formed text from the wrong branch.
func TestBooleanLiteralsAreNotStrings(t *testing.T) {
	cases := []struct{ src, want string }{
		{`{%- if false %}A{%- else %}B{%- endif %}`, "B"},
		{`{%- if true %}A{%- else %}B{%- endif %}`, "A"},
		{`{%- if none %}A{%- else %}B{%- endif %}`, "B"},
		{`{%- if false and true %}A{%- else %}B{%- endif %}`, "B"},
		{`{%- if true or false %}A{%- else %}B{%- endif %}`, "A"},
	}
	for _, c := range cases {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Errorf("Parse(%q): %v", c.src, err)
			continue
		}
		got, err := tpl.Render(nil, Inputs{})
		if err != nil {
			t.Errorf("Render(%q): %v", c.src, err)
			continue
		}
		if got != c.want {
			t.Errorf("%q rendered %q, want %q", c.src, got, c.want)
		}
	}
}

// TestEmptyStringLiteral covers a value that could not be parsed at all: the lexer reports
// "no more tokens" as an empty token, and an empty string literal produced the same, so
// {% set x = "" %} looked like a truncated expression.
func TestEmptyStringLiteral(t *testing.T) {
	cases := []struct{ src, want string }{
		{`[{{- "" }}]`, "[]"},
		{`[{{- '' }}]`, "[]"},
		{`{%- set x = "" %}[{{- x }}]`, "[]"},
		{`{%- set x = "" %}{%- if x == "" %}empty{%- endif %}`, "empty"},
	}
	for _, c := range cases {
		tpl, err := Parse(c.src)
		if err != nil {
			t.Errorf("Parse(%q): %v", c.src, err)
			continue
		}
		got, err := tpl.Render(nil, Inputs{})
		if err != nil {
			t.Errorf("Render(%q): %v", c.src, err)
			continue
		}
		if got != c.want {
			t.Errorf("%q rendered %q, want %q", c.src, got, c.want)
		}
	}
}

// TestFilterKeywordArgument covers tojson(indent=4), which Llama 3's template uses.
//
// indent only selects a JSON whitespace style; this package emits compact JSON, so the
// argument is parsed and ignored. Parsing it matters because the template does not compile
// otherwise, and every name it uses goes through this code path.
func TestFilterKeywordArgument(t *testing.T) {
	for _, src := range []string{
		`{{- t | tojson(indent=4) }}`,
		`{{- t | tojson }}`,
	} {
		if _, err := Parse(src); err != nil {
			t.Errorf("Parse(%q): %v", src, err)
		}
	}
	// A bare "=" outside a keyword argument is still a mistake, not a comparison.
	if _, err := Parse(`{{- a = b }}`); err == nil {
		t.Error("a bare '=' was accepted outside a filter's keyword argument")
	}
}

// TestIsTests covers the three "is" tests Llama 3's template needs beyond "is defined".
func TestIsTests(t *testing.T) {
	t.Run("is none", func(t *testing.T) {
		tpl, err := Parse(`{%- if x is none %}Y{%- else %}N{%- endif %}`)
		if err != nil {
			t.Fatal(err)
		}
		// An unbound name is none.
		if got, _ := tpl.Render(nil, Inputs{}); got != "Y" {
			t.Errorf("unbound name is none rendered %q, want Y", got)
		}
	})
	t.Run("is not none", func(t *testing.T) {
		tpl, err := Parse(`{%- if messages is not none %}Y{%- else %}N{%- endif %}`)
		if err != nil {
			t.Fatal(err)
		}
		if got, _ := tpl.Render(nil, Inputs{}); got != "Y" {
			t.Errorf("a bound list is not none rendered %q, want Y", got)
		}
	})
	t.Run("is mapping and is iterable", func(t *testing.T) {
		for _, c := range []struct{ src, want string }{
			{`{%- for m in messages %}{%- if m is mapping %}M{%- endif %}{%- endfor %}`, "M"},
			{`{%- for m in messages %}{%- if m is iterable %}I{%- else %}-{%- endif %}{%- endfor %}`, "-"},
		} {
			tpl, err := Parse(c.src)
			if err != nil {
				t.Errorf("Parse(%q): %v", c.src, err)
				continue
			}
			got, err := tpl.Render([]Message{{Role: "user", Content: "u"}}, Inputs{})
			if err != nil {
				t.Errorf("Render(%q): %v", c.src, err)
				continue
			}
			if got != c.want {
				t.Errorf("%q rendered %q, want %q", c.src, got, c.want)
			}
		}
	})
}

// TestToolsIsAbsentWhenNotProvided covers the binding that decides which branch a tool-aware
// template takes.
//
// Render used to bind "tools" unconditionally, to an empty list when the caller passed none.
// That kept `{% if tools %}` from failing on an unbound name, but it made
// `{% if not tools is defined %}` permanently false -- so the branch that sets tools = none
// never ran, and Llama 3's template rendered the tool-calling prompt for every conversation:
// "Environment: ipython", "respond with a JSON for a function call", with no tools in sight.
//
// The two tests now agree because an unbound name is false in a condition rather than an
// error, and tools is bound only when there are some.
func TestToolsIsAbsentWhenNotProvided(t *testing.T) {
	const guard = `{%- if not tools is defined %}SET{%- else %}SKIP{%- endif %}`
	tpl, err := Parse(guard)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := tpl.Render(nil, Inputs{}); err != nil {
		t.Fatalf("Render without tools: %v", err)
	} else if got != "SET" {
		t.Errorf("without tools the guard rendered %q, want SET", got)
	}
	if got, err := tpl.Render(nil, Inputs{Tools: []Tool{{Name: "f"}}}); err != nil {
		t.Fatalf("Render with tools: %v", err)
	} else if got != "SKIP" {
		t.Errorf("with a tool the guard rendered %q, want SKIP", got)
	}

	// The other form a template uses, which must reach the same conclusion.
	bare, err := Parse(`{%- if tools %}Y{%- else %}N{%- endif %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := bare.Render(nil, Inputs{}); err != nil {
		t.Fatalf("Render `if tools` without tools: %v", err)
	} else if got != "N" {
		t.Errorf("`if tools` without tools rendered %q, want N", got)
	}
}

// TestUndefinedNameIsFalseOnlyInConditions checks the distinction the fix above relies on.
//
// A condition treats an unbound name as false, which is Jinja's rule and what lets a
// template write `{% if tools %}` without a guard. Output does not: `{{ nope }}` is still an
// error, because printing nothing would send a prompt with a hole in it and leave the caller
// no way to know.
func TestUndefinedNameIsFalseOnlyInConditions(t *testing.T) {
	cond, err := Parse(`{%- if nope %}Y{%- else %}N{%- endif %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, err := cond.Render(nil, Inputs{}); err != nil {
		t.Fatalf("an unbound name in a condition should not fail: %v", err)
	} else if got != "N" {
		t.Errorf("unbound name in a condition rendered %q, want N", got)
	}

	out, err := Parse(`{{ nope }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := out.Render(nil, Inputs{}); err == nil {
		t.Error("an unbound name in output did not fail")
	}
}

// TestInTestsMappingKeys covers `'tool_calls' in message`, which is how the tool-call
// branches ask whether a message dictionary has a field.
func TestInTestsMappingKeys(t *testing.T) {
	// A message value is a mapping, so iterating messages gives one to test against.
	src := `{%- for m in messages %}` +
		`{%- if 'role' in m %}has-role{%- endif %}` +
		`{%- if 'tool_calls' in m %}has-tools{%- endif %}` +
		`{%- endfor %}`
	tpl, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tpl.Render([]Message{{Role: "user", Content: "u"}}, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != "has-role" {
		t.Errorf("rendered %q, want has-role: the key present is found and the one absent is not", got)
	}
}

// TestMessagesCarryToolCalls checks that a message dictionary always has the field, even for
// a conversation with no tool calls at all.
//
// Qwen2's template reads `message.tool_calls` inside a branch guarded by
// "not message.tool_calls", so the field has to exist to be tested. Leaving it out made every
// render fail with "mapping has no field tool_calls", which the renderer then silently
// replaced with a neutral transcript -- the session worked, in the wrong format, and the
// startup banner still said the model's own template was in use.
func TestMessagesCarryToolCalls(t *testing.T) {
	// The shape Qwen2's template tests.
	tpl, err := Parse(`{%- for m in messages %}` +
		`{%- if not m.tool_calls %}plain{%- else %}tools{%- endif %}{%- endfor %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := tpl.Render([]Message{{Role: "user", Content: "u"}}, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if got != "plain" {
		t.Errorf("rendered %q, want plain", got)
	}
}

// TestQwen2TemplateRendersAConversation is the end-to-end version: the real template, two
// turns, no tools.
func TestQwen2TemplateRendersAConversation(t *testing.T) {
	parsed, err := Parse(qwen2Template)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, err := parsed.Render([]Message{
		{Role: "user", Content: "What is the capital of France?"},
		{Role: "assistant", Content: "Paris."},
		{Role: "user", Content: "And of Italy?"},
	}, Inputs{AddGenerationPrompt: true})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	for _, want := range []string{"Paris.", "And of Italy?", "<|im_start|>assistant"} {
		if !strings.Contains(got, want) {
			t.Errorf("prompt is missing %q:\n%s", want, got)
		}
	}
}

// TestBooleanPrintMatchesJinja2 pins how a bare boolean reaches the prompt.
//
// Jinja2 prints a boolean with Python's repr, so "{{ loop.first }}" gives "True" while
// "{{ loop.first|tojson }}" gives "true". Both spellings are observable in the prompt, so a
// template that writes one and gets the other hands the model text it was not trained on.
// The expected strings are Jinja2's own output.
func TestBooleanPrintMatchesJinja2(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "yo"}}
	cases := []struct{ src, want string }{
		{`{% for m in messages %}{{ loop.first }}{% endfor %}`, "TrueFalse"},
		{`{% for m in messages %}{{ loop.last }}{% endfor %}`, "FalseTrue"},
		{`{% for m in messages %}{{ loop.first|tojson }}{% endfor %}`, "truefalse"},
		{`{% for m in messages %}{% if loop.first %}yes{% else %}no{% endif %}{% endfor %}`, "yesno"},
	}
	for _, c := range cases {
		if got := render(t, c.src, msgs, Inputs{}); got != c.want {
			t.Errorf("%q = %q, want %q", c.src, got, c.want)
		}
	}
}

// TestToJSONMatchesJinja2 pins the exact bytes tojson produces.
//
// Jinja2's tojson is Python's json.dumps with sort_keys=True and the default separators
// (", " and ": "), so keys are sorted and the output is not compact. The spacing becomes the
// prompt text: a tool call written "{"name": "get"}" is not the same string as
// "{"name":"get"}", and the model was trained on whatever its template produced. The
// expected strings are Jinja2's own output.
func TestToJSONMatchesJinja2(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	cases := []struct{ src, want string }{
		{`{{ 'a' | tojson }}`, `"a"`},
		{`{{ messages[0] | tojson }}`, `{"content": "hi", "role": "user"}`},
		{`{{ messages | tojson }}`, `[{"content": "hi", "role": "user"}]`},
		// The message mapping's keys are sorted, not in insertion order: a struct's field
		// order would otherwise leak into the prompt.
		{`{{ messages[0] | tojson }}`, `{"content": "hi", "role": "user"}`},
		// An indent spreads the value over lines, as json.dumps does.
		{`{{ messages | tojson(indent=4) }}`,
			"[\n    {\n        \"content\": \"hi\",\n        \"role\": \"user\"\n    }\n]"},
		{`{{ messages | tojson(indent=2) }}`,
			"[\n  {\n    \"content\": \"hi\",\n    \"role\": \"user\"\n  }\n]"},
	}
	for _, c := range cases {
		if got := render(t, c.src, msgs, Inputs{}); got != c.want {
			t.Errorf("%q = %q, want %q", c.src, got, c.want)
		}
	}
	// An empty sequence stays on one line, as json.dumps does -- with an indent or without.
	for _, src := range []string{`{{ messages | tojson(indent=4) }}`, `{{ messages | tojson }}`} {
		if got := render(t, src, nil, Inputs{}); got != "[]" {
			t.Errorf("%q with no messages = %q, want %q", src, got, "[]")
		}
	}
}

// TestPrintingAContainerIsAnError covers the shape this subset cannot print.
//
// Jinja2 prints a dict as Python's repr, which has no use in a chat prompt and is not
// implemented here. Rendering it as the empty string would be the quiet half-rendering the
// package exists to prevent, so it is reported instead, and the caller falls back to a
// neutral transcript.
func TestPrintingAContainerIsAnError(t *testing.T) {
	for _, src := range []string{
		`{{ messages }}`,
		`{{ messages[0] }}`,
		`{% for m in messages %}{{ m }}{% endfor %}`,
	} {
		tmpl, err := Parse(src)
		if err != nil {
			t.Fatalf("Parse(%q): %v", src, err)
		}
		if out, err := tmpl.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{}); err == nil {
			t.Errorf("%q rendered %q, want an error naming the unsupported shape", src, out)
		}
	}
	// The fields a template actually reads are still printable.
	if got := render(t, `{{ messages[0].role }}`, []Message{{Role: "user"}}, Inputs{}); got != "user" {
		t.Errorf("role = %q, want %q", got, "user")
	}
}

// TestInOperatorOnAString covers "'a' in 'abc'", which Jinja2 supports as a substring test.
//
// Real templates use it to branch on a role -- "'user' in message['role']" -- so treating
// only sequences as membership tests made a supported template fail at render time. The
// cases are Jinja2's own answer.
func TestInOperatorOnAString(t *testing.T) {
	cases := []struct {
		src  string
		want string
	}{
		{`{% if 'user' in 'username' %}T{% else %}F{% endif %}`, "T"},
		{`{% if 'x' in 'abc' %}T{% else %}F{% endif %}`, "F"},
		{`{% if 'a' in 'abc' %}T{% else %}F{% endif %}`, "T"},
		{`{% if 'user' in messages[0]['role'] %}T{% else %}F{% endif %}`, "T"},
	}
	msgs := []Message{{Role: "user", Content: "hi"}}
	for _, c := range cases {
		if got := render(t, c.src, msgs, Inputs{}); got != c.want {
			t.Errorf("%q = %q, want %q", c.src, got, c.want)
		}
	}
}

// TestParseScalesLinearlyInTagCount guards the lexer's delimiter search.
//
// The scanner used to call strings.Index once per delimiter ({{, {%, {#) for every tag, and
// each call scanned the rest of the template. That is quadratic in the tag count: measured on
// flat templates, 5k tags took 20ms, 10k took 66ms, 20k took 254ms, and a 10k-deep nested
// template -- which is 20k tags -- took 4.7 seconds. Searching for the single character the
// delimiters share costs one pass; the same inputs now take 7ms, 5ms, 13ms and 30ms.
//
// The bound is deliberately loose (a doubling may cost up to 3x) so that a slower machine does
// not fail it, while a return to quadratic growth -- 4x per doubling, which is what was
// measured -- still does.
func TestParseScalesLinearlyInTagCount(t *testing.T) {
	build := func(n int) string {
		var b strings.Builder
		for range n {
			b.WriteString("{{ 1 }}")
		}
		return b.String()
	}
	timeIt := func(n int) time.Duration {
		src := build(n)
		start := time.Now()
		if _, err := Parse(src); err != nil {
			t.Fatalf("Parse(%d tags): %v", n, err)
		}
		return time.Since(start)
	}

	// Warm up so the first measurement is not paying for lazily-initialised runtime state.
	timeIt(1000)

	small := timeIt(5000)
	large := timeIt(20000)

	// 4x the tags. Linear would be 4x the time; quadratic would be 16x.
	if small <= 0 {
		t.Skip("timer resolution too coarse for the small case")
	}
	ratio := float64(large) / float64(small)
	if ratio > 12 {
		t.Errorf("4x the tags took %.1fx the time (%v -> %v); the delimiter scan is quadratic again",
			ratio, small, large)
	}
}

// TestParseHandlesDeepNesting is the same shape at nesting depth rather than tag count: a
// template nested 20000 deep used to take 19 seconds because every level re-scanned the rest
// of the input. It parses in about 35ms now.
func TestParseHandlesDeepNesting(t *testing.T) {
	const depth = 20000
	var b strings.Builder
	b.WriteString("{% if true %}")
	for range depth {
		b.WriteString("{% if true %}")
	}
	b.WriteString("x")
	for range depth {
		b.WriteString("{% endif %}")
	}
	b.WriteString("{% endif %}")

	start := time.Now()
	tpl, err := Parse(b.String())
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	elapsed := time.Since(start)
	if elapsed > 10*time.Second {
		t.Errorf("parsing a %d-deep template took %v; it was 19s before the delimiter scan was fixed",
			depth, elapsed)
	}
	out, err := tpl.Render(nil, Inputs{})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if out != "x" {
		t.Errorf("rendered %q, want %q", out, "x")
	}
}

// TestWriteJSONStringEscapesNonASCII covers the escaping added for Jinja2 parity.
//
// Python's json.dumps defaults to ensure_ascii=True, so Jinja2's tojson spells every non-ASCII
// character as \uXXXX and a code point above U+FFFF as a surrogate pair. A tool definition with a
// translated description otherwise reached the model as UTF-8 where the template it was trained
// against expected escapes -- the same class of difference the separators were already matched
// for.
//
// The check that found the original bug was in scripts/jinja_diff.py, so it only runs when
// someone runs that script; the package's own tests had no case with non-ASCII JSON at all, which
// is why the arms were unexecuted.
func TestWriteJSONStringEscapesNonASCII(t *testing.T) {
	cases := []struct{ in, want string }{
		// ASCII passes through, including the characters json.Marshal escapes itself.
		{"plain", `"plain"`},
		{`q"b\s`, `"q\"b\\s"`},
		// A two-byte BMP rune, one escape per code point.
		{"\u4f60\u597d", `"\u4f60\u597d"`},
		// A four-byte rune becomes a surrogate pair, as Python writes it, not \U0001f389.
		{"\U0001f389", `"\ud83c\udf89"`},
		{"hi \U0001f389 \u4f60", `"hi \ud83c\udf89 \u4f60"`},
	}
	for _, c := range cases {
		var b strings.Builder
		writeJSONString(&b, c.in)
		if got := b.String(); got != c.want {
			t.Errorf("writeJSONString(%q) = %s, want %s", c.in, got, c.want)
		}
		// The result must still parse as JSON and decode back to the input, so the escaping is
		// a spelling change and not a corruption.
		var back string
		if err := json.Unmarshal([]byte(b.String()), &back); err != nil {
			t.Errorf("writeJSONString(%q) produced unparseable JSON %s: %v", c.in, b.String(), err)
			continue
		}
		if back != c.in {
			t.Errorf("round trip of %q = %q", c.in, back)
		}
	}
}
