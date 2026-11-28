package chat

import (
	"encoding/json"
	"strings"
	"testing"
)

// renderVars is the common shape: one template, one message, some variables.
func renderVars(t *testing.T, src string, vars map[string]any) string {
	t.Helper()
	tmpl, err := Parse(src)
	if err != nil {
		t.Fatalf("Parse(%q): %v", src, err)
	}
	out, err := tmpl.Render([]Message{{Role: "user", Content: "hi"}}, Inputs{Variables: vars})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	return out
}

// TestTojsonRendersAnEmptyMapAsBraces covers writeJSONDepth's empty-map arm.
//
// A JSON body can carry {} for a chat_template_kwarg -- an object with nothing in it is what
// a caller sends when the field exists but has no settings. Rendering it needs the
// two-character answer rather than the map writer's loop, which would emit "{" and then look
// for keys that are not there. Disabling the arm left the package green: no test rendered an
// empty object.
func TestTojsonRendersAnEmptyMapAsBraces(t *testing.T) {
	got := renderVars(t, `{{ opts | tojson }}`, map[string]any{"opts": map[string]any{}})
	if got != "{}" {
		t.Errorf("tojson of an empty object = %q, want {}", got)
	}

	// With an indent the two paths diverge: the loop writes closeSep, which is a newline and
	// a level for indented output, so an empty object would come out as two lines. This is
	// what makes the arm observable -- without it the compact form happened to match anyway,
	// since "{" + "}" is "{}".
	if got := renderVars(t, `{{ opts | tojson(indent=4) }}`,
		map[string]any{"opts": map[string]any{}}); got != "{}" {
		t.Errorf("tojson(indent=4) of an empty object = %q, want {} on one line", got)
	}

	// The non-empty case still goes through the writer, so the arm is a special case rather
	// than the only path. Only the braces and the key are asserted: how a number inside is
	// rendered is TestVariablesRenderLikeJinja2's subject, and pinning it twice would make
	// this test fail for a reason it is not about.
	if got := renderVars(t, `{{ opts | tojson }}`, map[string]any{
		"opts": map[string]any{"a": float64(1)},
	}); got != `{"a": "1"}` {
		t.Errorf("tojson of {a: 1} = %q, want the writer's braced form", got)
	}
}

// TestVariablesRenderLikeJinja2 pins the values a request body can carry.
//
// The expectation is jinja2 3.1.6 rendering what json.loads makes of the same bytes, because a
// checkpoint's template was written and tested against that. The two disagree in a way that is
// easy to get wrong: this package receives every JSON number as a float64, while Python keeps 3
// an int and 3.0 a float, so the printed text differs unless the value decides rather than the
// Go type. Measured there: `{{ n }}` gives "3" for int and "3.0" for float.
func TestVariablesRenderLikeJinja2(t *testing.T) {
	cases := []struct {
		name string
		src  string
		vars map[string]any
		want string
	}{
		{"json integer", `{{ n }}`, map[string]any{"n": float64(3)}, "3"},
		{"json fraction", `{{ n }}`, map[string]any{"n": 2.5}, "2.5"},
		{"large integer", `{{ n }}`, map[string]any{"n": float64(1234567)}, "1234567"},
		{"string", `{{ s }}`, map[string]any{"s": "a b"}, "a b"},
		{"true prints capitalised", `{{ b }}`, map[string]any{"b": true}, "True"},
		{"false prints capitalised", `{{ b }}`, map[string]any{"b": false}, "False"},
		// A boolean is not the string "true", so a comparison against one must not match.
		{"bool is not a string", `{% if b == 'true' %}eq{% else %}ne{% endif %}`,
			map[string]any{"b": true}, "ne"},
		{"a list is iterable", `{% for x in xs %}{{ x }}{% endfor %}`,
			map[string]any{"xs": []any{float64(1), float64(2)}}, "12"},
		{"a list has a length", `{{ xs|length }}`,
			map[string]any{"xs": []any{"a", "b"}}, "2"},
		{"a mapping reads by field", `{{ m.k }}`,
			map[string]any{"m": map[string]any{"k": "v"}}, "v"},
		{"a mapping serialises", `{{ m|tojson }}`,
			map[string]any{"m": map[string]any{"k": "v"}}, `{"k": "v"}`},
		// null prints nothing and is false in a condition, which is what jinja2 does with a
		// None that is bound rather than absent.
		{"null prints nothing", `[{{ z }}]`, map[string]any{"z": nil}, "[]"},
		{"null is falsy", `{% if z %}T{% else %}F{% endif %}`, map[string]any{"z": nil}, "F"},
		// The Go shapes below never arrive through JSON -- encoding/json produces float64 for
		// every number -- but variableValue documents them as what "a Go caller reaches for
		// first", and none had a case. Making int64 render "MUTANT" left the whole tree green.
		{"Go int", `{{ n }}`, map[string]any{"n": 3}, "3"},
		{"Go int64", `{{ n }}`, map[string]any{"n": int64(3)}, "3"},
		{"Go int64 past 2^32", `{{ n }}`, map[string]any{"n": int64(1) << 40}, "1099511627776"},
		// json.Number keeps its spelling, which is the reason the case exists: a decoder
		// configured with UseNumber would otherwise lose "3" to "3.0".
		{"json.Number", `{{ n }}`, map[string]any{"n": json.Number("3")}, "3"},
		{"json.Number fraction", `{{ n }}`, map[string]any{"n": json.Number("2.50")}, "2.50"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := renderVars(t, c.src, c.vars); got != c.want {
				t.Errorf("rendered %q, want %q", got, c.want)
			}
		})
	}
}

// TestVariablesFromJSONBodies covers the path a request actually takes.
//
// A caller handing Go values directly could pass a float64 for an integer and never notice the
// formatting rule; a request cannot, because encoding/json produces a float64 for every number.
// This decodes a body the way the server does and renders it, so the rule is exercised where it
// matters.
func TestVariablesFromJSONBodies(t *testing.T) {
	const body = `{"enable_thinking": false, "budget": 3, "ratio": 0.5, "labels": ["a","b"]}`
	var vars map[string]any
	if err := json.Unmarshal([]byte(body), &vars); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	got := renderVars(t,
		`{% if not enable_thinking %}no-think {% endif %}{{ budget }}/{{ ratio }}/{{ labels|length }}`,
		vars)
	if want := "no-think 3/0.5/2"; got != want {
		t.Errorf("rendered %q, want %q", got, want)
	}
}

// TestVariableRefusedWhenTheRendererBindsIt covers the names a caller may not set.
//
// Each would change something the request already states: messages is the conversation, and
// add_generation_prompt opens the assistant's turn. Refused rather than merged, because a merged
// one produces a well-formed prompt for a request nobody made.
func TestVariableRefusedWhenTheRendererBindsIt(t *testing.T) {
	tmpl, err := Parse(`{{ messages|length }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	for _, name := range []string{"messages", "bos_token", "eos_token", "add_generation_prompt", "tools"} {
		_, err := tmpl.Render([]Message{{Role: "user", Content: "hi"}},
			Inputs{Variables: map[string]any{name: "x"}})
		if err == nil {
			t.Errorf("%q was accepted as a template variable", name)
			continue
		}
		if !strings.Contains(err.Error(), name) {
			t.Errorf("the refusal for %q does not name it: %v", name, err)
		}
	}
}

// TestVariableRefusalIsStableAcrossRequests pins the ordering.
//
// Two bad names have to report the same one every time, or a client that retries after fixing
// the first sees an error it cannot reproduce.
func TestVariableRefusalIsStableAcrossRequests(t *testing.T) {
	tmpl, err := Parse(`{{ x }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	vars := map[string]any{"messages": 1, "bos_token": 2, "add_generation_prompt": 3}
	var first string
	for range 20 {
		_, err := tmpl.Render(nil, Inputs{Variables: vars})
		if err == nil {
			t.Fatal("two reserved names were accepted")
		}
		if first == "" {
			first = err.Error()
			continue
		}
		if err.Error() != first {
			t.Fatalf("the refusal changed between identical calls:\n%s\n%s", first, err.Error())
		}
	}
}

// TestVariableValueRejectsUnrenderableTypes covers the arm that exists so a caller learns.
//
// A struct reaching a template means the caller expected the renderer to know about it, and
// printing it with %v is not that. The error names the type, because "cannot be a template
// variable" alone leaves a reader guessing what would be accepted.
func TestVariableValueRejectsUnrenderableTypes(t *testing.T) {
	tmpl, err := Parse(`{{ x }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	type notRenderable struct{ A int }
	_, err = tmpl.Render(nil, Inputs{Variables: map[string]any{"x": notRenderable{1}}})
	if err == nil {
		t.Fatal("a struct was accepted as a template variable")
	}
	if !strings.Contains(err.Error(), "notRenderable") {
		t.Errorf("the error does not name the type: %v", err)
	}
	if !strings.Contains(err.Error(), "x") {
		t.Errorf("the error does not name the variable: %v", err)
	}
}

// TestReadsCoversEveryBranch is the property internal/server depends on to refuse a variable.
//
// The answer is static: a name read only in a branch this conversation does not take still
// counts, because the question is about the template rather than about one render.
func TestReadsCoversEveryBranch(t *testing.T) {
	tmpl, err := Parse(`{% if tools %}{{ inside_tools }}{% endif %}{% for m in messages %}{{ m.role }}{% endfor %}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !tmpl.Reads("inside_tools") {
		t.Error("a name read in a branch this render does not take was reported as unread")
	}
	if !tmpl.Reads("messages") {
		t.Error("messages is read by the for loop and was reported as unread")
	}
	if !tmpl.Reads("tools") {
		t.Error("tools is tested by the condition and was reported as unread")
	}
	for _, absent := range []string{"enable_thinking", "loop", "m"} {
		// loop and m are bound by constructs rather than supplied, and a name nothing names is
		// not read. Reporting either as read would let a caller think its value arrived.
		if tmpl.Reads(absent) {
			t.Errorf("%q is not supplied by a caller but Reads reported it", absent)
		}
	}
}

// TestReadsSeesSetExpressions covers the case a naive walk would miss.
//
// {% set x = other %} reads other, even though it writes x. A collector that skipped set nodes
// entirely would report other as unread and a caller's value would be refused for no reason.
func TestReadsSeesSetExpressions(t *testing.T) {
	tmpl, err := Parse(`{% set greeting = prefix + '!' %}{{ greeting }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !tmpl.Reads("prefix") {
		t.Error("a name read on the right of {% set %} was reported as unread")
	}
	if tmpl.Reads("greeting") {
		t.Error("a name the template assigns to itself was reported as a caller variable")
	}
}

// TestReadsIsFalseForANilTemplate keeps the accessor safe to call through a nil renderer.
func TestReadsIsFalseForANilTemplate(t *testing.T) {
	var tmpl *Template
	if tmpl.Reads("anything") {
		t.Error("a nil Template reported reading a name")
	}
}

// TestReadsSeesFilterArguments covers the gap that made a caller refuse a variable the template
// needed.
//
// filterExpr carries its positional arguments twice: arg holds the first, for the filters that take
// one, and args holds all of them. collectReads walked arg only, so a name in the second or third
// position of selectattr was invisible -- `items | selectattr("a", "equalto", target)` rendered with
// target's value and still reported Reads("target") false. A caller using Reads to refuse a
// variable that would do nothing therefore refused a variable that did something, and the prompt
// lost it with nothing in the output saying why.
//
// All three selectattr positions are checked, because the first one happened to work through arg
// and only the later ones were broken: a test written from the working case would have passed.
func TestReadsSeesFilterArguments(t *testing.T) {
	tmpl, err := Parse(`{{ items | selectattr(attrname, "equalto", target) }}`)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	for _, want := range []string{"items", "attrname", "target"} {
		if !tmpl.Reads(want) {
			t.Errorf("Reads(%q) is false; the template reads it: one as the operand, one as "+
				"selectattr's first argument and one as its third", want)
		}
	}

	// The literal arguments are not names, so nothing else should be reported.
	for _, absent := range []string{"equalto", "a"} {
		if tmpl.Reads(absent) {
			t.Errorf("Reads(%q) is true, but that is a string literal in the template, not a "+
				"variable a caller supplies", absent)
		}
	}
}
