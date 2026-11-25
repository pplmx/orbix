package chat

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// value is a renderable value: text, a list, or a message-like map.
//
// The set is deliberately small. Every published chat template works with strings,
// booleans, message dictionaries, and lists of messages; anything richer belongs to a
// real template engine, which this package is not.
type value struct {
	// text is the string form, used for output and concatenation.
	text string
	// list holds the elements of a sequence value.
	list []value
	// fields holds a message dictionary's keys.
	fields map[string]value
	// isBool distinguishes a boolean from the string "true", which matters because
	// the template compares `x == true` and tests bare conditions.
	isBool  bool
	boolean bool
	// isList marks a sequence even when empty, so `for` can tell [] from a string.
	isList bool
	// isNone marks the none literal. Empty text is not enough: Jinja2 renders none as "" but
	// `none is none` is True while `"" is none` is False.
	isNone bool
	// isNumber marks an integer, so the is tests can answer the way Jinja2 does for a number.
	isNumber bool
	// isMap marks a dictionary, so attribute access on a missing key errors rather
	// than silently yielding the empty string.
	isMap bool
	// isRawJSON marks text that is already JSON, so tojson emits it unquoted.
	isRawJSON bool
	// isNamespace marks the mutable bag namespace() returns. It renders as nothing and is only
	// reachable through attribute reads and `{% set ns.field = ... %}`, so the flag is what
	// keeps an ordinary map from being written through by mistake.
	isNamespace bool
}

// strValue wraps a string.
func strValue(s string) value { return value{text: s} }

// boolValue wraps a boolean.
func boolValue(b bool) value { return value{isBool: true, boolean: b, text: strconv.FormatBool(b)} }

// listValue wraps a sequence.
func listValue(items []value) value { return value{list: items, isList: true} }

// print renders the value the way Jinja2 prints it: a bare boolean is "True" or "False".
//
// This differs from the JSON spelling in text, and the difference is observable. A template
// that writes "{{ loop.first }}" gets "True", while "{{ loop.first|tojson }}" gets "true" --
// Jinja2 uses Python's repr for the first and JSON for the second. The text field holds the
// JSON spelling because tojson and the type comparisons read it; output goes through here.
func (v value) print() string {
	if v.isBool && v.boolean {
		return "True"
	}
	if v.isBool {
		return "False"
	}
	return v.text
}

// printErr renders a value to output, reporting the shapes this package cannot print rather
// than emitting nothing for them.
//
// A dict prints in Jinja2 as Python's repr -- "{'role': 'user'}" -- which has no use in a
// chat prompt and is not implemented here. Emitting the empty string instead would be the
// quiet half-rendering the package exists to avoid, so it is an error: a template that does
// this is not one this subset supports, and the caller's fallback is better than a prompt
// with a hole in it.
func (v value) printErr() (string, error) {
	if v.isMap || v.isList {
		return "", fmt.Errorf("chat: cannot print a %s directly; a template that does is "+
			"outside the supported subset", v.shapeName())
	}
	return v.print(), nil
}

// shapeName names a value's container shape for an error message.
//
// The final "value" is unreachable from the one caller: printErr only asks after checking
// isMap || isList, and a value that is neither is printed rather than refused. It stays as the
// answer for "a shape with no name" instead of being deleted, because the switch has to return
// something and an empty string in an error message reads like a formatting bug. Not worth a
// test: there is no input that produces it.
func (v value) shapeName() string {
	switch {
	case v.isMap:
		return "mapping"
	case v.isList:
		return "sequence"
	}
	return "value"
}

// truthy reports whether a value counts as true in a condition.
//
// Jinja's rule: an empty string, an empty list, and a false boolean are false; a
// non-empty anything is true. Getting this wrong turns `{% if messages %}` into an
// always-true branch.
func (v value) truthy() bool {
	if v.isBool {
		return v.boolean
	}
	if v.isList {
		return len(v.list) > 0
	}
	return v.text != ""
}

// scope is a variable environment with one level of nesting per loop body.
//
// Reading walks inward, so the loop variable shadows an outer name of the same kind.
//
// Assignments write to the innermost frame, which follows Jinja 3.1: a for body is a scope, so
// an assignment there does not escape it, and an if body is not a scope, so one there does. Each
// iteration gets a fresh frame (renderNodes calls child per item), so a set in iteration 1 is not
// visible in iteration 2 -- measured against jinja2 3.1.6.
//
// This used to write to the outermost frame instead, and the comment here described that as a
// recorded disagreement that no template depended on. The second half stopped being true when the
// real Qwen2.5 template arrived: its tool-call branch does
//
//	{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}
//	    {%- set tool_call = tool_call.function %}{%- endif %}
//	    {{- tool_call.name }} …
//
// and under the old rule that set wrote past the loop frame, so the template read the *outer*
// dict's name -- empty for a call that carries only function.name -- and the prompt lost every
// tool call. TestQwen2ToolConversationMatchesJinja2 fails on the old rule and
// TestSetIsScopedToTheLoopBody pins the new one, both against values measured from jinja2 3.1.6.
type scope struct {
	vars   map[string]value
	parent *scope
}

// newScope returns a root scope holding the render inputs.
func newScope(vars map[string]value) *scope { return &scope{vars: vars} }

// lookup finds a name, walking outward.
func (s *scope) lookup(name string) (value, bool) {
	for cur := s; cur != nil; cur = cur.parent {
		if v, ok := cur.vars[name]; ok {
			return v, true
		}
	}
	return value{}, false
}

// assign writes a name to the innermost frame, which is how a for body scopes its own sets.
func (s *scope) assign(name string, v value) { s.vars[name] = v }

// assignAttr writes one field of a namespace, which is what `{% set ns.x = ... %}` does.
//
// The bag is shared rather than copied, so the write is visible wherever the variable is. That
// is the difference between this and an ordinary set, and the reason the construct exists: a
// `{% set %}` inside a for body binds in that body's frame and vanishes with it, while a
// namespace field survives the loop.
//
// The name has to resolve to a namespace. Falling back to a plain assign would make
// `{% set ns.x = 1 %}` on an undefined `ns` silently create nothing the template can read,
// which is the class of failure that produces a prompt that looks right and is not.
func (r *renderer) assignAttr(sc *scope, name, attr string, v value) error {
	target, ok := sc.lookup(name)
	if !ok {
		return fmt.Errorf("chat: {%% set %s.%s %%} but %q is not defined", name, attr, name)
	}
	if !target.isNamespace {
		return fmt.Errorf("chat: {%% set %s.%s %%} but %q is not a namespace", name, attr, name)
	}
	target.fields[attr] = v
	return nil
}

// child returns a nested scope for a loop iteration.
func (s *scope) child() *scope { return &scope{vars: map[string]value{}, parent: s} }

// renderer walks a node tree, accumulating output.
type renderer struct {
	buf strings.Builder
	// nodes counts node evaluations for this render, so a template whose nesting makes the
	// work exponential stops before it allocates rather than after. See maxRenderNodes.
	nodes int
}

// maxRenderNodes bounds the node evaluations one render may perform.
//
// The output bound alone is not enough: reaching it costs the work that produced it.
// Measured over 8 messages on a template nesting for-loops, the output multiplied by the
// message count per level, so six levels took 420ms to produce 256 KiB and eight took 17.4s
// to produce 16 MiB -- and a render that stops *at* the byte bound has already done all
// that work. Counting evaluations catches the same shape in milliseconds.
//
// The budget is generous against real templates: the published ones nest one level, so an
// eight-message conversation evaluates on the order of a hundred nodes. A hundred thousand
// leaves three orders of magnitude of headroom while cutting the exponential case off
// long before it becomes visible.
const maxRenderNodes = 100_000

// maxRenderBytes bounds one render's output, as a second line of defence.
//
// The node budget above stops the exponential shape; this one stops a template that stays
// within the node count and still emits something enormous -- a single tojson of a large
// value, say. 16 MiB is far above any prompt a model could attend to (the largest context
// this engine documents is 128k positions, and the request body bound allows 48 bytes per
// position, so about 6 MiB) and far below what a runaway render reaches. The error is
// reported rather than the output truncated: a truncated prompt still looks like a prompt,
// and the model would answer the wrong question.
const maxRenderBytes = 16 << 20

// errRenderTooLarge marks a render that exceeded its size or work budget.
//
// It names the bound, not the message count: the count is the caller's, and it tells a
// template author nothing about which construct multiplied.
var errRenderTooLarge = errors.New("chat: rendered prompt exceeds the renderer's bound")

// charge counts one node evaluation, failing once the render has done more than
// maxRenderNodes of them.
func (r *renderer) charge() error {
	r.nodes++
	if r.nodes > maxRenderNodes {
		return fmt.Errorf("%w (more than %d nodes evaluated)", errRenderTooLarge, maxRenderNodes)
	}
	return nil
}

// write appends to the buffer, failing once the render has produced more than
// maxRenderBytes.
//
// Every write goes through here rather than calling buf.WriteString directly, so a new
// node type cannot grow the output past the bound by forgetting the check.
func (r *renderer) write(s string) error {
	if r.buf.Len()+len(s) > maxRenderBytes {
		return fmt.Errorf("%w (%d bytes)", errRenderTooLarge, maxRenderBytes)
	}
	r.buf.WriteString(s)
	return nil
}

// renderNodes runs each node in order.
func (r *renderer) renderNodes(nodes []node, sc *scope) error {
	for _, n := range nodes {
		if err := r.renderNode(n, sc); err != nil {
			return err
		}
	}
	return nil
}

// undefinedVarError reports a name that is not bound. It is a type rather than a formatted
// string so a condition can recognise it without matching on message text.
type undefinedVarError struct{ name string }

func (e undefinedVarError) Error() string {
	return fmt.Sprintf("chat: undefined variable %q", e.name)
}

// evalCond evaluates a condition, where an unbound name is false rather than an error.
//
// Templates branch on `{% if tools %}` without first testing `tools is defined`, and Jinja
// treats an undefined name as false in that position. The distinction is only for
// conditions: `{{ nope }}` in output still fails, because there is nothing to print and the
// caller can fall back to a neutral transcript instead of sending a prompt with a hole in it.
func evalCond(e expr, sc *scope) (value, error) {
	v, err := evalExpr(e, sc)
	if err == nil {
		return v, nil
	}
	// Only an unbound name is absorbed. Any other failure -- a non-sequence in a for, a
	// field read from a non-mapping -- is a defect in the template and must surface.
	var missing undefinedVarError
	if errors.As(err, &missing) {
		return value{}, nil
	}
	return value{}, err
}

// renderNode runs one node.
//
// The evaluation is charged before the work, so a template that multiplies itself is
// stopped by the count rather than by an allocation it has already made.
func (r *renderer) renderNode(n node, sc *scope) error {
	if err := r.charge(); err != nil {
		return err
	}
	switch t := n.(type) {
	case textNode:
		return r.write(t.text)
	case exprNode:
		v, err := evalExpr(t.expr, sc)
		if err != nil {
			return err
		}
		out, err := v.printErr()
		if err != nil {
			return err
		}
		return r.write(out)
	case setNode:
		v, err := evalExpr(t.expr, sc)
		if err != nil {
			return err
		}
		if t.attr != "" {
			// Assignment through a namespace. The variable holds a mutable bag, so writing a
			// field is visible to every scope that can still see the variable -- which is the
			// whole point of the construct, and why it is not a plain assign.
			if err := r.assignAttr(sc, t.name, t.attr, v); err != nil {
				return err
			}
			return nil
		}
		sc.assign(t.name, v)
		return nil
	case ifNode:
		for _, br := range t.branches {
			v, err := evalCond(br.cond, sc)
			if err != nil {
				return err
			}
			if v.truthy() {
				return r.renderNodes(br.body, sc)
			}
		}
		if t.hasElse {
			return r.renderNodes(t.elseBody, sc)
		}
		return nil
	case forNode:
		seq, err := evalExpr(t.iter, sc)
		if err != nil {
			return err
		}
		if !seq.isList {
			return fmt.Errorf("chat: {%% for %s in ... %%} iterates a non-sequence", t.varName)
		}
		for i, item := range seq.list {
			inner := sc.child()
			inner.vars[t.varName] = item
			inner.vars["loop"] = loopValue(i, len(seq.list))
			if err := r.renderNodes(t.body, inner); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("chat: internal error: unknown node %T", n)
	}
}

// loopValue builds the loop object for one iteration.
func loopValue(i, n int) value {
	fields := map[string]value{
		"index0": strValue(strconv.Itoa(i)),
		"index":  strValue(strconv.Itoa(i + 1)),
		"first":  boolValue(i == 0),
		"last":   boolValue(i == n-1),
		"length": strValue(strconv.Itoa(n)),
		// Counting down is what "how many remain" means in Jinja, and the two are what
		// templates reach for when a separator depends on the position. Checked against
		// jinja2 3.1.6 with two items: revindex is "21" and revindex0 is "10".
		"revindex":  strValue(strconv.Itoa(n - i)),
		"revindex0": strValue(strconv.Itoa(n - i - 1)),
	}
	return value{fields: fields, isMap: true}
}

// evalExpr computes an expression's value.
func evalExpr(e expr, sc *scope) (value, error) {
	switch t := e.(type) {
	case litExpr:
		v := strValue(t.val)
		v.isNone = t.isNone
		v.isNumber = t.isNumber
		return v, nil
	case boolLitExpr:
		return boolValue(t.val), nil
	case varExpr:
		v, ok := sc.lookup(t.name)
		if !ok {
			return value{}, undefinedVarError(t)
		}
		return v, nil
	case callExpr:
		// raise_exception is the one global implemented, because its semantics are to abort with
		// the message the template wrote. Gemma's template uses it to reject a system role and a
		// conversation that does not alternate, and Jinja's own behaviour is to raise -- so
		// reporting "no function is implemented" would hide the sentence that says what is wrong
		// with the conversation and leave an operator reading a fallback with no reason.
		if t.name == "raise_exception" {
			msg := "the template raised an exception"
			if len(t.args) > 0 {
				arg, err := evalExpr(t.args[0], sc)
				if err != nil {
					return value{}, err
				}
				msg = strings.TrimSpace(arg.text)
			}
			return value{}, fmt.Errorf("chat: template raised: %s", msg)
		}
		// A method call dispatches on its receiver. Only the string methods the targeted
		// templates use are implemented, and each mirrors Python's own semantics for the
		// argument it takes -- trim characters, not a substring, which is `strip("\n")`.
		if t.recv != nil {
			recv, err := evalExpr(t.recv, sc)
			if err != nil {
				return value{}, err
			}
			return evalStringMethod(t, recv, sc)
		}
		// namespace() builds the mutable bag `{% set ns.field = ... %}` writes through. Qwen3's
		// template uses it to carry state out of a for body, which a plain set there cannot do
		// because the body is its own frame.
		//
		// The fields start at whatever the keyword arguments evaluate to, and the bag itself
		// renders as nothing -- reading it directly is a template bug, and printing an empty
		// string is what Jinja2 does with it.
		if t.name == "namespace" {
			if len(t.args) > 0 {
				return value{}, fmt.Errorf("chat: namespace() takes keyword arguments only")
			}
			fields := make(map[string]value, len(t.kwNames))
			for i, name := range t.kwNames {
				v, err := evalExpr(t.kwArgs[i], sc)
				if err != nil {
					return value{}, err
				}
				fields[name] = v
			}
			return value{isNamespace: true, isMap: true, fields: fields}, nil
		}
		// Everything else is unimplemented. A template reaches this only if it does not guard
		// the call, and then the honest answer is an error rather than a fabricated value:
		// strftime_now in particular would otherwise put a wrong date in the system prompt.
		return value{}, fmt.Errorf("chat: call to %q: no function is implemented", t.name)
	case attrExpr:
		base, err := evalExpr(t.base, sc)
		if err != nil {
			return value{}, err
		}
		if !base.isMap {
			return value{}, fmt.Errorf("chat: field %q read from a non-mapping", t.name)
		}
		v, ok := base.fields[t.name]
		if !ok {
			// A field that is not there is empty, not an error. Templates disagree about
			// how to ask: Qwen2 writes `not message.tool_calls`, Llama 3 writes
			// `'tool_calls' in message`, and the second only means "no tool calls" if the
			// field is genuinely absent. Both read the same conversation, so a missing
			// field has to be empty for the first and absent for the second -- which is
			// what an empty value plus a key test give.
			//
			// isNone is set because that is what the value is: `message.nosuchfield is none`
			// is True in Jinja2, while `"" is none` is False. Rendering is unaffected --
			// both print nothing.
			return value{isNone: true}, nil
		}
		return v, nil
	case indexExpr:
		base, err := evalExpr(t.base, sc)
		if err != nil {
			return value{}, err
		}
		idx, err := evalExpr(t.index, sc)
		if err != nil {
			return value{}, err
		}
		// A bracket does two jobs: it indexes a sequence by integer and reads a
		// mapping by key. The operand decides which, which is why the dispatch is on
		// base rather than on the index's spelling.
		if base.isMap {
			v, ok := base.fields[idx.text]
			if !ok {
				return value{}, fmt.Errorf("chat: mapping has no field %q", idx.text)
			}
			return v, nil
		}
		i, convErr := strconv.Atoi(idx.text)
		if convErr != nil {
			return value{}, fmt.Errorf("chat: index %q is not a number", idx.text)
		}
		if !base.isList {
			return value{}, fmt.Errorf("chat: index applied to a non-sequence")
		}
		// A negative index counts from the end, which templates use to look at the
		// previous message.
		if i < 0 {
			i += len(base.list)
		}
		if i < 0 || i >= len(base.list) {
			return value{}, fmt.Errorf("chat: index %d outside a sequence of %d", i, len(base.list))
		}
		return base.list[i], nil
	case sliceExpr:
		base, err := evalExpr(t.base, sc)
		if err != nil {
			return value{}, err
		}
		// A slice applies to a sequence or to text. Qwen3's template slices a string
		// (`current_content[:tool_start_length]` to find the payload before the marker), where
		// Python slices characters; the string case returns text rather than a list so the
		// result can be concatenated the way the template does.
		if !base.isList {
			if base.isMap || base.isBool {
				return value{}, fmt.Errorf("chat: slice applied to a non-sequence")
			}
			runes := []rune(base.text)
			idx, err := sliceIndices(t, sc, len(runes))
			if err != nil {
				return value{}, err
			}
			out := make([]rune, len(idx))
			for i, j := range idx {
				out[i] = runes[j]
			}
			return strValue(string(out)), nil
		}
		idx, err := sliceIndices(t, sc, len(base.list))
		if err != nil {
			return value{}, err
		}
		out := make([]value, len(idx))
		for i, j := range idx {
			out[i] = base.list[j]
		}
		return value{list: out, isList: true}, nil
	case concatExpr:
		// Two integers add; every other combination is joined as text. See parseConcat for why
		// both shapes are needed: Qwen2.5's tool branch subscripts messages[loop.index0 + 1],
		// which is a sum, while every template in the corpus also writes marker + field, which
		// is a join.
		//
		// The fold is left to right over the accumulated value, so 1 + 2 + 3 is 6 and not "12"
		// then "123": each step decides on the running result rather than on the original
		// operands, which is what Python's own left-to-right evaluation does.
		var acc *value
		for _, part := range t.parts {
			v, err := evalExpr(part, sc)
			if err != nil {
				return value{}, err
			}
			if acc == nil {
				acc = &v
				continue
			}
			sum, ok := addValues(*acc, v)
			if !ok {
				sum = strValue(acc.print() + v.print())
			}
			acc = &sum
		}
		return *acc, nil
	case arithExpr:
		left, err := evalExpr(t.left, sc)
		if err != nil {
			return value{}, err
		}
		right, err := evalExpr(t.right, sc)
		if err != nil {
			return value{}, err
		}
		return evalArithmetic(t.op, left.text, right.text)
	case filterExpr:
		return evalFilter(t, sc)
	case isNoneExpr:
		// An undefined name is none, so the lookup is done directly rather than by
		// evaluating first and catching the error.
		var absent bool
		if v, ok := t.operand.(varExpr); ok {
			_, present := sc.lookup(v.name)
			absent = !present
		}
		if absent {
			return boolValue(!t.negate), nil
		}
		v, err := evalExpr(t.operand, sc)
		if err != nil {
			return value{}, err
		}
		isNone := v.isNone
		if t.negate {
			return boolValue(!isNone), nil
		}
		return boolValue(isNone), nil
	case isMappingExpr:
		v, err := evalExpr(t.operand, sc)
		if err != nil {
			return value{}, err
		}
		return boolValue(v.isMap), nil
	case condExpr:
		// Jinja tests the condition's truthiness the same way an {% if %} does, so this reuses
		// evalCond rather than inventing a second rule.
		cond, err := evalCond(t.cond, sc)
		if err != nil {
			return value{}, err
		}
		if cond.truthy() {
			return evalExpr(t.then, sc)
		}
		return evalExpr(t.otherwise, sc)
	case isTrueExpr:
		v, err := evalExpr(t.operand, sc)
		if err != nil {
			return value{}, err
		}
		// A non-boolean is never "true" or "false": `1 is true` is False in Jinja2.
		if !v.isBool {
			return boolValue(false), nil
		}
		return boolValue(v.boolean == t.want), nil
	case isStringExpr:
		// A string is a value that is neither a sequence nor a dictionary nor a boolean. The
		// distinction the templates draw is exactly that coarse -- Gemma 3 asks whether content is
		// a string to decide between a plain turn and a list of parts -- so testing the flags is
		// the faithful answer rather than inferring from the text.
		v, err := evalExpr(t.operand, sc)
		if err != nil {
			return value{}, err
		}
		// A number is not a string: `42 is string` is False in Jinja2.
		return boolValue(!v.isList && !v.isMap && !v.isBool && !v.isNumber), nil
	case isIterableExpr:
		v, err := evalExpr(t.operand, sc)
		if err != nil {
			return value{}, err
		}
		// Only a list. Jinja2 answers True for a string and for a dict as well, and the
		// difference is deliberate here: both real templates that use this test depend on a
		// string answering False. Llama 3.2 writes
		// `if message.content is mapping or message.content is iterable` to mean "content is a
		// list of parts, so JSON-encode it", and Gemma 3 reaches this branch only after
		// `is string` has already excluded strings. Under Jinja2's rule a plain string takes
		// the encode-it path and comes out quoted -- which is what
		// TestLlama32ToolMessageContentIsNotQuoted catches. See doc.go.
		return boolValue(v.isList), nil
	case definedExpr:
		_, ok := sc.lookup(t.name)
		return boolValue(ok), nil
	case indexDefinedExpr:
		base, err := evalExpr(t.index.base, sc)
		if err != nil {
			return value{}, err
		}
		key, err := evalExpr(t.index.index, sc)
		if err != nil {
			return value{}, err
		}
		if base.isMap {
			_, ok := base.fields[key.text]
			return boolValue(ok), nil
		}
		// A sequence subscript is defined only when the element exists; an out-of-range index
		// is the absent case, not an error, because that is what the template is asking about.
		if base.isList {
			n, err := strconv.Atoi(key.text)
			if err != nil {
				return boolValue(false), nil
			}
			return boolValue(n >= 0 && n < len(base.list)), nil
		}
		return boolValue(false), nil
	case attrDefinedExpr:
		base, err := evalExpr(t.base, sc)
		if err != nil {
			return value{}, err
		}
		if !base.isMap {
			return boolValue(false), nil
		}
		_, ok := base.fields[t.name]
		return boolValue(ok), nil
	case notExpr:
		v, err := evalExpr(t.inner, sc)
		if err != nil {
			return value{}, err
		}
		return boolValue(!v.truthy()), nil
	case binaryExpr:
		return evalBinary(t, sc)
	default:
		return value{}, fmt.Errorf("chat: internal error: unknown expression %T", e)
	}
}

// evalFilter applies one filter.
func evalFilter(f filterExpr, sc *scope) (value, error) {
	v, err := evalExpr(f.operand, sc)
	if err != nil {
		return value{}, err
	}
	switch f.name {
	case "trim":
		return strValue(strings.TrimSpace(v.text)), nil
	case "tojson":
		// "tojson(indent=n)" asks for indented JSON; with no argument Python's json.dumps
		// default applies. The indent is a formatting choice the template made, so it has
		// to be honored -- the tool call is part of the prompt text.
		if f.kwargs != nil {
			if ie, ok := f.kwargs["indent"]; ok {
				iv, err := evalExpr(ie, sc)
				if err != nil {
					return value{}, err
				}
				n, convErr := strconv.Atoi(iv.print())
				if convErr != nil {
					return value{}, fmt.Errorf("chat: tojson indent %q is not a number", iv.print())
				}
				if n, err = checkIndent(n); err != nil {
					return value{}, err
				}
				return strValue(toJSONIndent(v, n)), nil
			}
		}
		if f.arg != nil {
			av, err := evalExpr(f.arg, sc)
			if err != nil {
				return value{}, err
			}
			// Jinja2's tojson takes the indent positionally too.
			n, convErr := strconv.Atoi(av.print())
			if convErr != nil {
				return value{}, fmt.Errorf("chat: tojson indent %q is not a number", av.print())
			}
			if n, err = checkIndent(n); err != nil {
				return value{}, err
			}
			return strValue(toJSONIndent(v, n)), nil
		}
		return strValue(toJSON(v)), nil
	case "first", "last":
		// Empty input yields none rather than an error: Qwen3's template writes
		// `content.split('</think>')|last` on a reply that may not contain the marker at all,
		// and a one-element split is the case that has to keep working. Jinja2 gives undefined
		// for both, which this package represents as none.
		if !v.isList {
			return value{}, fmt.Errorf("chat: %q needs a sequence", f.name)
		}
		if len(v.list) == 0 {
			return value{isNone: true}, nil
		}
		if f.name == "first" {
			return v.list[0], nil
		}
		return v.list[len(v.list)-1], nil
	case "length":
		if v.isList {
			return strValue(strconv.Itoa(len(v.list))), nil
		}
		// Runes, not bytes: Jinja2's length is Python's len, which counts characters.
		// Measured against Jinja2 -- "中文" is 2 there and was 6 here, so a template
		// branching on a length threshold would have taken the other arm.
		return strValue(strconv.Itoa(utf8.RuneCountInString(v.text))), nil
	default:
		return value{}, fmt.Errorf("chat: unsupported template filter %q", f.name)
	}
}

// evalBinary evaluates a comparison or boolean operator.
func evalBinary(b binaryExpr, sc *scope) (value, error) {
	left, err := evalExpr(b.left, sc)
	if err != nil {
		return value{}, err
	}
	// Short-circuit, so `x is defined and x == y` does not evaluate y's read of x.
	switch b.op {
	case "and":
		if !left.truthy() {
			return boolValue(false), nil
		}
		right, err := evalExpr(b.right, sc)
		if err != nil {
			return value{}, err
		}
		return boolValue(right.truthy()), nil
	case "or":
		if left.truthy() {
			return boolValue(true), nil
		}
		right, err := evalExpr(b.right, sc)
		if err != nil {
			return value{}, err
		}
		return boolValue(right.truthy()), nil
	}

	right, err := evalExpr(b.right, sc)
	if err != nil {
		return value{}, err
	}
	switch b.op {
	case "==":
		return boolValue(equalValues(left, right)), nil
	case "!=":
		return boolValue(!equalValues(left, right)), nil
	case "<", ">", "<=", ">=":
		// Ordering compares numbers when both sides are numbers and text otherwise, which is
		// Python 3's rule. A string comparison is not a fallback for a failed number parse: the
		// templates this package targets compare `start_pos < 0` (numbers) and nothing else, so
		// text ordering is here because the alternative -- refusing -- would break a template
		// that compares two strings for no reason a caller can act on.
		cmp, ok := compareValues(left, right)
		if !ok {
			return boolValue(false), nil
		}
		switch b.op {
		case "<":
			return boolValue(cmp < 0), nil
		case ">":
			return boolValue(cmp > 0), nil
		case "<=":
			return boolValue(cmp <= 0), nil
		default:
			return boolValue(cmp >= 0), nil
		}
	case "in":
		// A mapping is tested by key, which is what "'tool_calls' in message" means in the
		// tool-call branches: does this message dict have that field at all.
		if right.isMap {
			_, present := right.fields[left.text]
			return boolValue(present), nil
		}
		if !right.isList {
			// Jinja2 also tests a substring in a string, which is what
			// "'user' in message['role']" means in the conditional templates write.
			return boolValue(strings.Contains(right.print(), left.print())), nil
		}
		for _, item := range right.list {
			if equalValues(left, item) {
				return boolValue(true), nil
			}
		}
		return boolValue(false), nil
	default:
		return value{}, fmt.Errorf("chat: unsupported operator %q", b.op)
	}
}

// addValues adds two integers, reporting false when either operand is not one.
//
// The text form is what is inspected, because that is where a number lives in this value type: a
// literal, a loop index and a variable from JSON all carry their digits there with isNumber unset
// in the first two cases. Parsing the text is therefore the check that makes `loop.index0 + 1`
// reach the right element, which is the whole reason + had to learn to add.
func addValues(a, b value) (value, bool) {
	if a.isBool || b.isBool || a.isList || b.isList || a.isMap || b.isMap || a.isNone || b.isNone {
		return value{}, false
	}
	x, err := strconv.ParseInt(a.text, 10, 64)
	if err != nil {
		return value{}, false
	}
	y, err := strconv.ParseInt(b.text, 10, 64)
	if err != nil {
		return value{}, false
	}
	// A sum outside int64 falls back to concatenation, which is the documented behaviour for
	// every other pair this function declines -- there is one fallback path, not two.
	//
	// The arithmetic used to be bounded at int32 and, on overflow, joined the operands: measured
	// before this, {{ 2147483647 + 1 }} rendered "21474836471". That is worse than either
	// alternative because it looks like a number and is not the one the template computed.
	// Jinja2 adds these exactly (Python's ints are unbounded), so int64 is the closest this
	// package can come without a big-integer dependency; a sum that exceeds it joins instead of
	// silently wrapping.
	if (y > 0 && x > math.MaxInt64-y) || (y < 0 && x < math.MinInt64-y) {
		return value{}, false
	}
	return value{text: strconv.FormatInt(x+y, 10), isNumber: true}, true
}

// equalValues compares two values by their string form, except for booleans and lists.
func equalValues(a, b value) bool {
	if a.isBool || b.isBool {
		// A boolean equals only another boolean, and then by its value. Comparing by truth
		// value instead -- which is what this did -- throws away the distinction isBool
		// exists to make: a non-empty string is truthy, so `true == 'false'` came out true.
		// Jinja2 compares strictly here, because the literals are bool and str.
		if a.isBool != b.isBool {
			return false
		}
		return a.boolean == b.boolean
	}
	if a.isList != b.isList {
		return false
	}
	return a.text == b.text
}

// maxJSONIndent bounds tojson's indent, because the value is multiplied by the nesting depth
// and the product becomes the length of a string.Repeat.
//
// Measured before this check: tojson(9223372036854775807) panicked with "makeslice: len out of
// range", and tojson(100000) on four short messages produced 2.4 MB of output. A template is
// model-file content, so neither is a shape a caller asked for.
//
// Python's json.dumps has no such bound, and a template in the wild writes a single digit. A
// limit well above anything a formatting choice needs keeps every real template working while
// turning the two failures above into an error the template author can read.
const maxJSONIndent = 1024

// checkIndent rejects an indent that would make the output unreasonably large. A negative
// indent is not an error: json.dumps treats 0 and negative alike as "no newlines", which the
// style already handles.
func checkIndent(n int) (int, error) {
	if n > maxJSONIndent {
		return 0, fmt.Errorf("chat: tojson indent %d is larger than the %d this renderer allows, "+
			"because the indent is repeated at every nesting level", n, maxJSONIndent)
	}
	return n, nil
}

// evalArithmetic applies one of the two arithmetic operators this package implements.
//
// It works on the operands' text because that is what a template produces: a number in a
// template is a string until something asks it to be arithmetic.
//
// Subtraction checks for overflow rather than wrapping. A template that indexes with
// loop.index - 1 never comes near a limit, but the operands are arbitrary text from the message
// list, so a value near MinInt64 can reach here -- measured, "-9223372036854775808" minus 1
// rendered "9223372036854775807", and MaxInt64 minus -1 wrapped the other way. Wrapping turns a
// number the template did not ask for into output, while an error names the operands.
func evalArithmetic(op, left, right string) (value, error) {
	a, aerr := strconv.Atoi(left)
	b, berr := strconv.Atoi(right)
	if aerr != nil || berr != nil {
		return value{}, fmt.Errorf("chat: arithmetic on non-numeric %q and %q", left, right)
	}
	switch op {
	case "-":
		if (b > 0 && a < math.MinInt64+b) || (b < 0 && a > math.MaxInt64+b) {
			return value{}, fmt.Errorf("chat: arithmetic overflow: %d - %d", a, b)
		}
		return strValue(strconv.Itoa(a - b)), nil
	case "%":
		// Modulo by zero is an error rather than a panic or a zero, and the sign follows Go's
		// truncating division, which is not what Jinja (and Python's %) do for negative
		// operands. Templates only use this for parity tests on non-negative loop indices, so
		// the difference is unreachable from the ones this package renders; the error below is
		// what keeps it from becoming reachable by accident.
		if b == 0 {
			return value{}, fmt.Errorf("chat: modulo by zero")
		}
		if a < 0 || b < 0 {
			return value{}, fmt.Errorf("chat: modulo of a negative operand (%d %% %d) is not "+
				"supported; Jinja's result differs from Go's for those operands", a, b)
		}
		return strValue(strconv.Itoa(a % b)), nil
	default:
		return value{}, fmt.Errorf("chat: unsupported arithmetic operator %q", op)
	}
}

// evalStringMethod evaluates `s.strip(...)` and its three siblings.
//
// The argument semantics are Python's, not Go's, and the difference matters for the templates
// this package targets: `s.strip("\n")` removes newlines from both ends (a character set), it
// does not remove the substring "\n". Qwen3's template writes `content.lstrip('\n')` and
// `m_content.split('</think>')`, so getting this wrong would leave the reasoning block in the
// prompt or cut it in the wrong place.
//
// split() with no argument splits on runs of whitespace, which is Python's rule and not Go's
// strings.Fields-by-name; it is implemented here rather than delegated because the empty
// separator has to be rejected as Python does, not silently treated as "no split".
func evalStringMethod(t callExpr, recv value, sc *scope) (value, error) {
	if !recv.isList && !recv.isMap && !recv.isNone && recv.isBool {
		return value{}, fmt.Errorf("chat: method %q called on a boolean", t.name)
	}
	s := recv.text
	switch t.name {
	case "strip", "lstrip", "rstrip":
		cut := " \t\r\n\v\f"
		if len(t.args) > 1 {
			return value{}, fmt.Errorf("chat: %s() takes at most one argument", t.name)
		}
		if len(t.args) == 1 {
			arg, err := evalExpr(t.args[0], sc)
			if err != nil {
				return value{}, err
			}
			if arg.text == "" {
				return value{}, fmt.Errorf("chat: %s of an empty set removes nothing", t.name)
			}
			cut = arg.text
		}
		switch t.name {
		case "strip":
			return strValue(strings.Trim(s, cut)), nil
		case "lstrip":
			return strValue(strings.TrimLeft(s, cut)), nil
		default:
			return strValue(strings.TrimRight(s, cut)), nil
		}
	case "split":
		var parts []string
		switch len(t.args) {
		case 0:
			parts = strings.Fields(s)
		case 1:
			sep, err := evalExpr(t.args[0], sc)
			if err != nil {
				return value{}, err
			}
			if sep.text == "" {
				return value{}, fmt.Errorf("chat: split of an empty separator is not supported")
			}
			parts = strings.Split(s, sep.text)
		default:
			return value{}, fmt.Errorf("chat: split() takes at most one argument")
		}
		out := make([]value, len(parts))
		for i, part := range parts {
			out[i] = strValue(part)
		}
		return value{list: out, isList: true}, nil
	case "startswith", "endswith":
		// Qwen3's released template guards its tool-response branch with these:
		// `not (message.content.startswith('<tool_response>') and
		// message.content.endswith('</tool_response>'))`. Without them the template does not
		// render at all, so this is what makes that checkpoint's own format usable rather
		// than a nicety.
		//
		// Python's signature also accepts a tuple of prefixes. There is no tuple type in this
		// package and no published template passes one, so a second argument is refused by
		// the arity check below rather than silently compared against the first.
		if len(t.args) != 1 {
			return value{}, fmt.Errorf("chat: %s() takes one argument", t.name)
		}
		arg, err := evalExpr(t.args[0], sc)
		if err != nil {
			return value{}, err
		}
		if t.name == "startswith" {
			return boolValue(strings.HasPrefix(s, arg.text)), nil
		}
		return boolValue(strings.HasSuffix(s, arg.text)), nil
	}
	// Everything else is refused at render time, because a method call parses anywhere: the
	// receiver is just an expression to the left of the dot, and the parser has no reason to
	// know which names are methods until something evaluates them.
	//
	// A "join" case lived here and could not be reached. Its argument is the separator and its
	// receiver has to be a sequence, but `s` above is recv.text, so the receiver a method call
	// can deliver is always text: `messages.join('-')` reads messages' text (empty) and then
	// fails "needs a sequence", and every other spelling does the same -- measured on five,
	// including one over a slice and one through a {% set %} binding. Jinja's join is a filter
	// (`list | join('-')`), and it is not in supportedFilters, so the parse refuses it before
	// this function runs. Removed rather than left as an unreachable branch: it made the
	// package look like it supported join.
	return value{}, fmt.Errorf("chat: method %q is not implemented", t.name)
}

// compareValues orders two scalars, reporting whether they are comparable at all.
//
// Numbers compare numerically, everything else lexically. NaN and a failed parse give false
// rather than an error: a template comparing those is asking a question with no answer, and
// Jinja2 answers False for it.
func compareValues(left, right value) (int, bool) {
	if left.isNumber && right.isNumber {
		l, lerr := strconv.ParseFloat(left.text, 64)
		r, rerr := strconv.ParseFloat(right.text, 64)
		if lerr != nil || rerr != nil {
			return 0, false
		}
		switch {
		case l < r:
			return -1, true
		case l > r:
			return 1, true
		default:
			return 0, true
		}
	}
	return strings.Compare(left.text, right.text), true
}

// sliceBounds resolves a slice expression's two bounds against a length.
//
// Either bound may be absent. A negative bound counts from the end, and a bound outside the
// sequence is clamped rather than refused: the point of messages[1:] on a one-message
// conversation is "nothing after the first", and a template that guards the length should not
// also have to guard this. Python clamps too, which is what these templates were written against.
// sliceIndices evaluates a slice into the list of indices it selects, which is Python's
// slice.indices followed by the selection itself.
//
// Indices rather than a [start:end] range because a step can be negative, and then the two
// bounds mean the opposite of what a range needs: `[::-1]` selects every element backwards,
// while its absent bounds stand for len-1 and "before the first", not 0 and len. Python's own
// slice object resolves this by clamping in a step-dependent way, and reproducing that here is
// the only way `messages[::-1]` and `messages[3:0:-1]` both come out right.
//
// The alternative -- computing start and end and then walking -- is where this started, and it
// silently returned an empty result for a negative step because end < start is exactly the
// condition it used to clamp to empty.
func sliceIndices(t sliceExpr, sc *scope, n int) ([]int, error) {
	step := 1
	if t.step != nil {
		v, err := evalExpr(t.step, sc)
		if err != nil {
			return nil, err
		}
		step, err = strconv.Atoi(v.text)
		if err != nil {
			return nil, fmt.Errorf("chat: slice step %q is not a number", v.text)
		}
		// Python raises ValueError for a zero step rather than returning an empty result.
		// A template that wrote it meant something; reporting is what lets the caller fall back
		// instead of rendering a prompt with a silently empty loop.
		if step == 0 {
			return nil, fmt.Errorf("chat: slice step is zero")
		}
	}

	// Defaults depend on the direction, which is the whole reason this is not a [start:end]
	// calculation: with a negative step the walk starts at the last element and stops short of
	// the first, so the absent bounds are n-1 and -1 rather than 0 and n.
	start, end := 0, n
	if step < 0 {
		start, end = n-1, -1
	}

	if t.from != nil {
		v, err := evalExpr(t.from, sc)
		if err != nil {
			return nil, err
		}
		got, err := strconv.Atoi(v.text)
		if err != nil {
			return nil, fmt.Errorf("chat: slice start %q is not a number", v.text)
		}
		start = clampIndex(got, n, step)
	}
	if t.to != nil {
		v, err := evalExpr(t.to, sc)
		if err != nil {
			return nil, err
		}
		got, err := strconv.Atoi(v.text)
		if err != nil {
			return nil, fmt.Errorf("chat: slice end %q is not a number", v.text)
		}
		end = clampIndex(got, n, step)
	}

	var out []int
	if step > 0 {
		for i := start; i < end; i += step {
			out = append(out, i)
		}
	} else {
		for i := start; i > end; i += step {
			out = append(out, i)
		}
	}
	return out, nil
}

// clampIndex brings one bound into range for the direction of travel, which is what Python's
// slice.indices does before the walk.
//
// A negative bound counts from the end in both directions, so `[-1:]` is the last element and
// `[:-1]` is everything but it. The clamps then differ by direction: a forward walk may not
// start below 0 or go past n, while a backward walk may not start above n-1 or go below -1 --
// -1 being "past the beginning" for a negative step, not a wrap of the last element.
func clampIndex(i, n, step int) int {
	if i < 0 {
		i += n
	}
	if step > 0 {
		return max(0, min(i, n))
	}
	return max(-1, min(i, n-1))
}
