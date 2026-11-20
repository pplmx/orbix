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
	// isMap marks a dictionary, so attribute access on a missing key errors rather
	// than silently yielding the empty string.
	isMap bool
	// isRawJSON marks text that is already JSON, so tojson emits it unquoted.
	isRawJSON bool
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

// scope is a variable environment with one level of nesting for loops.
//
// Reading walks inward, so the loop variable shadows an outer name of the same kind.
//
// Assignments write to the outermost frame, which is NOT what Jinja 3.1 does and the
// comment here said it was. Measured against jinja2 3.1.6: "{% set x = 'before' %}{% for
// m in messages %}{% set x = 'inside' %}{% endfor %}{{ x }}" renders "before" there and
// "inside" here, and a name first assigned inside a loop renders empty there and survives
// here. Jinja scopes a for body, so an assignment in one does not escape it.
//
// Kept as it is because the published templates this package was written for do not
// depend on either reading: Llama 3's assigns and reads are both inside the body, which
// the two agree on. Changing it would mean a second frame kind for a difference no
// template in the corpus shows -- but the disagreement is recorded here rather than
// described as agreement.
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

// assign writes a name to the root frame.
func (s *scope) assign(name string, v value) {
	root := s
	for root.parent != nil {
		root = root.parent
	}
	root.vars[name] = v
}

// child returns a nested scope for a loop iteration.
func (s *scope) child() *scope { return &scope{vars: map[string]value{}, parent: s} }

// renderer walks a node tree, accumulating output.
type renderer struct {
	buf strings.Builder
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
func (r *renderer) renderNode(n node, sc *scope) error {
	switch t := n.(type) {
	case textNode:
		r.buf.WriteString(t.text)
		return nil
	case exprNode:
		v, err := evalExpr(t.expr, sc)
		if err != nil {
			return err
		}
		out, err := v.printErr()
		if err != nil {
			return err
		}
		r.buf.WriteString(out)
		return nil
	case setNode:
		v, err := evalExpr(t.expr, sc)
		if err != nil {
			return err
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
		return strValue(t.val), nil
	case boolLitExpr:
		return boolValue(t.val), nil
	case varExpr:
		v, ok := sc.lookup(t.name)
		if !ok {
			return value{}, undefinedVarError(t)
		}
		return v, nil
	case callExpr:
		// No function is implemented. A template reaches this only if it does not guard the
		// call, and then the honest answer is an error rather than a fabricated value:
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
			return value{}, nil
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
		if !base.isList {
			return value{}, fmt.Errorf("chat: slice applied to a non-sequence")
		}
		from, err := evalExpr(t.from, sc)
		if err != nil {
			return value{}, err
		}
		start, convErr := strconv.Atoi(from.text)
		if convErr != nil {
			return value{}, fmt.Errorf("chat: slice start %q is not a number", from.text)
		}
		if start < 0 {
			start += len(base.list)
		}
		// An out-of-range start yields an empty sequence rather than an error: the point of
		// messages[1:] on a one-message conversation is "nothing after the first", and a
		// template that guards the length should not also have to guard this.
		if start < 0 {
			start = 0
		}
		if start > len(base.list) {
			start = len(base.list)
		}
		return value{list: base.list[start:], isList: true}, nil
	case concatExpr:
		// Every part is converted to text and joined, which is what the + operator
		// means for strings. Templates rely on mixed operands -- a marker plus a
		// number plus a message field -- so no type check is attempted here.
		var b strings.Builder
		for _, part := range t.parts {
			v, err := evalExpr(part, sc)
			if err != nil {
				return value{}, err
			}
			b.WriteString(v.print())
		}
		return strValue(b.String()), nil
	case arithExpr:
		left, err := evalExpr(t.left, sc)
		if err != nil {
			return value{}, err
		}
		right, err := evalExpr(t.right, sc)
		if err != nil {
			return value{}, err
		}
		a, aerr := strconv.Atoi(left.text)
		b, berr := strconv.Atoi(right.text)
		if aerr != nil || berr != nil {
			return value{}, fmt.Errorf("chat: arithmetic on non-numeric %q and %q", left.text, right.text)
		}
		switch t.op {
		case "-":
			// Overflow is checked rather than wrapped. A template that indexes with
			// loop.index - 1 never comes near a limit, but the operands are arbitrary text from
			// the message list, so a value near MinInt64 can reach here -- measured, "-9223372036854775808"
			// minus 1 rendered "9223372036854775807", and MaxInt64 minus -1 wrapped the other
			// way. Wrapping turns a number the template did not ask for into output, while an
			// error names the operands; the alternative is silent wrong text in a prompt.
			if (b > 0 && a < math.MinInt64+b) || (b < 0 && a > math.MaxInt64+b) {
				return value{}, fmt.Errorf("chat: arithmetic overflow: %d - %d", a, b)
			}
			return strValue(strconv.Itoa(a - b)), nil
		default:
			return value{}, fmt.Errorf("chat: unsupported arithmetic operator %q", t.op)
		}
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
		isNone := !v.isBool && !v.isList && !v.isMap && v.text == ""
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
	case isIterableExpr:
		v, err := evalExpr(t.operand, sc)
		if err != nil {
			return value{}, err
		}
		return boolValue(v.isList), nil
	case definedExpr:
		_, ok := sc.lookup(t.name)
		return boolValue(ok), nil
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
