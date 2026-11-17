package chat

import (
	"fmt"
	"strconv"
	"strings"
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
// Assignments write to the outermost frame, which is what Jinja does for {% set %}
// inside a for body: the variable survives the loop. Reading walks inward, so the loop
// variable shadows an outer name of the same kind.
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
		r.buf.WriteString(v.text)
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
			v, err := evalExpr(br.cond, sc)
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
			return value{}, fmt.Errorf("chat: undefined variable %q", t.name)
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
			return value{}, fmt.Errorf("chat: mapping has no field %q", t.name)
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
			b.WriteString(v.text)
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
		return strValue(toJSON(v)), nil
	case "length":
		if v.isList {
			return strValue(strconv.Itoa(len(v.list))), nil
		}
		return strValue(strconv.Itoa(len(v.text))), nil
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
		if !right.isList {
			return value{}, fmt.Errorf("chat: 'in' needs a sequence on the right")
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

// equalValues compares two values by their string form, except for lists.
func equalValues(a, b value) bool {
	if a.isBool || b.isBool {
		return a.truthy() == b.truthy()
	}
	if a.isList != b.isList {
		return false
	}
	return a.text == b.text
}
