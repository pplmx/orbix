package chat

import (
	"fmt"
	"strconv"
	"strings"
)

// expr is a parsed template expression.
type expr interface{}

// litExpr is a constant: a string or a number, already decoded.
type litExpr struct {
	val string
	// isNone marks Jinja's none literal, which renders as "" and is otherwise
	// indistinguishable from the empty string -- except by the is none test, where Jinja2
	// answers True for one and False for the other.
	isNone bool
	// isNumber marks an integer literal. The is tests need it: `42 is iterable` is False and
	// `42 is string` is False in Jinja2, and without the marker this subset cannot tell 42
	// from "42".
	isNumber bool
}

// varExpr reads a variable from scope.
type varExpr struct{ name string }

// attrExpr reads a field from a map or a field of the loop variable.
type attrExpr struct {
	base expr
	name string
}

// indexExpr reads a sequence element by integer index.
type indexExpr struct {
	base  expr
	index expr
}

// isNoneExpr tests for an absent or empty value, which is what `x is none` means in the
// templates this package reads: an unset variable and a null one are the same thing here
// because no distinction reaches the renderer.
type isNoneExpr struct {
	operand expr
	negate  bool
}

// isMappingExpr tests whether a value is a dictionary, which tool-call templates branch on
// to decide between re-emitting JSON and writing text.
type isMappingExpr struct{ operand expr }

// isIterableExpr tests whether a value is a sequence.
type isIterableExpr struct{ operand expr }

// isStringExpr is "x is string": true for a value that is not a sequence, a dictionary or a bool.
type isStringExpr struct{ operand expr }

// condExpr is an inline "a if cond else b".
type condExpr struct{ cond, then, otherwise expr }

// indexDefinedExpr is "x[key] is defined": the key is present in a mapping.
type indexDefinedExpr struct{ index indexExpr }

// boolLitExpr is a true or false literal, kept distinct from a string because a condition
// has to see the boolean rather than a non-empty text.
type boolLitExpr struct{ val bool }

// sliceExpr is a subscript range. Any bound may be absent, and so may the step:
// `messages[:n]` keeps a prefix and `messages[1:]` drops a leading element, both of which the
// templates this package targets write; `messages[::-1]` walks backwards from the end, which
// Qwen3 0.6B's released template uses to find the last real user turn.
//
// The step is a separate field rather than desugaring into the two bounds because Python's
// absent-bound rules differ between a forward and a backward step: for a negative step the
// missing start is the last element and the missing stop is "past the beginning". Treating the
// absent bounds as 0 and n and then stepping would render an empty loop instead of a reversed
// one, which is a plausible-looking prompt that is wrong.
type sliceExpr struct {
	base expr
	from expr // nil for the [:n] form
	to   expr // nil for the [n:] form
	step expr // nil for a single-colon slice, which is step 1
}

// isTrueExpr tests whether a value is the boolean true or false.
//
// It is not an equality test: Jinja2's `1 is true` is False, because the test asks whether the
// value is the boolean rather than whether it is truthy. Qwen3's template uses it on a
// namespace field it assigned `true` to, so the distinction does not bite there -- but writing
// it as equality would give the wrong answer the first time a template applies it to a number.
type isTrueExpr struct {
	operand expr
	want    bool
}

// concatExpr joins its operands, converting each to text.
type concatExpr struct{ parts []expr }

// filterExpr applies a named filter to its operand.
type filterExpr struct {
	// operand is what preceded the '|'.
	operand expr
	name    string
	arg     expr
	// kwargs holds name=value arguments. They are parsed so a template that passes one
	// compiles, and stored so a filter can read the ones it understands; tojson's indent
	// only selects a whitespace style, and this package emits compact JSON either way.
	kwargs map[string]expr
}

// definedExpr reports whether a name is bound, which templates use to test optional
// inputs like tools.
type definedExpr struct{ name string }

// attrDefinedExpr reports whether a mapping has a field, without failing when it does
// not. Templates use it to branch on an optional member.
type attrDefinedExpr struct {
	base expr
	name string
}

// notExpr negates a condition.
type notExpr struct{ inner expr }

// binaryExpr is a comparison, boolean, or arithmetic operator.
type binaryExpr struct {
	op          string
	left, right expr
}

// arithExpr is integer arithmetic, which templates use to index relative to the loop
// position ("messages[loop.index0 - 1]").
type arithExpr struct {
	op          string
	left, right expr
}

// token is one lexical unit of an expression.
//
// The isString flag matters: a decoded literal like "abc" is indistinguishable from a
// bare name "abc", and treating a marker such as "<|im_start|>" as a name is what made
// every template with a special token in a string fail to parse.
type token struct {
	text     string
	isString bool
}

// supportedFilters is the filter set this package implements.
//
// It is checked at parse time so a template using an unimplemented filter fails to
// compile. Discovering it at render time would mean the caller had already committed to
// a prompt that cannot be finished.
var supportedFilters = map[string]bool{
	"trim":   true,
	"tojson": true,
	"length": true,
	"first":  true,
	"last":   true,
}

// callExpr is a function call. It is parsed so a guarded call does not reject the whole
// template, and it fails at evaluation because no function is implemented.
type callExpr struct {
	name string
	args []expr
	// kwNames and kwArgs hold the `name=value` form, in written order. Only namespace() takes
	// keyword arguments in the templates this package targets, but the parser accepts them for
	// any call so an unsupported one is reported by the evaluator (which knows the function)
	// rather than by the parser (which does not).
	kwNames []string
	kwArgs  []expr
	// recv is the receiver of a method call (`s` in `s.strip()`), nil for a plain function.
	recv expr
}

// parseCall reads "( expr, ... )" after a function name.
func (p *exprParser) parseCall(name string) (expr, error) {
	p.pos++ // the "("
	c := callExpr{name: name}
	if next, ok := p.peekToken(); ok && next.text == ")" {
		p.pos++
		return c, nil
	}
	for {
		// A keyword argument is an identifier followed by "=". The lexer keeps "=" as its own
		// token, so this is a two-token lookahead rather than a name-mangling rule.
		if p.isKeywordArg() {
			name := p.toks[p.pos].text
			p.pos += 2 // the name and the "="
			arg, err := p.parseOr()
			if err != nil {
				return nil, err
			}
			c.kwNames = append(c.kwNames, name)
			c.kwArgs = append(c.kwArgs, arg)
		} else {
			arg, err := p.parseOr()
			if err != nil {
				return nil, err
			}
			c.args = append(c.args, arg)
		}
		next, ok := p.peekToken()
		if !ok {
			return nil, fmt.Errorf("chat: call to %q is not closed", name)
		}
		p.pos++
		if next.text == ")" {
			return c, nil
		}
		if next.text != "," {
			return nil, fmt.Errorf("chat: expected \",\" or \")\" in call to %q, got %q", name, next.text)
		}
	}
}

// isKeywordArg reports whether the next two tokens are `identifier =`.
//
// The "=" has to be checked as its own token: `x == y` lexes as "==", so an identifier
// followed by "==" is a comparison and not a keyword argument.
func (p *exprParser) isKeywordArg() bool {
	if p.pos+1 >= len(p.toks) {
		return false
	}
	name := p.toks[p.pos]
	return !name.isString && isIdentifier(name.text) && p.toks[p.pos+1].text == "="
}

// exprParser parses one expression from a token stream.
type exprParser struct {
	toks []token
	pos  int
}

// parseExpr parses a complete expression, rejecting trailing input.
func parseExpr(src string) (expr, error) {
	toks, err := lexExpr(src)
	if err != nil {
		return nil, err
	}
	p := &exprParser{toks: toks}
	e, err := p.parseConditional()
	if err != nil {
		return nil, err
	}
	if p.pos != len(p.toks) {
		return nil, fmt.Errorf("chat: unexpected %q after expression %q", p.toks[p.pos].text, src)
	}
	return e, nil
}

// lexExpr splits an expression into tokens, keeping quoted strings whole.
//
// Strings are decoded here rather than in the parser because a template's literals
// carry escapes ("\n" is a newline) and a chat format depends on them: a marker split
// across two lines is a different prompt.
func lexExpr(src string) ([]token, error) {
	var toks []token
	for i := 0; i < len(src); {
		c := src[i]
		switch {
		case c == ' ' || c == '\t' || c == '\r' || c == '\n':
			i++
		case c == '\'' || c == '"':
			s, n, err := lexString(src[i:])
			if err != nil {
				return nil, err
			}
			toks = append(toks, token{text: s, isString: true})
			i += n
		case c == '=' || c == '!' || c == '<' || c == '>':
			// Two-character comparisons first, so "==" is not read as two '='.
			if i+1 < len(src) && src[i+1] == '=' {
				toks = append(toks, token{text: src[i : i+2]})
				i += 2
				continue
			}
			// A bare '=' is only legal inside a filter's keyword argument, where the parser
			// consumes it; emitting it as a token lets that happen, and parseOr still
			// rejects it anywhere else.
			if c == '=' {
				toks = append(toks, token{text: "="})
				i++
				continue
			}
			toks = append(toks, token{text: string(c)})
			i++
		case c == '|' || c == '(' || c == ')' || c == '[' || c == ']' ||
			c == '+' || c == '-' || c == '%' || c == '.' || c == ',' || c == ':':
			toks = append(toks, token{text: string(c)})
			i++
		default:
			start := i
			for i < len(src) && isExprWordByte(src[i]) {
				i++
			}
			if start == i {
				return nil, fmt.Errorf("chat: unexpected %q in expression %q", string(c), src)
			}
			toks = append(toks, token{text: src[start:i]})
		}
	}
	if len(toks) == 0 {
		return nil, fmt.Errorf("chat: empty expression")
	}
	return toks, nil
}

// isExprWordByte reports whether c can be part of a bare word: a name, a number, or a
// keyword. Booleans and operators spelled as words go through here too, which is why it
// accepts letters, digits, and underscore.
func isExprWordByte(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

// lexString reads one quoted literal and returns its decoded value and byte length.
func lexString(src string) (string, int, error) {
	quote := src[0]
	var b strings.Builder
	for i := 1; i < len(src); i++ {
		switch src[i] {
		case quote:
			return b.String(), i + 1, nil
		case '\\':
			if i+1 >= len(src) {
				return "", 0, fmt.Errorf("chat: string ends with a backslash")
			}
			i++
			switch src[i] {
			case 'n':
				b.WriteByte('\n')
			case 't':
				b.WriteByte('\t')
			case 'r':
				b.WriteByte('\r')
			case '\\':
				b.WriteByte('\\')
			case '\'':
				b.WriteByte('\'')
			case '"':
				b.WriteByte('"')
			default:
				// An unknown escape keeps the backslash, which is what Jinja does and
				// what a template containing a Windows path needs.
				b.WriteByte('\\')
				b.WriteByte(src[i])
			}
		default:
			b.WriteByte(src[i])
		}
	}
	return "", 0, fmt.Errorf("chat: unterminated string literal")
}

// peek returns the current token, or "" at end of input.
func (p *exprParser) peek() string {
	if p.pos >= len(p.toks) {
		return ""
	}
	return p.toks[p.pos].text
}

// peekToken returns the current token and whether it is a string literal.
func (p *exprParser) peekToken() (token, bool) {
	if p.pos >= len(p.toks) {
		return token{}, false
	}
	return p.toks[p.pos], true
}

// accept consumes the current token when it matches, reporting whether it did.
func (p *exprParser) accept(tok string) bool {
	if p.peek() != tok {
		return false
	}
	p.pos++
	return true
}

// parseOr handles "or", the lowest-precedence operator.
// parseConditional handles Jinja's inline conditional, "a if cond else b".
//
// It sits outside the operator precedence chain because it binds loosest: the whole "a if cond
// else b" is one expression, and both arms are full expressions themselves. Gemma 3's template
// uses it to emit its system prefix only on the first turn,
//
//	{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
//
// which is the difference between one system prompt in the conversation and one before every
// turn. Without it the template does not parse and the whole thing falls back to the neutral
// transcript.
//
// The else arm is required: Jinja allows "a if cond" alone, which yields undefined otherwise, and
// no template here relies on an undefined result, so the shorter form is reported rather than
// guessed at.
func (p *exprParser) parseConditional() (expr, error) {
	then, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	if !p.accept("if") {
		return then, nil
	}
	cond, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	if !p.accept("else") {
		return nil, fmt.Errorf("chat: an inline 'if' needs 'else': Jinja's shorter form yields an " +
			"undefined value, which no template here relies on")
	}
	otherwise, err := p.parseConditional()
	if err != nil {
		return nil, err
	}
	return condExpr{cond: cond, then: then, otherwise: otherwise}, nil
}

func (p *exprParser) parseOr() (expr, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for p.accept("or") {
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		left = binaryExpr{op: "or", left: left, right: right}
	}
	return left, nil
}

// parseAnd handles "and".
func (p *exprParser) parseAnd() (expr, error) {
	left, err := p.parseNot()
	if err != nil {
		return nil, err
	}
	for p.accept("and") {
		right, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		left = binaryExpr{op: "and", left: left, right: right}
	}
	return left, nil
}

// parseNot handles a leading "not".
//
// It binds looser than the comparison layer, so "not x is defined" is "not (x is
// defined)" -- which is what templates write and what Jinja means.
func (p *exprParser) parseNot() (expr, error) {
	if p.accept("not") {
		inner, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		return notExpr{inner: inner}, nil
	}
	return p.parseCompare()
}

// parseCompare handles ==, !=, and "in".
func (p *exprParser) parseCompare() (expr, error) {
	left, err := p.parseConcat()
	if err != nil {
		return nil, err
	}
	for {
		op := p.peek()
		switch op {
		case "==", "!=", "<", ">", "<=", ">=":
			p.pos++
			right, err := p.parseConcat()
			if err != nil {
				return nil, err
			}
			left = binaryExpr{op: op, left: left, right: right}
		case "in":
			p.pos++
			right, err := p.parseConcat()
			if err != nil {
				return nil, err
			}
			left = binaryExpr{op: "in", left: left, right: right}
		case "is":
			p.pos++
			// "is not ..." only appears with none in practice, and negating it here keeps
			// the negation next to the test it applies to.
			negate := p.accept("not")
			if p.accept("none") {
				left = isNoneExpr{operand: left, negate: negate}
				continue
			}
			if !negate && p.accept("mapping") {
				left = isMappingExpr{operand: left}
				continue
			}
			if !negate && p.accept("iterable") {
				left = isIterableExpr{operand: left}
				continue
			}
			// "is string" is how the Gemma 3 template tells a plain content string from the
			// multimodal list of parts. Without it the whole template falls back to the neutral
			// transcript, so every Gemma 3 conversation loses the turn markers the model was
			// trained on.
			if !negate && p.accept("string") {
				left = isStringExpr{operand: left}
				continue
			}
			// "is true" and "is false" test truth rather than equality with the literal: in
			// Jinja2 `1 is true` is False because the test asks whether the value *is* the
			// boolean, and Qwen3's template uses `flag is false` on a value that came from
			// namespace(), which is always a boolean here.
			if !negate && p.accept("true") {
				left = isTrueExpr{operand: left, want: true}
				continue
			}
			if !negate && p.accept("false") {
				left = isTrueExpr{operand: left, want: false}
				continue
			}
			if negate {
				// "is not true" / "is not false" invert the same test rather than needing a
				// second node type, which is how `is not none` is handled below.
				if p.accept("true") {
					left = isTrueExpr{operand: left, want: false}
					continue
				}
				if p.accept("false") {
					left = isTrueExpr{operand: left, want: true}
					continue
				}
				// `is not defined` is the negation of a test this package already has, so it is
				// wrapped in notExpr rather than given a fourth node type. DeepSeek-R1's template
				// writes it twice to decide whether a tool-call id is present, and without it the
				// whole 2937-character template was refused.
				if p.accept("defined") {
					inner, err := p.definedTest(left)
					if err != nil {
						return nil, err
					}
					left = notExpr{inner: inner}
					continue
				}
				return nil, fmt.Errorf("chat: only 'is not none', 'is not true', 'is not false' " +
					"and 'is not defined' are supported after 'is not'")
			}
			if !p.accept("defined") {
				// The list has to match what the switch above accepts, or a reader whose
				// template wrote `is true` is told it is unsupported. It said "is defined,
				// is not defined, is none, is mapping, is iterable and is string" -- six names,
				// missing two the parser takes and three the message's own neighbour mentions.
				return nil, fmt.Errorf("chat: unsupported test after 'is'. Supported: defined, " +
					"is not defined, none, is not none, mapping, iterable, string, true, is not " +
					"true, false, is not false")
			}
			// "x is defined" and "x.y is defined" are both used: the first asks
			// whether a name is bound, the second whether a mapping has a key.
			left, err = p.definedTest(left)
			if err != nil {
				return nil, err
			}
		default:
			return left, nil
		}
	}
}

// definedTest builds the node that answers "is this name, field or subscript present".
//
// Shared by `is defined` and `is not defined` so the two cannot drift: the negation wraps this
// result rather than re-deriving it, and a name that is not a variable, a field or a subscript
// is refused in one place instead of two.
func (p *exprParser) definedTest(left expr) (expr, error) {
	switch v := left.(type) {
	case varExpr:
		return definedExpr(v), nil
	case attrExpr:
		return attrDefinedExpr(v), nil
	case indexExpr:
		// message['content'] is defined, which Gemma 3's template uses after the string and
		// iterable branches to reject a content value that is neither.
		return indexDefinedExpr{index: v}, nil
	default:
		return nil, fmt.Errorf("chat: 'is defined' needs a variable, field or subscript name")
	}
}

// parseConcat handles + between operands.
//
// Two integers add; anything else concatenates by text. That matches Jinja2 for both shapes: it
// adds 1 + 2, and its templates rely on the addition -- Qwen2.5's tool branch ranges over the
// messages with `messages[loop.index0 + 1]`, which only reaches the right element if the operand
// is a number. The evaluator concatenated every operand regardless, which rendered that subscript
// as the string "01" and made the template index a missing key.
//
// Whether an operand is a number is decided from its text, not from where it came from. Every value
// in this package is text once it comes from a template variable or a loop index, so there is no
// type bit to consult, and "5" + 1 is 6 here exactly as 5 + 1 is. Jinja2 raises TypeError for the
// quoted form; the difference only shows up for a literal digit string, and the alternative -- a
// flag set by the lexer -- would make + depend on whether the digits were quoted, which is not a
// distinction any template draws.
//
// Concatenation stays the answer when either side is not a number, because that is what every
// template in the corpus actually writes: {{ "user: " + m.content }}, a marker plus a loop index,
// a prefix plus a field. Jinja2 raises a TypeError for "a" + 1 instead; rendering it as text here
// is the documented divergence and the one real templates depend on, since a prompt half-rendered
// and then abandoned is worse than one whose parts were simply joined.
func (p *exprParser) parseConcat() (expr, error) {
	first, err := p.parseArith()
	if err != nil {
		return nil, err
	}
	parts := []expr{first}
	for p.accept("+") {
		next, err := p.parseArith()
		if err != nil {
			return nil, err
		}
		parts = append(parts, next)
	}
	if len(parts) == 1 {
		return parts[0], nil
	}
	return concatExpr{parts: parts}, nil
}

// parseArith handles - between operands.
func (p *exprParser) parseArith() (expr, error) {
	left, err := p.parseMul()
	if err != nil {
		return nil, err
	}
	for p.accept("-") {
		right, err := p.parseMul()
		if err != nil {
			return nil, err
		}
		left = arithExpr{op: "-", left: left, right: right}
	}
	return left, nil
}

// parseMul handles '%', which binds tighter than '-' as it does in Jinja.
//
// The level exists for one operator because one template needed it: Gemma's alternation check
// is "loop.index0 % 2 == 0". "*", "/" and "//" sit at the same precedence in Jinja and are not
// implemented, because no template this package renders uses them -- adding an operator is not
// free, since each one is another way for a template to compile and then mean something else.
func (p *exprParser) parseMul() (expr, error) {
	left, err := p.parsePostfix()
	if err != nil {
		return nil, err
	}
	for p.accept("%") {
		right, err := p.parsePostfix()
		if err != nil {
			return nil, err
		}
		left = arithExpr{op: "%", left: left, right: right}
	}
	return left, nil
}

// parseBracket reads what follows a "[" up to its "]" and builds either an index or a slice.
//
// Python writes three forms inside one pair of brackets and this reads all of them with one
// scanner rather than a branch per shape:
//
//	[i]      index
//	[a:b]    slice, either bound optional
//	[a:b:c]  slice with a step, any of the three optional: [:], [::2], [::-1], [3:0:-1]
//
// Which one it is falls out of which colons are seen: no colon is an index, one is a slice with
// step 1, two is a slice whose third field is the step. Splitting the shapes into separate
// branches is what the previous version did, and it left `[:]` and `[::-1]` unreachable because
// each branch expected a different token next.
//
// The base is passed in rather than read from the receiver because a bracket may follow any
// postfix chain: `messages[1:][0]` is two of these.
func (p *exprParser) parseBracket(base expr) (expr, error) {
	// field reads one optional bound. A missing one is nil, which is how an absent bound stays
	// distinct from a bound that happened to evaluate to 0 -- the two mean different things.
	//
	// The stop set is checked against the peeked token rather than the parser position: an
	// absent field is one where the next token is already a separator, and reading it would
	// report "unexpected "]"" for the legal form `messages[:]`.
	field := func() (expr, error) {
		switch p.peek() {
		case ":", "]", "":
			return nil, nil
		}
		return p.parseOr()
	}

	first, err := field()
	if err != nil {
		return nil, err
	}
	if !p.accept(":") {
		if first == nil {
			return nil, fmt.Errorf("chat: empty index")
		}
		if !p.accept("]") {
			return nil, fmt.Errorf("chat: missing ']' in index expression")
		}
		return indexExpr{base: base, index: first}, nil
	}

	// A colon was consumed, so this is a slice. Everything after it is optional.
	s := sliceExpr{base: base}
	second, err := field()
	if err != nil {
		return nil, err
	}
	hasStep := p.accept(":")
	if hasStep {
		var step expr
		if step, err = field(); err != nil {
			return nil, err
		}
		s.step = step
	}
	// Position decides what each field meant, which is why the assignment happens after the
	// scanning rather than during it: [a:b:c] and [:b:c] reach the same place with one fewer
	// field read, and both have the step last.
	if !hasStep {
		s.from, s.to = first, second
	} else if first != nil || second != nil {
		s.from, s.to = first, second
	}
	if !p.accept("]") {
		return nil, fmt.Errorf("chat: missing ']' in slice expression")
	}
	return s, nil
}

// parsePostfix handles attribute access, indexing, and filters after a primary.
func (p *exprParser) parsePostfix() (expr, error) {
	e, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	for {
		switch {
		case p.accept("."):
			name := p.peek()
			if !isIdentifier(name) {
				return nil, fmt.Errorf("chat: expected a field name after '.'")
			}
			p.pos++
			e = attrExpr{base: e, name: name}
			// A "(" straight after a field name is a method call: `s.strip()`, `s.split(x)`.
			// Qwen3's template uses four of them to trim the reasoning block out of a reply.
			// It is parsed here rather than in parsePrimary because the receiver is the
			// expression to the left, not a name in scope.
			if next, ok := p.peekToken(); ok && next.text == "(" {
				call, err := p.parseCall(name)
				if err != nil {
					return nil, err
				}
				cc := call.(callExpr)
				cc.recv = e.(attrExpr).base
				e = cc
			}
		case p.accept("["):
			e, err = p.parseBracket(e)
			if err != nil {
				return nil, err
			}
		case p.accept("|"):
			name := p.peek()
			if !isIdentifier(name) {
				return nil, fmt.Errorf("chat: expected a filter name after '|'")
			}
			p.pos++
			if !supportedFilters[name] {
				return nil, fmt.Errorf("chat: unsupported template filter %q", name)
			}
			fe := filterExpr{operand: e, name: name}
			// A filter may take one parenthesised argument: trim has none, tojson
			// has none, and selectattr-style filters are outside the subset.
			if p.accept("(") {
				for {
					// name=value is a keyword argument; anything else is the single
					// positional one. The lookahead decides before consuming, so a plain
					// argument is not mistaken for a name.
					if name, value, ok, err := p.parseKwarg(); err != nil {
						return nil, err
					} else if ok {
						if fe.kwargs == nil {
							fe.kwargs = make(map[string]expr, 1)
						}
						fe.kwargs[name] = value
					} else {
						arg, err := p.parseOr()
						if err != nil {
							return nil, err
						}
						fe.arg = arg
					}
					if p.accept(")") {
						break
					}
					if !p.accept(",") {
						return nil, fmt.Errorf("chat: missing ')' in filter arguments")
					}
				}
			}
			e = fe
		default:
			return e, nil
		}
	}
}

// parseKwarg consumes "name = expr" when that is what comes next, and reports whether it
// did so. It looks ahead before consuming anything so a positional argument still parses.
func (p *exprParser) parseKwarg() (string, expr, bool, error) {
	first, ok := p.peekToken()
	if !ok || !isIdentifier(first.text) {
		return "", nil, false, nil
	}
	if p.pos+1 >= len(p.toks) || p.toks[p.pos+1].text != "=" {
		return "", nil, false, nil
	}
	p.pos += 2 // name and '='
	v, err := p.parseOr()
	if err != nil {
		return "", nil, false, err
	}
	return first.text, v, true, nil
}

// parsePrimary handles literals, names, and parenthesised expressions.
func (p *exprParser) parsePrimary() (expr, error) {
	// The token's text is not enough to tell "no more tokens" from an empty string literal,
	// which is a real value templates use -- {% set system_message = "" %} among them. The
	// position decides.
	if p.pos >= len(p.toks) {
		return nil, fmt.Errorf("chat: expression ends early")
	}
	tok := p.peek()
	if p.toks[p.pos].isString {
		p.pos++
		return litExpr{val: tok}, nil
	}
	if tok == "(" {
		p.pos++
		// The grouping parses a whole expression, conditional included: Gemma 3 writes
		// (first_user_prefix if loop.first else ""), and parseOr alone stops at "if" and then
		// reports a missing parenthesis.
		inner, err := p.parseConditional()
		if err != nil {
			return nil, err
		}
		if !p.accept(")") {
			return nil, fmt.Errorf("chat: missing ')'")
		}
		return inner, nil
	}
	cur, _ := p.peekToken()
	p.pos++
	if cur.isString {
		return litExpr{val: cur.text}, nil
	}
	if tok == "-" {
		// A leading minus belongs to the number that follows, so it is folded here
		// rather than becoming a unary node. Only integers can be negated in this
		// subset, which is all a template index needs.
		next, ok := p.peekToken()
		if !ok {
			return nil, fmt.Errorf("chat: expression ends after '-'")
		}
		p.pos++
		return litExpr{val: "-" + next.text, isNumber: true}, nil
	}
	switch tok {
	case "true", "True":
		// A boolean literal, not the string "true": as a literal it is non-empty, which
		// makes every condition using it true and turns `{% if false %}` into an
		// always-taken branch.
		return boolLitExpr{val: true}, nil
	case "false", "False":
		return boolLitExpr{val: false}, nil
	case "none", "None":
		return litExpr{val: "", isNone: true}, nil
	}
	if n, err := strconv.Atoi(tok); err == nil {
		return litExpr{val: strconv.Itoa(n), isNumber: true}, nil
	}
	if !isIdentifier(tok) {
		return nil, fmt.Errorf("chat: unexpected %q in expression", tok)
	}
	// name(...) is a call. Only strftime_now exists in Jinja's own function set, and this
	// package implements none of them, so a call parses but cannot be evaluated. Parsing it
	// anyway is what lets a template that guards the call -- "if strftime_now is defined" --
	// be used at all: the guard is false here, the branch never runs, and rejecting the
	// whole template over a construct that cannot execute would be stricter than the
	// template requires.
	if next, ok := p.peekToken(); ok && next.text == "(" {
		return p.parseCall(tok)
	}
	return varExpr{name: tok}, nil
}
