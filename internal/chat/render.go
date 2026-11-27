package chat

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Message is one conversation turn, as a template sees it.
//
// It is a plain struct rather than a map because a template reads fields by name and a
// struct can be converted once at the boundary; keeping the public shape typed is what
// lets a caller pass a message list without knowing a template exists.
type Message struct {
	Role    string
	Content string
	// ToolCalls are the calls the assistant made in this turn, empty for every other role.
	//
	// A template renders them back into the conversation so the model sees the call it made
	// before it reads the result; the Qwen and Llama 3.1 formats both write the call as JSON
	// inside the assistant turn, and neither has a separate message type for it.
	ToolCalls []ToolCall
}

// ToolCall is one call an assistant turn carries.
//
// Arguments is the JSON text the model produced, kept verbatim rather than parsed: a template
// prints it with tojson, and re-encoding a decoded map would reorder keys and change the
// prompt from what the model wrote. A caller that needs the fields decodes it itself.
type ToolCall struct {
	// ID is the identifier the response assigned, or empty when the conversation did not
	// supply one -- a hand-written conversation, or a model format that numbers calls
	// itself.
	ID string
	// Name is the function the model asked to call.
	Name string
	// Arguments is the call's arguments as a JSON object.
	Arguments string
}

// Template is a compiled chat template. It is safe for concurrent use.
//
// Nothing in it changes after Parse: the node tree and the source text are read-only, and
// Render builds a fresh scope and buffer per call. That matters because the callers share
// one Template across goroutines -- engine.ChatRenderer is built once per model and the
// HTTP server renders from every request handler -- so a field added here that a render
// writes to would be a data race in production and nowhere in the tests.
//
// TestRenderIsSafeForConcurrentUse pins it. The property is a claim about the shape of the
// type rather than about any one output, so the test renders from many goroutines and runs
// under -race; a shared buffer or a cached scope fails it under the detector rather than on
// a value.
type Template struct {
	nodes []node
	// source is the original template text, kept for error messages so a failure
	// names the template that produced it rather than only the construct.
	source string
	// reads is every name the template can look up, collected once at parse time so Reads is a
	// map lookup rather than a tree walk per request. Read-only after Parse, like the rest of
	// the type, which is what keeps Render safe to call from many goroutines.
	reads map[string]bool
}

// Parse compiles a template.
//
// The error names the construct this package does not implement, because the caller's
// response to one is to fall back to a neutral transcript. A template that parsed here
// evaluates without further surprises: every construct is checked at compile time, so a
// render cannot fail halfway through on an unsupported statement.
func Parse(source string) (*Template, error) {
	pieces, err := lex(source)
	if err != nil {
		return nil, err
	}
	nodes, err := parse(pieces)
	if err != nil {
		return nil, err
	}
	reads := make(map[string]bool)
	collectReads(nodes, reads)
	return &Template{nodes: nodes, source: source, reads: reads}, nil
}

// Inputs is what a chat template reads from its environment beyond the message list.
//
// Templates also read a few names the tokenizer supplies: bos_token and eos_token are
// the vocabulary's special pieces as text, and add_generation_prompt selects whether the
// render ends with an open assistant turn. They are here rather than assumed because a
// template that omitted them would produce a prompt without its BOS, which is a silent
// change to the input.
type Inputs struct {
	// BOSToken and EOSToken are the special pieces as the template should spell them.
	BOSToken string
	EOSToken string
	// AddGenerationPrompt opens the assistant's turn. A template that ignores it still
	// renders, which is why the caller must pass it rather than rely on the default.
	AddGenerationPrompt bool
	// Tools is the tool list, or nil. Present so a template's tools branch can be
	// taken; this package does not implement tool calling itself.
	Tools []Tool
	// Variables are extra names the caller wants the template to see.
	//
	// This is where a checkpoint's own switches live. Qwen3's template branches its prompt on
	// enable_thinking, and a template that reads a name nobody bound takes its else branch
	// without saying so -- so a caller with no way to set one gets a prompt for the default
	// mode only. Before this field existed, /v1/chat/completions accepted the OpenAI-compatible
	// (vLLM-spelled) chat_template_kwargs member and ignored it: the reply was well formed and
	// nothing distinguished it from the one that was asked for.
	//
	// The values are what a JSON body decodes to -- string, bool, nil, float64, json.Number,
	// int, int64, []any, map[string]any -- because a request's variables come from JSON. See
	// variableValue for how each becomes a renderable value.
	//
	// A name this package binds itself is refused rather than shadowed; see reservedVariables.
	// A name the template never reads is not an error here, because whether that is a problem
	// depends on the caller: Template.Reads answers the question, and internal/server turns it
	// into a refusal with the field name a client can act on.
	Variables map[string]any
}

// Tool is one entry of the tools list.
type Tool struct {
	Name        string
	Description string
	// Parameters is the JSON schema, as text, because a template only ever prints it.
	Parameters string
}

// Render evaluates the template for a conversation.
func (t *Template) Render(msgs []Message, in Inputs) (string, error) {
	// tools is bound only when the caller passed some. A template tests it both ways --
	// `{% if tools %}` and `{% if not tools is defined %}` -- and those two only agree if
	// an absent tools list is genuinely absent. Binding an empty list made the second test
	// always false, so the branch that sets tools = none never ran and every conversation
	// took the tool-calling path: Llama 3's template rendered a prompt asking the model to
	// answer with a JSON function call, with no tools provided.
	//
	// A condition tolerates an unbound name (see evalCond), so `{% if tools %}` still takes
	// the no-tools branch instead of failing.
	vars := map[string]value{
		"messages":              messagesValue(msgs),
		"bos_token":             strValue(in.BOSToken),
		"eos_token":             strValue(in.EOSToken),
		"add_generation_prompt": boolValue(in.AddGenerationPrompt),
	}
	if len(in.Tools) > 0 {
		vars["tools"] = toolsValue(in.Tools)
	}
	// The caller's variables are bound last, and the names the map above holds are refused
	// rather than overwritten: a variable named messages would replace the conversation with
	// whatever else the caller put in the same request, and one named add_generation_prompt
	// would close the assistant turn the template exists to open. Both produce a plausible
	// prompt for a request nobody made, which is the failure this package is written against.
	//
	// Iterated in sorted order so a request carrying two bad names always reports the same one:
	// an error that changes between identical requests is one a client cannot reproduce.
	for _, name := range slices.Sorted(maps.Keys(in.Variables)) {
		if reservedVariables[name] {
			return "", fmt.Errorf("chat: variable %q is bound by the renderer and cannot be set "+
				"as a template variable", name)
		}
		v, err := variableValue(in.Variables[name])
		if err != nil {
			return "", fmt.Errorf("chat: variable %q: %w", name, err)
		}
		vars[name] = v
	}
	sc := newScope(vars)
	var r renderer
	if err := r.renderNodes(t.nodes, sc); err != nil {
		return "", fmt.Errorf("chat: render: %w", err)
	}
	return r.buf.String(), nil
}

// messagesValue converts turns into the list a template iterates.
//
// tool_calls is present only when the turn made some. The two real templates that read it ask
// opposite questions of an absent field: Qwen2 writes `not message.tool_calls` and Llama 3 writes
// `'tool_calls' in message` to mean "this turn has calls". A field that is always present answers
// one of them wrongly whichever way it is spelled -- bound to an empty list, Qwen2's guard is
// satisfied but Llama 3's `in` test is true and its template raises "This model only supports
// single tool-calls at once!" on every plain conversation; omitted, Qwen2 is right and Llama 3 is
// right. Omitting is what Jinja2 does for a dict that has no such key, which is the behaviour both
// templates were written against.
//
// Reading an absent field does not fail here: a missing key evaluates as none, which doc.go
// records as one of the deliberate divergences from Jinja2 and is what makes `not message.tool_calls`
// true rather than an error.
func messagesValue(msgs []Message) value {
	items := make([]value, len(msgs))
	for i, m := range msgs {
		fields := map[string]value{
			"role":    strValue(m.Role),
			"content": strValue(m.Content),
		}
		if len(m.ToolCalls) > 0 {
			calls := make([]value, len(m.ToolCalls))
			for j, c := range m.ToolCalls {
				calls[j] = value{isMap: true, fields: map[string]value{
					"id":   strValue(c.ID),
					"type": strValue("function"),
					"function": {isMap: true, fields: map[string]value{
						"name": strValue(c.Name),
						// Verbatim: the template prints this with tojson, and a re-encoded map
						// would reorder the model's own keys and change the prompt.
						"arguments": rawJSONValue(c.Arguments),
					}},
				}}
			}
			fields["tool_calls"] = listValue(calls)
		}
		items[i] = value{isMap: true, fields: fields}
	}
	return listValue(items)
}

// toolsValue converts the tool list for a template's tools branch.
func toolsValue(tools []Tool) value {
	items := make([]value, len(tools))
	for i, t := range tools {
		// A template prints a tool with tojson, so the shape here is the JSON shape a
		// template expects: a function object with name, description, and parameters.
		fn := map[string]value{
			"name":        strValue(t.Name),
			"description": strValue(t.Description),
			"parameters":  rawJSONValue(t.Parameters),
		}
		items[i] = value{isMap: true, fields: map[string]value{
			"type":     strValue("function"),
			"function": {isMap: true, fields: fn},
		}}
	}
	return listValue(items)
}

// rawJSONValue marks text that tojson should emit verbatim rather than quote.
func rawJSONValue(s string) value {
	return value{text: s, isMap: false, isList: false, isRawJSON: s != ""}
}

// toJSON renders a value as JSON, which is what a template's tojson filter produces.
//
// The output matches Jinja2's, because that is what a checkpoint's template was written and
// tested against: Python's json.dumps with sort_keys=True, and the default separators
// (", " and ": ") unless an indent was asked for. The spacing is part of the prompt text, so
// compact JSON here would hand the model a tool call spelled differently from the one it was
// trained on -- harmless-looking, and a difference the model may well act on.
func toJSON(v value) string {
	var b strings.Builder
	writeJSON(&b, v, jsonStyle{})
	return b.String()
}

// toJSONIndent renders a value as JSON with an indent of n spaces per level, which is what
// "tojson(indent=n)" produces. A non-positive indent means the default style.
func toJSONIndent(v value, indent int) string {
	var b strings.Builder
	writeJSON(&b, v, jsonStyle{indent: indent})
	return b.String()
}

// jsonStyle carries the formatting choices Python's json.dumps exposes.
type jsonStyle struct {
	// indent is spaces per nesting level; 0 means compact with default separators.
	indent int
}

// level renders the indentation for a depth, or the empty string when compact.
func (st jsonStyle) level(depth int) string {
	if st.indent <= 0 {
		return ""
	}
	return "\n" + strings.Repeat(" ", st.indent*depth)
}

// itemSep is what goes between sequence items.
func (st jsonStyle) itemSep(depth int) string {
	if st.indent > 0 {
		return "," + st.level(depth)
	}
	return ", "
}

// keySep is what goes between a key and its value, which Python writes the same way in both
// styles.
func (st jsonStyle) keySep() string { return ": " }

// closeSep is what goes before the closing bracket of a non-empty container.
func (st jsonStyle) closeSep(depth int) string {
	if st.indent > 0 {
		return st.level(depth)
	}
	return ""
}

// writeJSON writes one value as JSON at the given depth.
func writeJSON(b *strings.Builder, v value, st jsonStyle) {
	writeJSONDepth(b, v, st, 0)
}

// writeJSONDepth writes one value as JSON, tracking nesting so indentation is right.
func writeJSONDepth(b *strings.Builder, v value, st jsonStyle, depth int) {
	switch {
	case v.isRawJSON:
		// Already JSON; emitting it quoted would double-encode the schema.
		b.WriteString(v.text)
	case v.isMap:
		if len(v.fields) == 0 {
			b.WriteString("{}")
			return
		}
		keys := make([]string, 0, len(v.fields))
		for k := range v.fields {
			keys = append(keys, k)
		}
		// Sorted, as Python's sort_keys=True does: a prompt whose key order changed between
		// runs would change the tokens.
		sortStrings(keys)
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteString(st.itemSep(depth + 1))
			} else {
				b.WriteString(st.level(depth + 1))
			}
			writeJSONString(b, k)
			b.WriteString(st.keySep())
			writeJSONDepth(b, v.fields[k], st, depth+1)
		}
		b.WriteString(st.closeSep(depth))
		b.WriteByte('}')
	case v.isList:
		if len(v.list) == 0 {
			b.WriteString("[]")
			return
		}
		b.WriteByte('[')
		for i, item := range v.list {
			if i > 0 {
				b.WriteString(st.itemSep(depth + 1))
			} else {
				b.WriteString(st.level(depth + 1))
			}
			writeJSONDepth(b, item, st, depth+1)
		}
		b.WriteString(st.closeSep(depth))
		b.WriteByte(']')
	case v.isBool:
		b.WriteString(v.text)
	default:
		writeJSONString(b, v.text)
	}
}

// writeJSONString writes a quoted, escaped JSON string.
//
// Everything above U+007F is written as a \uXXXX escape, which is what Python's json.dumps
// does by default (ensure_ascii=True) and therefore what Jinja2's tojson filter produces.
// Go's json.Marshal leaves the text as UTF-8 instead, so a tool definition with a Chinese
// description reached the model spelled differently from the one its template was written
// against -- the separators were already matched for the same reason.
//
// A code point above U+FFFF becomes a surrogate pair, again as Python writes it: "🎉" is
// \ud83c\udf89 and not \U0001f389.
func writeJSONString(b *strings.Builder, s string) {
	raw, err := json.Marshal(s)
	if err != nil {
		// Marshalling a string cannot fail, but a silent fallback would hide it.
		b.WriteString(`""`)
		return
	}
	for _, r := range string(raw) {
		switch {
		case r < utf8.RuneSelf:
			b.WriteRune(r)
		case r <= 0xFFFF:
			fmt.Fprintf(b, `\u%04x`, r)
		default:
			r -= 0x10000
			hi := 0xD800 + (r >> 10)
			lo := 0xDC00 + (r & 0x3FF)
			fmt.Fprintf(b, `\u%04x\u%04x`, hi, lo)
		}
	}
}

// sortStrings sorts in place; kept local so the file needs no import for one call.
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// reservedVariables are the names Render binds itself, and a caller may not supply.
//
// Refused rather than merged, because each one would change something the request already
// states: messages is the conversation the client sent, add_generation_prompt opens the
// assistant turn, and bos_token/eos_token are the vocabulary's own pieces. A variable of any of
// those names produces a prompt for a request nobody made -- well formed, and wrong.
//
// tools is on the list for a narrower reason: it is bound from Inputs.Tools, and a variable of
// that name would let a caller take a template's tool-calling branch without sending a tool list
// anything could act on.
var reservedVariables = map[string]bool{
	"messages":              true,
	"bos_token":             true,
	"eos_token":             true,
	"add_generation_prompt": true,
	"tools":                 true,
}

// variableValue converts one caller-supplied variable into a renderable value.
//
// The accepted types are the ones a JSON body decodes to, plus the string and integer shapes a
// Go caller reaches for first. Numbers keep their spelling: templates compare and print
// integers, and a 3 that became "3.0" would change the prompt.
//
// A type outside that set is an error rather than a string conversion. A struct or a pointer
// reaching a template means the caller expected the renderer to know something about it, and
// fmt's %v of it is not that thing.
func variableValue(v any) (value, error) {
	switch x := v.(type) {
	case nil:
		// An explicit null prints as nothing and is false in a condition. It is still bound, so
		// `x is defined` is true for it and `x is none` is true as well -- which is the same pair
		// of answers Jinja2 gives a None value, and a template that guards a switch with both gets
		// the same branch here.
		//
		// isNone has to be set for the second half of that to hold. Without it this returned a
		// value indistinguishable from the empty string, so `x is none` was False for a JSON null
		// while the comment claimed True -- measured before the fix: a request with
		// `chat_template_kwargs: {"x": null}` rendered the else branch of
		// `{% if x is none %}`. The empty string is a real value a caller can pass, and a template
		// distinguishing absent from empty would have taken the wrong arm either way.
		v := strValue("")
		v.isNone = true
		return v, nil
	case string:
		return strValue(x), nil
	case bool:
		return boolValue(x), nil
	case json.Number:
		return strValue(x.String()), nil
	case int:
		return strValue(strconv.Itoa(x)), nil
	case int64:
		return strValue(strconv.FormatInt(x, 10)), nil
	case float64:
		return strValue(renderNumber(x)), nil
	case []any:
		items := make([]value, len(x))
		for i, item := range x {
			iv, err := variableValue(item)
			if err != nil {
				return value{}, fmt.Errorf("element %d: %w", i, err)
			}
			items[i] = iv
		}
		return listValue(items), nil
	case map[string]any:
		fields := make(map[string]value, len(x))
		for k, item := range x {
			iv, err := variableValue(item)
			if err != nil {
				return value{}, fmt.Errorf("field %q: %w", k, err)
			}
			fields[k] = iv
		}
		return value{fields: fields, isMap: true}, nil
	default:
		return value{}, fmt.Errorf("a %T cannot be a template variable; use a string, a "+
			"boolean, a number, null, or a list or object of those", v)
	}
}

// renderNumber prints a number the way Jinja2 would print the value the same JSON produces.
//
// The name says what it is for rather than what type it takes: internal/server has a
// formatFloat too, which writes Prometheus exposition text, and two functions with one name
// doing two jobs is how a reader picks the wrong one to change.
//
// That is not obvious, and it is measured rather than assumed. encoding/json gives this package
// a float64 for every number in the request body, but Jinja2 receives what json.loads produced
// in Python: 3 becomes an int and prints "3", while 3.0 becomes a float and prints "3.0". So the
// formatting has to look at the value rather than at the Go type, or every integer-valued kwarg
// reaches the prompt with a spurious decimal point.
//
// Measured against jinja2 3.1.6: `{{ n }}` renders "3" for n=3, "3.0" for n=3.0, and "True" for
// n=True. The bool case never reaches here -- variableValue maps it to boolValue, which print()
// renders as Jinja does.
func renderNumber(f float64) string {
	if f == math.Trunc(f) && math.Abs(f) < 1e15 {
		// Whole and inside the range where float64 represents integers exactly, so it came from
		// a JSON integer and Jinja2 would have held an int.
		return strconv.FormatInt(int64(f), 10)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// Reads reports whether the template can look the name up anywhere -- in output, in a
// condition, or inside a branch this conversation does not take.
//
// Callers use it to refuse a variable that would do nothing. The alternative is silence: a
// template that never names enable_thinking renders the identical prompt whether the caller sent
// it or not, and nothing in the response says which happened.
//
// The answer is static, so it covers every branch: a name read only inside the tools branch
// counts as read even when no tools were passed, because the question is about the template
// rather than about one conversation.
func (t *Template) Reads(name string) bool {
	if t == nil {
		return false
	}
	return t.reads[name]
}

// collectReads records every name the node tree can look up.
//
// Three node forms read a name -- varExpr for output and conditions, definedExpr for `x is
// defined`, and isNoneExpr's variable operand -- and everything else reaches one of them through
// a base expression. A name only written by {% set %} is not collected: it is the template's own
// variable, and reporting it would let a caller think a value it never set reached the prompt.
func collectReads(nodes []node, into map[string]bool) {
	// bound holds the names the template binds for itself: a for loop's variable and anything
	// {% set %} assigns. They are removed at the end, because a caller cannot set them -- the
	// template overwrites the value -- so reporting one as read would tell a caller its value
	// reached the prompt when the template replaced it.
	bound := make(map[string]bool)
	var walkExpr func(e expr)
	var walkNodes func(ns []node)
	walkExpr = func(e expr) {
		switch x := e.(type) {
		case varExpr:
			into[x.name] = true
		case definedExpr:
			into[x.name] = true
		case isNoneExpr:
			walkExpr(x.operand)
		case attrExpr:
			walkExpr(x.base)
		case attrDefinedExpr:
			walkExpr(x.base)
		case indexExpr:
			walkExpr(x.base)
			walkExpr(x.index)
		case indexDefinedExpr:
			walkExpr(x.index.base)
			walkExpr(x.index.index)
		case sliceExpr:
			walkExpr(x.base)
			walkExpr(x.from)
		case concatExpr:
			for _, part := range x.parts {
				walkExpr(part)
			}
		case arithExpr:
			walkExpr(x.left)
			walkExpr(x.right)
		case binaryExpr:
			walkExpr(x.left)
			walkExpr(x.right)
		case notExpr:
			walkExpr(x.inner)
		case condExpr:
			walkExpr(x.cond)
			walkExpr(x.then)
			walkExpr(x.otherwise)
		case isMappingExpr:
			walkExpr(x.operand)
		case isIterableExpr:
			walkExpr(x.operand)
		case isStringExpr:
			walkExpr(x.operand)
		case callExpr:
			for _, arg := range x.args {
				walkExpr(arg)
			}
		case filterExpr:
			walkExpr(x.operand)
			if x.arg != nil {
				walkExpr(x.arg)
			}
			for _, kw := range x.kwargs {
				walkExpr(kw)
			}
		}
	}
	walkNodes = func(ns []node) {
		for _, n := range ns {
			switch x := n.(type) {
			case exprNode:
				walkExpr(x.expr)
			case setNode:
				// The expression side reads; the name side writes.
				walkExpr(x.expr)
				bound[x.name] = true
			case ifNode:
				for _, br := range x.branches {
					walkExpr(br.cond)
					walkNodes(br.body)
				}
				walkNodes(x.elseBody)
			case forNode:
				walkExpr(x.iter)
				bound[x.varName] = true
				walkNodes(x.body)
			}
		}
	}
	walkNodes(nodes)
	for name := range bound {
		delete(into, name)
	}
}
