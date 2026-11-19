package chat

import (
	"encoding/json"
	"fmt"
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
}

// Template is a compiled chat template.
type Template struct {
	nodes []node
	// source is the original template text, kept for error messages so a failure
	// names the template that produced it rather than only the construct.
	source string
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
	return &Template{nodes: nodes, source: source}, nil
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
	sc := newScope(vars)
	var r renderer
	if err := r.renderNodes(t.nodes, sc); err != nil {
		return "", fmt.Errorf("chat: render: %w", err)
	}
	return r.buf.String(), nil
}

// messagesValue converts turns into the list a template iterates.
func messagesValue(msgs []Message) value {
	items := make([]value, len(msgs))
	for i, m := range msgs {
		items[i] = value{
			isMap: true,
			fields: map[string]value{
				"role":    strValue(m.Role),
				"content": strValue(m.Content),
			},
		}
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
