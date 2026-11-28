package server

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"github.com/pplmx/orbix/internal/engine"
)

// Function calling, request side and response side.
//
// The two halves are inseparable, which is why they are one file. A server that renders tools
// into the prompt but never emits tool_calls answers a request with prose about a call it did not
// make; one that emits tool_calls without rendering tools answers as though the model knew the
// tools existed. The request-side fields used to be parsed only to be refused with a 400, and the
// message said so; what replaced that refusal is this file.
//
// The format is the model's, not this server's. Qwen2.5 asks for
//
//	<tool_call>
//	{"name": <name>, "arguments": <args>}
//	</tool_call>
//
// and Llama 3.1 for `{"name": …, "parameters": …}` inside a turn, and the two are different
// strings. The template renders the instructions, so the prompt half is exact; the response half
// parses the marker the model was told to use, which is why the set below is narrow and named.

// toolRequest is the request's tools member: the OpenAI shape, which every client sends.
//
// Only the three fields a chat template prints are kept. A caller may send more -- a strict
// schema flag, a vendor extension -- and dropping those is not the silent-ignore failure the rest
// of this package refuses: the template has no place to put them, and the tool signature the model
// reads is name, description and parameters.
type toolRequest struct {
	// Type is "function", which is the only kind OpenAI defines and the only one any template
	// here knows how to print. Anything else is refused by checkTools.
	Type string `json:"type"`
	// Function is nil when the member is malformed, which checkTools reports.
	Function *toolFunction `json:"function"`
}

type toolFunction struct {
	Name string `json:"name"`
	// Description is optional in the schema and empty is a real value.
	Description string `json:"description"`
	// Parameters is kept as raw JSON and re-emitted verbatim. Decoding it into a map and
	// encoding it again would sort its keys, and the signature the model reads would then differ
	// from the one the caller sent.
	Parameters json.RawMessage `json:"parameters"`
	// Strict is OpenAI's structured-outputs flag. It is refused rather than ignored when true,
	// because it promises the arguments match the schema exactly and that needs constrained
	// decoding: a caller relying on it would get arguments that parse but may not validate.
	Strict *bool `json:"strict"`
}

// toolChoice is the request's tool_choice, in the three shapes the schema defines.
//
// "auto", "none" and "required" are strings; a named function is an object. An omitted field
// means "auto" and a null means "no tools in play", which is what the four spellings in
// jsonAsks were about before the fields did anything.
type toolChoice struct {
	// Mode is "auto", "none", "required" or "named" after normalisation.
	Mode string
	// Name is the function a named choice selects.
	Name string
}

// modeAuto is what an absent tool_choice means.
const (
	modeAuto     = "auto"
	modeNone     = "none"
	modeRequired = "required"
	modeNamed    = "named"
)

// checkTools validates the tools member and returns the list a template can print.
//
// The refusals are about things a template or this server cannot honour, not about the shape of
// the caller's JSON -- those are decode errors and are named by the decoder.
func checkTools(raw json.RawMessage) ([]toolRequest, []engine.ToolSpec, error) {
	if !jsonAsks(raw) {
		return nil, nil, nil
	}
	var tools []toolRequest
	if err := json.Unmarshal(raw, &tools); err != nil {
		return nil, nil, &paramError{
			message: "tools must be an array of {\"type\": \"function\", \"function\": {...}} " +
				"objects: " + err.Error(),
			param: "tools", code: "invalid_value",
		}
	}
	if len(tools) == 0 {
		return nil, nil, nil
	}
	specs := make([]engine.ToolSpec, len(tools))
	seen := make(map[string]bool, len(tools))
	for i, t := range tools {
		field := fmt.Sprintf("tools[%d]", i)
		if t.Type != "function" {
			return nil, nil, &paramError{
				message: fmt.Sprintf("%s.type is %q; function is the only tool type the "+
					"chat schema defines and the only one a chat template can print, so a %q "+
					"tool would reach no prompt", field, t.Type, t.Type),
				param: field + ".type", code: "unsupported_parameter",
			}
		}
		if t.Function == nil {
			return nil, nil, &paramError{
				message: fmt.Sprintf("%s has no function member; a function tool is "+
					"{\"type\": \"function\", \"function\": {\"name\": ..., \"parameters\": ...}}",
					field),
				param: field + ".function", code: "missing_required_parameter",
			}
		}
		name := t.Function.Name
		if strings.TrimSpace(name) == "" {
			return nil, nil, &paramError{
				message: fmt.Sprintf("%s.function.name is required: the model calls a tool by "+
					"name, and an unnamed one could not be called", field),
				param: field + ".function.name", code: "missing_required_parameter",
			}
		}
		// Two tools with one name make every call to that name ambiguous, and the model has
		// no way to say which it meant.
		if seen[name] {
			return nil, nil, &paramError{
				message: fmt.Sprintf("%s.function.name %q is declared more than once; a call "+
					"to it would name both tools, so the reply could not be acted on", field, name),
				param: field + ".function.name", code: "invalid_value",
			}
		}
		seen[name] = true
		if t.Function.Strict != nil && *t.Function.Strict {
			return nil, nil, &paramError{
				message: fmt.Sprintf("%s.function.strict promises the arguments match the "+
					"schema exactly, which needs constrained decoding this engine does not "+
					"implement; drop the flag, or validate the arguments in your own code",
					field),
				param: field + ".function.strict", code: "unsupported_parameter",
			}
		}
		params := strings.TrimSpace(string(t.Function.Parameters))
		if params == "" || params == "null" {
			// A tool with no arguments is a real shape, and a template prints parameters as
			// {} in that case rather than as nothing, which is what OpenAI's own examples show.
			params = "{}"
		}
		specs[i] = engine.ToolSpec{
			Name:        name,
			Description: t.Function.Description,
			Parameters:  params,
		}
	}
	return tools, specs, nil
}

// checkToolChoice normalises tool_choice against the tools the request declared.
func checkToolChoice(raw json.RawMessage, tools []toolRequest) (toolChoice, error) {
	// "none" has to be read before jsonAsks, which answers false for it: the helper exists so an
	// empty form is treated as "asked for nothing", and for tool_choice the string "none" is
	// exactly that -- but it is also a *decision* this server has to act on, by withholding the
	// tool list from the prompt. Normalising it to auto left the tools in the prompt, so
	// tool_choice "none" behaved like "auto" while the documentation said the tools were
	// withheld; measured before this fix, mode was "auto" for a request that sent "none".
	if strings.TrimSpace(string(raw)) == `"none"` {
		return toolChoice{Mode: modeNone}, nil
	}
	if !jsonAsks(raw) {
		return toolChoice{Mode: modeAuto}, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case modeAuto, modeNone, modeRequired:
			return toolChoice{Mode: s}, nil
		default:
			return toolChoice{}, &paramError{
				message: fmt.Sprintf("tool_choice %q is not one of auto, none, required, or "+
					"{\"type\": \"function\", \"function\": {\"name\": ...}}", s),
				param: "tool_choice", code: "invalid_value",
			}
		}
	}
	var named struct {
		Type     string        `json:"type"`
		Function *toolFunction `json:"function"`
	}
	if err := json.Unmarshal(raw, &named); err != nil {
		return toolChoice{}, &paramError{
			message: "tool_choice must be \"auto\", \"none\", \"required\", or an object " +
				"naming one function: " + err.Error(),
			param: "tool_choice", code: "invalid_value",
		}
	}
	if named.Type != "function" || named.Function == nil || strings.TrimSpace(named.Function.Name) == "" {
		return toolChoice{}, &paramError{
			message: "a named tool_choice must be {\"type\": \"function\", \"function\": " +
				"{\"name\": ...}}, naming one of the tools in this request",
			param: "tool_choice", code: "invalid_value",
		}
	}
	name := named.Function.Name
	for _, t := range tools {
		if t.Function != nil && t.Function.Name == name {
			return toolChoice{Mode: modeNamed, Name: name}, nil
		}
	}
	// A name that is not in the list is refused rather than passed to the prompt: no template
	// here prints "call only this function", so honouring it would mean answering with whatever
	// the model chose while the caller believes the constraint held.
	return toolChoice{}, &paramError{
		message: fmt.Sprintf("tool_choice names %q, which is not one of the tools this request "+
			"declares; a template can only force a tool by printing its constraint, and this "+
			"server does not rewrite the prompt that way, so the choice would not be honoured",
			name),
		param: "tool_choice.function.name", code: "invalid_value",
	}
}

// checkToolMessages validates the tool turns in a conversation and converts them for rendering.
//
// A tool result is a real turn a model reads, so unlike the fields above it is no longer refused:
// Qwen2.5's template renders it as a <tool_response> inside a user turn, and Llama 3.1's puts it
// in an ipython turn. What is still refused is a tool turn that cannot be placed: one with no
// content, or a tool_call_id naming a call the conversation never made -- the model would read a
// result for a call it cannot see.
func checkToolMessages(msgs []chatMessage) ([]engine.Turn, error) {
	calls := make(map[string]bool)
	for _, m := range msgs {
		for _, c := range m.parsedToolCalls {
			calls[c.ID] = true
		}
	}
	turns := make([]engine.Turn, len(msgs))
	for i, m := range msgs {
		turn := engine.Turn{Role: m.Role, Text: string(m.Content)}
		for _, c := range m.parsedToolCalls {
			turn.ToolCalls = append(turn.ToolCalls, engine.ToolCall{
				ID: c.ID, Name: c.Name, Arguments: c.Arguments,
			})
		}
		switch {
		case m.Role == "tool":
			if strings.TrimSpace(turn.Text) == "" {
				return nil, &paramError{
					message: fmt.Sprintf("messages[%d] is a tool result with no content; the "+
						"model reads the result as the answer to its call, so an empty one "+
						"is a turn it cannot act on", i),
					param: fmt.Sprintf("messages[%d].content", i), code: "missing_required_parameter",
				}
			}
			if m.ToolCallID != "" && !calls[m.ToolCallID] {
				return nil, &paramError{
					message: fmt.Sprintf("messages[%d].tool_call_id is %q, which no earlier "+
						"assistant turn declared; the model would read a result for a call it "+
						"cannot see", i, m.ToolCallID),
					param: fmt.Sprintf("messages[%d].tool_call_id", i), code: "invalid_value",
				}
			}
		case m.ToolCallID != "":
			return nil, &paramError{
				message: fmt.Sprintf("messages[%d] carries tool_call_id but its role is %q, "+
					"not \"tool\"; a call id only means anything on the turn that answers it",
					i, m.Role),
				param: fmt.Sprintf("messages[%d].tool_call_id", i), code: "invalid_value",
			}
		}
		turns[i] = turn
	}
	return turns, nil
}

// proseSpan is one piece of a reply the parser returned as prose, as a byte range into the text it
// was given.
//
// The ranges exist because a caller that reports per-token log probabilities has to describe the
// same text the response shows, and the parser is what decides which parts of the reply those are.
// Recomputing them by searching for the markers again would be a second implementation of the same
// rule, and the two would drift the first time either changes.
//
// Concatenating them yields exactly the prose string parseToolCalls returns -- trimming included.
// That is the property the logprob list depends on, and it is why the ranges are computed by the
// same code that computes the string rather than alongside it: a span that disagrees with the text
// reports the probability of bytes the caller cannot see.
type proseSpan struct {
	// start and end are byte offsets into the text parseToolCallsSpans was given.
	start, end int
}

// parseToolCalls extracts the calls in a generated assistant turn.
//
// The marker set is the two the published templates ask for, and the text outside them is the
// assistant's prose. The rules are deliberately narrow because the alternative -- a lenient parser
// that guesses -- turns a model's ordinary text into a call:
//
//   - a call is exactly <tool_call> … </tool_call>, with the JSON object between them;
//   - an unclosed <tool_call> ends the reply and is not a call, because the model was cut off
//     mid-object and emitting half a call would hand the caller arguments that do not parse;
//   - the JSON must be an object with a non-empty "name". A "name" of the wrong type, or one
//     whose "arguments" is neither an object nor a string, is not a call: the response schema
//     says arguments is a JSON object, and passing a caller something else would fail in their
//     code far from this response.
//
// The returned text is what remains once the calls are removed, trimmed of the whitespace the
// template's own layout leaves around them. A model that wrote no calls returns its text
// unchanged, so this cannot alter an ordinary reply.
func parseToolCalls(text string) (string, []toolCall) {
	prose, calls, _ := parseToolCallsSpans(text)
	return prose, calls
}

// parseToolCallsSpans is parseToolCalls plus the byte ranges the prose came from.
//
// The spans are the untrimmed pieces, in reply order, and concatenating them gives the text with
// the calls cut out -- before the empty-prose and no-call special cases below, which is what a
// caller mapping tokens to what the response shows needs. Those special cases are about the prose
// *string*: an empty one is what an all-call reply trims to, and a reply with no calls keeps its
// text exactly, but the mapping is the same in both.
func parseToolCallsSpans(text string) (string, []toolCall, []proseSpan) {
	const open, closeTag = "<tool_call>", "</tool_call>"
	if !strings.Contains(text, open) {
		return text, nil, []proseSpan{{0, len(text)}}
	}
	var calls []toolCall
	var spans []proseSpan
	var prose strings.Builder
	rest := text
	// base is where rest starts in text, so the spans can be expressed in text's offsets.
	base := 0
	for {
		start := strings.Index(rest, open)
		if start < 0 {
			prose.WriteString(rest)
			if len(rest) > 0 {
				spans = append(spans, proseSpan{base, len(text)})
			}
			break
		}
		prose.WriteString(rest[:start])
		if start > 0 {
			spans = append(spans, proseSpan{base, base + start})
		}
		body := rest[start+len(open):]
		end := strings.Index(body, closeTag)
		if end < 0 {
			// Cut off mid-call: the fragment cannot be a call, and keeping it as prose would put
			// a bare "<tool_call>" in front of a caller who asked for text. It is dropped
			// instead, along with whatever followed it -- so a truncated reply loses its tail.
			// When this is the only marker the whole text comes back unchanged from the
			// len(calls) == 0 return below, which is the common shape: a model that started a
			// call and stopped has produced no calls, and its prose is all there is to show.
			break
		}
		c, ok := decodeToolCall(body[:end])
		if !ok {
			// Text that merely looks like a call stays text. The alternative is dropping what
			// the model wrote because it mentioned the marker.
			whole := rest[start : start+len(open)+end+len(closeTag)]
			prose.WriteString(whole)
			spans = append(spans, proseSpan{base + start, base + start + len(whole)})
			base += start + len(whole)
			rest = body[end+len(closeTag):]
			continue
		}
		calls = append(calls, c)
		base += start + len(open) + end + len(closeTag)
		rest = body[end+len(closeTag):]
	}
	if len(calls) == 0 {
		return text, nil, []proseSpan{{0, len(text)}}
	}
	trimmed, ranges := trimSpans(prose.String(), spans)
	return trimmed, calls, ranges
}

// trimSpans trims the given spans the way strings.TrimSpace trims their concatenation, and returns
// the result together with the trimmed ranges.
//
// The spans are recorded as the reply is walked, before the parse knows whether the whitespace
// around the calls is part of the answer. It is not: the templates wrap a call in newlines of their
// own, so the parser drops them, and the logprob list has to drop the tokens that produced them
// too. Keeping the two in one place is what stops the ranges from drifting from the string.
func trimSpans(prose string, spans []proseSpan) (string, []proseSpan) {
	trimmed := strings.TrimSpace(prose)
	if trimmed == "" {
		return "", nil
	}
	// The counts are byte counts, and strings.TrimSpace trims a rune at a time; counting the
	// difference in lengths gives the byte count of the whitespace actually removed, which is
	// what the span walk below needs.
	left := len(prose) - len(strings.TrimLeftFunc(prose, unicode.IsSpace))
	right := len(prose) - len(strings.TrimRightFunc(prose, unicode.IsSpace))

	// Walk the spans in order, dropping the first `left` bytes and then the last `right` of what
	// remains.
	out := make([]proseSpan, 0, len(spans))
	for _, sp := range spans {
		start, end := sp.start, sp.end
		if left > 0 {
			drop := min(left, end-start)
			start += drop
			left -= drop
		}
		if start < end {
			out = append(out, proseSpan{start, end})
		}
	}
	for i := len(out) - 1; i >= 0 && right > 0; i-- {
		drop := min(right, out[i].end-out[i].start)
		out[i].end -= drop
		right -= drop
		if out[i].end == out[i].start {
			out = out[:i]
		}
	}
	return trimmed, out
}

// toolCall is one call parsed out of a reply.
type toolCall struct {
	// ID is empty until the response assigns one; see assignToolCallIDs.
	ID string
	// Name is the function the model asked for.
	Name string
	// Arguments is the JSON object text, kept verbatim so the caller sees the model's own
	// spelling.
	Arguments string
}

// decodeToolCall decodes the JSON between the markers.
func decodeToolCall(body string) (toolCall, bool) {
	var v struct {
		Name      json.RawMessage `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		// Parameters is Llama 3.1's spelling of the same field, and its template asks for it
		// by name. Accepting both is not leniency about malformed input: the two families
		// really do instruct the model differently, so which key arrives is a property of the
		// checkpoint being served.
		Parameters json.RawMessage `json:"parameters"`
	}
	if err := json.Unmarshal([]byte(body), &v); err != nil {
		return toolCall{}, false
	}
	var name string
	if err := json.Unmarshal(v.Name, &name); err != nil || strings.TrimSpace(name) == "" {
		return toolCall{}, false
	}
	args := v.Arguments
	if len(args) == 0 {
		args = v.Parameters
	}
	switch {
	case len(args) == 0:
		// No arguments member at all. The schema calls arguments a string containing an
		// object, so {} is the honest answer for a tool that takes none.
		args = json.RawMessage(`{}`)
	case jsonKind(args) == '{':
		// Already an object, which is what Qwen2.5 emits. Verbatim.
	case jsonKind(args) == '"':
		// A JSON string holding the object, which is what OpenAI's response schema specifies
		// and what a model trained on that phrasing emits. Unquote it so the response carries
		// a string in both cases.
		var s string
		if err := json.Unmarshal(args, &s); err != nil {
			return toolCall{}, false
		}
		if jsonKind(json.RawMessage(s)) != '{' {
			return toolCall{}, false
		}
		args = json.RawMessage(s)
	default:
		return toolCall{}, false
	}
	return toolCall{Name: name, Arguments: string(args)}, true
}

// jsonKind returns the first non-space byte of a raw JSON value, or 0.
func jsonKind(raw json.RawMessage) byte {
	for _, b := range raw {
		switch b {
		case ' ', '\t', '\n', '\r':
			continue
		}
		return b
	}
	return 0
}

// assignToolCallIDs gives each call the identifier the response reports.
//
// The model does not write ids, and the schema requires one, so the server makes them: call_ and
// a per-response counter, which is the shape OpenAI's own responses carry and what a client echoes
// back in tool_call_id. The id is stable within a response and unique across responses because it
// is built from the response id.
func assignToolCallIDs(responseID string, calls []toolCall) []toolCall {
	out := make([]toolCall, len(calls))
	for i, c := range calls {
		c.ID = fmt.Sprintf("call_%s_%d", responseID, i)
		out[i] = c
	}
	return out
}

// toolCallsJSON renders calls in the shape the response schema documents: a list of objects with
// id, type and a function carrying name and arguments.
//
// arguments is a string, not an object. OpenAI's schema says "a JSON string", every SDK unmarshals
// it as one, and a response carrying an object fails in the client rather than here.
func toolCallsJSON(calls []toolCall) []map[string]any {
	out := make([]map[string]any, len(calls))
	for i, c := range calls {
		out[i] = map[string]any{
			"id":   c.ID,
			"type": "function",
			"function": map[string]string{
				"name":      c.Name,
				"arguments": c.Arguments,
			},
		}
	}
	return out
}

// resolveToolCallMessages decodes the tool_calls on each message and fills parsedToolCalls.
//
// It is separate from checkToolMessages because the two answer different questions and a caller
// needs the first to fix the second: this one is "is this JSON the shape the schema documents",
// and that one is "is the conversation these calls describe one the model can read".
//
// A call's arguments are required to be a JSON object, either spelled as an object or as a string
// containing one -- the two spellings the response itself can carry, and the two real clients use.
// Anything else is refused here rather than passed to a template that would print it as text the
// model cannot use.
func resolveToolCallMessages(msgs []chatMessage) error {
	for i := range msgs {
		raw := msgs[i].ToolCalls
		if !jsonAsks(raw) {
			continue
		}
		if msgs[i].Role != "assistant" {
			return &paramError{
				message: fmt.Sprintf("messages[%d].tool_calls is set but its role is %q; only an "+
					"assistant turn makes calls, and a template renders them inside one",
					i, msgs[i].Role),
				param: fmt.Sprintf("messages[%d].tool_calls", i), code: "invalid_value",
			}
		}
		var wire []struct {
			ID       string `json:"id"`
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
				// Kept raw so the object form and the string form can both be checked; see
				// argumentsText.
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			return &paramError{
				message: fmt.Sprintf(`messages[%d].tool_calls must be an array of {"id", `+
					`"type", "function": {"name", "arguments"}} objects: %s`, i, err.Error()),
				param: fmt.Sprintf("messages[%d].tool_calls", i), code: "invalid_value",
			}
		}
		calls := make([]engine.ToolCall, len(wire))
		for j, c := range wire {
			field := fmt.Sprintf("messages[%d].tool_calls[%d]", i, j)
			if strings.TrimSpace(c.Function.Name) == "" {
				return &paramError{
					message: fmt.Sprintf("%s.function.name is required: a call with no name "+
						"names no tool, so the turn it belongs to cannot be rendered", field),
					param: field + ".function.name", code: "missing_required_parameter",
				}
			}
			args, err := argumentsText(c.Function.Arguments)
			if err != nil {
				return &paramError{
					message: fmt.Sprintf("%s.function.arguments %v", field, err),
					param:   field + ".function.arguments", code: "invalid_value",
				}
			}
			calls[j] = engine.ToolCall{ID: c.ID, Name: c.Function.Name, Arguments: args}
		}
		msgs[i].parsedToolCalls = calls
	}
	return nil
}

// argumentsText normalises a call's arguments to the JSON object text a template prints.
//
// Both spellings the response can carry are accepted: an object (what Qwen2.5 emits and what an
// SDK that sends its own history back produces) and a string containing an object (what OpenAI's
// schema specifies and what the response below writes). Absent means no arguments, which is a real
// shape and becomes {}.
func argumentsText(raw json.RawMessage) (string, error) {
	switch {
	case len(raw) == 0:
		return "{}", nil
	case jsonKind(raw) == '{':
		return string(raw), nil
	case jsonKind(raw) == '"':
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", fmt.Errorf("is not a JSON string: %w", err)
		}
		if jsonKind(json.RawMessage(s)) != '{' {
			return "", fmt.Errorf("is a string that does not contain a JSON object: %q", s)
		}
		return s, nil
	default:
		return "", fmt.Errorf("must be a JSON object or a string containing one, got %s",
			strings.TrimSpace(string(raw)))
	}
}

// toolCallGate holds back the part of a streamed reply that may still become a tool call.
//
// Streaming a tool-calling reply is not the same problem as streaming text. A call is only a call
// once its closing marker arrives, so the bytes between <tool_call> and </tool_call> cannot be
// shown as they are generated -- a client that received them as content would render a JSON blob
// in the middle of an answer, and the tool_calls array that replaces them arrives only at the end.
// The alternatives are to buffer the whole reply and emit nothing until it finishes (which makes
// streaming pointless) or to hold back exactly the suspect part, which is what this does:
//
//   - text before any '<' goes out immediately, since a call cannot start without one;
//   - a '<' that could still become "<tool_call>" is held, at most that marker's length minus one
//     byte, so a plain '<' in prose costs one chunk of latency and nothing else;
//   - once the opening marker matches, everything up to and including </tool_call> is held.
//
// A held region that turns out not to be a call is released on flush, so text that merely mentions
// the marker still reaches the client.
//
// text is the only source of truth: sent decides how far the client's copy goes and never what the
// reply contains. parseToolCalls runs over the whole of text at the end regardless of how much was
// released, which is why withholding a call's bytes loses nothing -- the call is reported from
// text. push carries no decision from one push to the next beyond "a call is open": a single push
// may hold the end of one call, prose, and the whole of the next, and each region is handled in the
// order it appears, so what a client reads does not depend on where the token boundaries fell.
type toolCallGate struct {
	// text is everything pushed so far.
	text string
	// sent is how much of text the client's copy covers: everything before it was either
	// released as prose or withheld for good as part of a call.
	sent int
	// inCall is true once an opening marker has matched and its closing marker has not.
	inCall bool
	// dropped counts the bytes withheld for good as part of a call. It exists for the stream
	// reader: a described token may only carry a log probability when its text reaches the
	// client, and this is how the reader tells "held for now" from "gone". sent cannot say it,
	// because it advances over both -- what the client's copy covers is the same whether the
	// bytes went out or were dropped. Measured before it existed: with tools declared,
	// `x <tool_call>{...}</tool_call> y` reported logprob entries for the call's own bytes.
	dropped int
}

const (
	toolCallOpen  = "<tool_call>"
	toolCallClose = "</tool_call>"
)

// push adds generated text and returns what may now be sent.
//
// One push may hold several regions -- the tail of a call, the prose between two calls, the whole
// of the next call -- so the work is a loop over them rather than one decision. Every exit leaves
// sent at the first byte whose release is still undecided, which is what makes the output depend on
// the text and not on where the caller split it: the same reply pushed whole and pushed a byte at a
// time must release the same bytes in the same order.
func (g *toolCallGate) push(gen string) string {
	g.text += gen
	var out strings.Builder
	for {
		if g.inCall {
			// Inside a call nothing is released; the close marker ends it, and the call itself
			// is reported through the response's tool_calls rather than as content.
			j := strings.Index(g.text[g.sent:], toolCallClose)
			if j < 0 {
				// Unterminated so far. flush and parseToolCalls decide once, from the whole
				// text, whether this was a call; nothing inside the marker goes out as prose
				// either way.
				break
			}
			g.inCall = false
			body := g.text[g.sent+len(toolCallOpen) : g.sent+j]
			end := g.sent + j + len(toolCallClose)
			if _, ok := decodeToolCall(body); !ok {
				// A marker pair that does not hold a call is text, by the same rule the parser
				// uses: text that merely mentions the marker stays text, because dropping what
				// the model wrote is worse than showing a marker. Measured before this check:
				// `<tool_call></</tool_call>Hi` streamed "Hi" where the blocking body returned
				// the whole string, since the parser found no call and returned its input.
				//
				// The bytes between the markers are released here as prose, which is what the
				// already-released text before them leads into.
				out.WriteString(g.text[g.sent:end])
				g.sent = end
				continue
			}
			// The call's own bytes are dropped, which is the point: what the client sees is
			// the parsed call. sent moves just past the marker, not to the end of the text --
			// anything after it in this same push is prose, or the next call, and dropping it
			// would lose output the client never gets another chance to see.
			skip := j + len(toolCallClose)
			g.sent += skip
			g.dropped += skip
			continue
		}
		i := strings.Index(g.text[g.sent:], toolCallOpen)
		if i < 0 {
			// No complete marker yet. Release everything except a tail that is a prefix of the
			// opening marker -- that is the only way a marker can still be forming across a
			// chunk boundary.
			safe := g.safeEnd()
			if safe > g.sent {
				out.WriteString(g.text[g.sent:safe])
				g.sent = safe
			}
			break
		}
		// Everything before the marker is prose and goes out; the marker itself opens a region
		// whose release is decided by the pass above, which may find its close in this same push.
		out.WriteString(g.text[g.sent : g.sent+i])
		g.sent += i
		g.inCall = true
	}
	return out.String()
}

// safeEnd is the offset just past the last byte of text that can be released as prose: a tail that
// is a prefix of the opening marker is held, because the next push may complete it.
func (g *toolCallGate) safeEnd() int {
	for n := min(len(toolCallOpen)-1, len(g.text)-g.sent); n > 0; n-- {
		if strings.HasPrefix(toolCallOpen, g.text[len(g.text)-n:]) {
			return len(g.text) - n
		}
	}
	return len(g.text)
}

// flush returns whatever is still held once generation has ended.
//
// A complete call is never released here: parseToolCalls is what turns it into tool_calls, and
// duplicating its bytes as content would show the caller both the JSON and the parsed call. An
// unterminated opening marker is a different matter, and which way it goes depends on the rest of
// the reply: parseToolCalls drops a cut-off fragment when the reply holds a complete call beside
// it, and returns the whole text unchanged when it holds none -- so a reply that began a call and
// stopped has produced no calls, and its prose is all there is to show. Releasing the held region
// exactly when the parser will call it prose is what keeps the streamed and non-streamed answers
// the same text; dropping it unconditionally lost "Let me check. <tool_call>{...}" off the end of
// a streamed reply whose blocking answer kept it.
func (g *toolCallGate) flush() string {
	if g.inCall {
		if _, calls := parseToolCalls(g.text); len(calls) > 0 {
			// The parser will drop this fragment too, so releasing it would put a call's bytes
			// in front of the caller as prose.
			return ""
		}
	}
	out := g.text[g.sent:]
	g.sent = len(g.text)
	return out
}
