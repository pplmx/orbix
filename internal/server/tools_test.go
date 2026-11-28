package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/engine"
)

// TestParseToolCalls is the unit behind the response half: what counts as a call, and what does
// not.
//
// Every negative case here is a shape that a lenient parser would accept and then hand a client
// arguments that are not a JSON object -- which fails in their code, far from the response that
// caused it. The positives are the two spellings the templates in this project actually ask for:
// Qwen2.5 emits an object under "arguments", and Llama 3.1's template asks for "parameters".
func TestParseToolCalls(t *testing.T) {
	cases := []struct {
		name string
		in   string
		// prose is what remains once the calls are removed.
		prose string
		calls []toolCall
	}{
		{
			name:  "no call at all is returned unchanged",
			in:    "Paris is the capital of France.",
			prose: "Paris is the capital of France.",
		},
		{
			name:  "a bare call leaves no prose",
			in:    `<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>`,
			prose: "",
			calls: []toolCall{{Name: "get_weather", Arguments: `{"city": "Paris"}`}},
		},
		{
			name: "prose around the call survives",
			in: "Let me check.\n<tool_call>{\"name\": \"get_weather\", " +
				`"arguments": {"city": "Paris"}}</tool_call>` + "\nOne moment.",
			prose: "Let me check.\n\nOne moment.",
			calls: []toolCall{{Name: "get_weather", Arguments: `{"city": "Paris"}`}},
		},
		{
			name: "two calls in one reply",
			in: `<tool_call>{"name": "a", "arguments": {}}</tool_call>` +
				`<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>`,
			prose: "",
			calls: []toolCall{
				{Name: "a", Arguments: `{}`},
				{Name: "b", Arguments: `{"x": 1}`},
			},
		},
		{
			// A reply cut off before the closing tag: the fragment is not a call, and the whole
			// text comes back unchanged when it is the only marker.
			name:  "an unclosed call with no other call is returned as text",
			in:    `hello <tool_call>{"name": "x"}`,
			prose: `hello <tool_call>{"name": "x"}`,
		},
		{
			// Beside a complete call, the cut-off fragment is dropped: it cannot be a call, and
			// keeping it would put a bare "<tool_call>" in front of a caller who asked for text.
			// What is lost is the fragment, not the prose before it -- this case pins both.
			name: "a cut-off call beside a complete one is dropped",
			in: `<tool_call>{"name": "a", "arguments": {}}</tool_call>` +
				` and then <tool_call>{"name": "b"`,
			prose: "and then",
			calls: []toolCall{{Name: "a", Arguments: `{}`}},
		},
		{
			// Llama 3.1's template instructs the model to answer with this key instead.
			name:  "parameters is accepted as the arguments key",
			in:    `<tool_call>{"name": "get_weather", "parameters": {"city": "Paris"}}</tool_call>`,
			prose: "",
			calls: []toolCall{{Name: "get_weather", Arguments: `{"city": "Paris"}`}},
		},
		{
			// OpenAI's own response schema says arguments is a string, and a model trained on
			// that phrasing emits one.
			name:  "arguments as a JSON string is unquoted",
			in:    `<tool_call>{"name": "f", "arguments": "{\"city\": \"Paris\"}"}</tool_call>`,
			prose: "",
			calls: []toolCall{{Name: "f", Arguments: `{"city": "Paris"}`}},
		},
		{
			// A tool that takes no arguments is a real shape; {} is the honest answer.
			name:  "no arguments member becomes an empty object",
			in:    `<tool_call>{"name": "list_files"}</tool_call>`,
			prose: "",
			calls: []toolCall{{Name: "list_files", Arguments: `{}`}},
		},
		{
			// The model was cut off, or never found the closing marker. Emitting half a call
			// would give the caller arguments that do not parse.
			name:  "an unterminated call is not a call",
			in:    `<tool_call>{"name": "get_weather", "arguments": {"ci`,
			prose: `<tool_call>{"name": "get_weather", "arguments": {"ci`,
		},
		{
			name:  "malformed JSON is left as text",
			in:    `<tool_call>{not json}</tool_call>`,
			prose: `<tool_call>{not json}</tool_call>`,
		},
		{
			name:  "a call with no name is left as text",
			in:    `<tool_call>{"arguments": {}}</tool_call>`,
			prose: `<tool_call>{"arguments": {}}</tool_call>`,
		},
		{
			name:  "a call whose name is not a string is left as text",
			in:    `<tool_call>{"name": 42, "arguments": {}}</tool_call>`,
			prose: `<tool_call>{"name": 42, "arguments": {}}</tool_call>`,
		},
		{
			name:  "arguments that are not an object or an object string are refused",
			in:    `<tool_call>{"name": "f", "arguments": 42}</tool_call>`,
			prose: `<tool_call>{"name": "f", "arguments": 42}</tool_call>`,
		},
		{
			name:  "a string argument that is not an object is refused",
			in:    `<tool_call>{"name": "f", "arguments": "just text"}</tool_call>`,
			prose: `<tool_call>{"name": "f", "arguments": "just text"}</tool_call>`,
		},
		{
			// The marker is prose here. Dropping the text because it mentions the tag would
			// silently delete what the model said.
			name:  "a mentioned marker without a following call stays text",
			in:    "Use <tool_call> like this.",
			prose: "Use <tool_call> like this.",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			prose, calls := parseToolCalls(c.in)
			if len(c.calls) == 0 {
				// The no-call path returns the input untouched rather than a trimmed copy, and
				// that matters: an ordinary reply must come back byte-for-byte.
				if prose != c.in {
					t.Errorf("text = %q, want the input unchanged (%q)", prose, c.in)
				}
				if len(calls) != 0 {
					t.Errorf("calls = %+v, want none", calls)
				}
				return
			}
			if prose != c.prose {
				t.Errorf("prose = %q, want %q", prose, c.prose)
			}
			if len(calls) != len(c.calls) {
				t.Fatalf("got %d calls, want %d: %+v", len(calls), len(c.calls), calls)
			}
			for i, w := range c.calls {
				if calls[i].Name != w.Name {
					t.Errorf("call %d name = %q, want %q", i, calls[i].Name, w.Name)
				}
				// Compared as text rather than as decoded JSON so the verbatim guarantee is
				// what is asserted: a re-encode that reordered keys would still be equal as a
				// map and would change what the caller reads.
				if calls[i].Arguments != w.Arguments {
					t.Errorf("call %d arguments = %q, want %q", i, calls[i].Arguments, w.Arguments)
				}
			}
		})
	}
}

// TestToolCallIDsAreUniquePerResponse covers the id the response assigns.
//
// A tool_call_id is how a client tells two calls apart when it sends the results back, so two
// calls in one response must not share one, and two responses must not either -- a client running
// two conversations against one server would otherwise have its history collide.
func TestToolCallIDsAreUniquePerResponse(t *testing.T) {
	calls := []toolCall{{Name: "a", Arguments: "{}"}, {Name: "b", Arguments: "{}"}}
	first := assignToolCallIDs("chatcmpl-1-1", calls)
	second := assignToolCallIDs("chatcmpl-2-2", calls)
	seen := make(map[string]bool)
	for _, group := range [][]toolCall{first, second} {
		for _, c := range group {
			if c.ID == "" {
				t.Errorf("a call came back without an id: %+v", c)
			}
			if seen[c.ID] {
				t.Errorf("id %q was assigned twice", c.ID)
			}
			seen[c.ID] = true
		}
	}
	// The name and arguments survive the assignment, which is easy to lose in a rebuild.
	if first[0].Name != "a" || first[1].Name != "b" {
		t.Errorf("assigning ids changed the calls: %+v", first)
	}
}

// TestToolCallGateStreamsOnlyProse covers the streaming half: a call's bytes must not reach the
// client as content, and text that only looks like the start of one must.
//
// The gate's job is to release as much as is safe at every push. The cases are the boundaries
// where "safe" changes: before a '<' arrives, partway through the opening marker, inside a call,
// and after it closes.
func TestToolCallGateStreamsOnlyProse(t *testing.T) {
	t.Run("text with no marker streams through", func(t *testing.T) {
		var g toolCallGate
		var got strings.Builder
		for _, chunk := range []string{"Paris", " is", " the", " capital."} {
			got.WriteString(g.push(chunk))
		}
		got.WriteString(g.flush())
		if want := "Paris is the capital."; got.String() != want {
			t.Errorf("streamed %q, want %q", got.String(), want)
		}
	})

	t.Run("a marker split across chunks is withheld whole", func(t *testing.T) {
		var g toolCallGate
		var got strings.Builder
		// The marker arrives in fragments, which is what a token stream actually does.
		for _, chunk := range []string{"hi ", "<tool", "_call>", `{"name": "f"}`, "</tool_call>", " done"} {
			got.WriteString(g.push(chunk))
		}
		got.WriteString(g.flush())
		if want := "hi  done"; got.String() != want {
			t.Errorf("streamed %q, want %q (the call's bytes must not appear)", got.String(), want)
		}
	})

	t.Run("a trailing partial marker is released when it never completes", func(t *testing.T) {
		var g toolCallGate
		var got strings.Builder
		got.WriteString(g.push("a <too"))
		// Nothing was releasable: "<too" could still be the marker.
		if got.String() != "a " {
			t.Fatalf("released %q before the marker could be ruled out", got.String())
		}
		got.WriteString(g.flush())
		if want := "a <too"; got.String() != want {
			t.Errorf("streamed %q, want %q: prose that merely starts like the marker is prose",
				got.String(), want)
		}
	})

	t.Run("a bare lt sign does not stall the stream", func(t *testing.T) {
		var g toolCallGate
		if out := g.push("a < b"); out != "a < b" {
			t.Errorf("released %q, want the whole chunk: '<' followed by a space cannot begin "+
				"the marker", out)
		}
	})
}

// TestFinalChoiceCarriesCalls covers the chunk shape a streamed tool call ends with.
//
// This test exists because a bug the unit tests below could not see reached a real checkpoint
// first: the stream emitted the calls as an item of its own, writeItem classified it as a delta
// and drew an empty payload, and the response ended with finish_reason "tool_calls" and no
// tool_calls anywhere in it. The shape is asserted here rather than only end to end because the
// end-to-end path needs a model that produces a call.
func TestFinalChoiceCarriesCalls(t *testing.T) {
	calls := []toolCall{{ID: "call_1", Name: "get_weather", Arguments: `{"city": "Paris"}`}}
	chunk := finalChoice(true, 0, "tool_calls", calls)
	choices, ok := chunk["choices"].([]map[string]any)
	if !ok || len(choices) != 1 {
		t.Fatalf("choices = %#v, want one entry", chunk["choices"])
	}
	if got := choices[0]["finish_reason"]; got != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", got)
	}
	delta, ok := choices[0]["delta"].(map[string]any)
	if !ok {
		t.Fatalf("delta = %#v, want an object carrying tool_calls", choices[0]["delta"])
	}
	got, ok := delta["tool_calls"].([]map[string]any)
	if !ok || len(got) != 1 {
		t.Fatalf("delta.tool_calls = %#v, want one call", delta["tool_calls"])
	}
	fn, _ := got[0]["function"].(map[string]string)
	if fn["name"] != "get_weather" || fn["arguments"] != `{"city": "Paris"}` {
		t.Errorf("call = %#v, want the name and arguments verbatim", got[0])
	}
	// The empty content is sent rather than omitted: the same field appears on every other delta
	// of the turn, and a client that indexes it would otherwise special-case the last chunk.
	if content, ok := delta["content"]; !ok || content != "" {
		t.Errorf("delta content = %#v, want an empty string", content)
	}

	// With no calls the chunk is the ordinary terminal, so a plain reply's stream is unchanged.
	plain := finalChoice(true, 0, "stop", nil)
	pc := plain["choices"].([]map[string]any)
	if _, present := pc[0]["delta"].(map[string]string); !present {
		t.Errorf("a call-free terminal chunk changed shape: %#v", pc[0]["delta"])
	}
}

// TestStreamOneChoiceCarriesCallsToTheTerminalChunk is the layer the earlier test could not
// reach: streamOneChoice building the terminal item.
//
// TestFinalChoiceCarriesCalls checks what finalChoice does with calls it is given, and passes for
// a caller that never gives it any -- which is precisely the bug a real checkpoint found. The
// calls were emitted as an item of their own, writeItem classified that item as a delta, and the
// stream ended with finish_reason "tool_calls" and nothing in tool_calls. Asserting on the items
// streamOneChoice produces is the only way to see the wiring without a model that emits a call.
func TestStreamOneChoiceCarriesCallsToTheTerminalChunk(t *testing.T) {
	srv, tok := newTestServer(t)
	// The reply is a call and nothing else, so the only thing the gate can release is the
	// terminal chunk.
	const reply = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}" +
		"</tool_call>"
	ids := tok.Encode(reply, false)
	ch := make(chan engine.Event, len(ids)+1)
	for _, id := range ids {
		ch <- engine.Event{Token: id, HasToken: true}
	}
	ch <- engine.Event{Done: true, FinishReason: "stop"}
	close(ch)

	p := generateParams{chat: true, responseID: "chatcmpl-test-1",
		tools: []engine.ToolSpec{{Name: "get_weather"}}}
	var items []streamItem
	if _, _, ok := srv.streamOneChoice(ch, 0, p, func(it streamItem) bool {
		items = append(items, it)
		return true
	}); !ok {
		t.Fatal("streamOneChoice reported a failed write")
	}
	if len(items) == 0 {
		t.Fatal("no items were emitted")
	}
	last := items[len(items)-1]
	if last.reason != "tool_calls" {
		t.Errorf("terminal reason = %q, want tool_calls", last.reason)
	}
	// The load-bearing assertion: the calls travel on the terminal item, which is the only item
	// writeItem renders as finalChoice. A separate item is dropped on the floor.
	if len(last.calls) != 1 || last.calls[0].Name != "get_weather" {
		t.Fatalf("terminal item carries %+v, want the parsed call", last.calls)
	}
	if got := last.calls[0].Arguments; got != `{"city": "Paris"}` {
		t.Errorf("arguments = %q, want them verbatim", got)
	}
	// And none of the call's bytes leaked out as content.
	for _, it := range items {
		if strings.Contains(it.delta, "tool_call") || strings.Contains(it.delta, "get_weather") {
			t.Errorf("a delta carried part of the call: %q", it.delta)
		}
	}
}

// TestCheckToolsRefusesWhatItCannotHonour covers every branch of the validation that decides
// which tool lists reach a prompt.
//
// Each row asserts three things, because each one is a separate way a refusal can be useless:
// the status the handler would use (set by the code), the *param* naming the field the caller has
// to fix, and a phrase from the message saying why. A message that only says "invalid_value" would
// leave the caller guessing which of their tools was wrong, and the param is what an SDK shows.
//
// These branches had no test at all when the feature landed. The coverage was checkTools 75%;
// every row below was an uncovered statement.
func TestCheckToolsRefusesWhatItCannotHonour(t *testing.T) {
	const okTool = `{"type": "function", "function": {"name": "get_weather"}}`
	cases := []struct {
		name      string
		raw       string
		wantParam string
		wantCode  string
		wantMsg   string
	}{
		{
			name:      "not an array",
			raw:       `{"type": "function"}`,
			wantParam: "tools",
			wantCode:  "invalid_value",
			wantMsg:   "must be an array",
		},
		{
			name:      "a tool type that is not function",
			raw:       `[{"type": "retrieval", "function": {"name": "r"}}]`,
			wantParam: "tools[0].type",
			wantCode:  "unsupported_parameter",
			wantMsg:   `"retrieval"`,
		},
		{
			name:      "a type with no function member",
			raw:       `[{"type": "function"}]`,
			wantParam: "tools[0].function",
			wantCode:  "missing_required_parameter",
			wantMsg:   "has no function member",
		},
		{
			name:      "an empty name",
			raw:       `[{"type": "function", "function": {"name": "   "}}]`,
			wantParam: "tools[0].function.name",
			wantCode:  "missing_required_parameter",
			wantMsg:   "an unnamed one could not be called",
		},
		{
			// Two tools with one name make every call to it ambiguous, so the refusal has to
			// name the second declaration rather than the first.
			name: "a name declared twice",
			raw: `[` + okTool + `, {"type": "function", "function": ` +
				`{"name": "get_weather", "description": "again"}}]`,
			wantParam: "tools[1].function.name",
			wantCode:  "invalid_value",
			wantMsg:   "declared more than once",
		},
		{
			name:      "strict, which needs constrained decoding",
			raw:       `[{"type": "function", "function": {"name": "f", "strict": true}}]`,
			wantParam: "tools[0].function.strict",
			wantCode:  "unsupported_parameter",
			wantMsg:   "constrained decoding",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, _, err := checkTools(json.RawMessage(c.raw))
			if err == nil {
				t.Fatalf("%s was accepted", c.raw)
			}
			var pe *paramError
			if !errors.As(err, &pe) {
				t.Fatalf("error %v is not a paramError, so the response would carry no field", err)
			}
			if pe.param != c.wantParam {
				t.Errorf("param = %q, want %q", pe.param, c.wantParam)
			}
			if pe.code != c.wantCode {
				t.Errorf("code = %q, want %q", pe.code, c.wantCode)
			}
			if !strings.Contains(pe.message, c.wantMsg) {
				t.Errorf("message %q does not contain %q", pe.message, c.wantMsg)
			}
		})
	}

	// The accepting side, so the table above cannot pass by refusing everything.
	t.Run("a valid list is accepted", func(t *testing.T) {
		_, specs, err := checkTools(json.RawMessage(`[` + okTool + `]`))
		if err != nil {
			t.Fatalf("a valid tool was refused: %v", err)
		}
		if len(specs) != 1 || specs[0].Name != "get_weather" {
			t.Fatalf("specs = %+v, want the one tool", specs)
		}
		// A tool with no parameters is a real shape and its schema becomes {}, which is what a
		// template prints; leaving it empty would render a signature with no argument document.
		if specs[0].Parameters != "{}" {
			t.Errorf("parameters = %q, want {}", specs[0].Parameters)
		}
	})
	t.Run("an explicit null parameters also becomes {}", func(t *testing.T) {
		_, specs, err := checkTools(json.RawMessage(
			`[{"type": "function", "function": {"name": "f", "parameters": null}}]`))
		if err != nil {
			t.Fatalf("null parameters was refused: %v", err)
		}
		if specs[0].Parameters != "{}" {
			t.Errorf("parameters = %q, want {}", specs[0].Parameters)
		}
	})
	t.Run("the empty forms declare nothing", func(t *testing.T) {
		for _, raw := range []string{"", "null", "[]"} {
			tools, specs, err := checkTools(json.RawMessage(raw))
			if err != nil {
				t.Errorf("%q was refused: %v", raw, err)
			}
			if tools != nil || specs != nil {
				t.Errorf("%q produced %d tools and %d specs, want none", raw, len(tools), len(specs))
			}
		}
	})
}

// TestCheckToolChoice pins which tool_choice values are accepted and which are refused.
//
// This function was at 35% coverage when the feature landed: only the "auto" path had ever run,
// and "auto" is what an absent field produces. The refusal of a forcing choice is the whole reason
// the function exists -- no template in the subset can print the constraint, so accepting
// "required" would answer with whatever the model chose while the caller believed it was forced.
func TestCheckToolChoice(t *testing.T) {
	tools := []toolRequest{{Type: "function", Function: &toolFunction{Name: "get_weather"}}}

	t.Run("the accepted spellings", func(t *testing.T) {
		cases := []struct {
			name, raw, wantMode, wantName string
		}{
			{"absent", "", modeAuto, ""},
			{"null", "null", modeAuto, ""},
			{"auto", `"auto"`, modeAuto, ""},
			{"none", `"none"`, modeNone, ""},
			{"required parses", `"required"`, modeRequired, ""},
			{"named", `{"type": "function", "function": {"name": "get_weather"}}`, modeNamed, "get_weather"},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				got, err := checkToolChoice(json.RawMessage(c.raw), tools)
				if err != nil {
					t.Fatalf("%q was refused: %v", c.raw, err)
				}
				if got.Mode != c.wantMode {
					t.Errorf("mode = %q, want %q", got.Mode, c.wantMode)
				}
				if want := c.wantName; got.Name != want {
					t.Errorf("name = %q, want %q", got.Name, want)
				}
			})
		}
	})

	t.Run("the refusals", func(t *testing.T) {
		cases := []struct {
			name, raw, wantParam, wantMsg string
		}{
			{"an unknown string", `"sometimes"`, "tool_choice", "not one of auto, none, required"},
			// A name that is not in this request's list cannot be forced, so it is refused with
			// the reason rather than silently treated as auto.
			{"a name the request did not declare",
				`{"type": "function", "function": {"name": "nope"}}`,
				"tool_choice.function.name", "not one of the tools this request declares"},
			{"an object with no function", `{"type": "function"}`,
				"tool_choice", "must be"},
			{"a non-function object", `{"type": "retrieval", "function": {"name": "f"}}`,
				"tool_choice", "must be"},
			{"a name that is only whitespace",
				`{"type": "function", "function": {"name": "  "}}`,
				"tool_choice", "must be"},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				_, err := checkToolChoice(json.RawMessage(c.raw), tools)
				if err == nil {
					t.Fatalf("%q was accepted", c.raw)
				}
				var pe *paramError
				if !errors.As(err, &pe) {
					t.Fatalf("error %v is not a paramError", err)
				}
				if pe.param != c.wantParam {
					t.Errorf("param = %q, want %q", pe.param, c.wantParam)
				}
				if !strings.Contains(pe.message, c.wantMsg) {
					t.Errorf("message %q does not contain %q", pe.message, c.wantMsg)
				}
			})
		}
	})
}

// TestCheckToolMessages pins the validation of a conversation's tool turns.
//
// The three refusals are all about a result the model cannot place: an empty one, one naming a
// call nobody declared, and a call id on a turn that is not a tool result. Each was uncovered.
func TestCheckToolMessages(t *testing.T) {
	withCall := func() []chatMessage {
		msgs := []chatMessage{
			{Role: "user", Content: "hi"},
			{Role: "assistant", ToolCalls: json.RawMessage(
				`[{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]`)},
		}
		if err := resolveToolCallMessages(msgs); err != nil {
			t.Fatalf("resolveToolCallMessages: %v", err)
		}
		return msgs
	}

	t.Run("a well-formed exchange is accepted and converted", func(t *testing.T) {
		msgs := append(withCall(), chatMessage{
			Role: "tool", Content: `{"t": 21}`, ToolCallID: "call_1"})
		turns, err := checkToolMessages(msgs)
		if err != nil {
			t.Fatalf("a valid exchange was refused: %v", err)
		}
		if len(turns) != 3 {
			t.Fatalf("got %d turns, want 3", len(turns))
		}
		// The call has to survive the conversion: it is what the template renders back.
		if len(turns[1].ToolCalls) != 1 || turns[1].ToolCalls[0].Name != "f" {
			t.Errorf("the assistant turn carries %+v, want the call", turns[1].ToolCalls)
		}
	})

	cases := []struct {
		name   string
		mutate func([]chatMessage) []chatMessage
		param  string
		msg    string
	}{
		{
			name: "an empty tool result",
			mutate: func(m []chatMessage) []chatMessage {
				return append(m, chatMessage{Role: "tool", Content: "   ", ToolCallID: "call_1"})
			},
			param: "messages[2].content",
			msg:   "tool result with no content",
		},
		{
			name: "a result naming an undeclared call",
			mutate: func(m []chatMessage) []chatMessage {
				return append(m, chatMessage{Role: "tool", Content: "x", ToolCallID: "call_9"})
			},
			param: "messages[2].tool_call_id",
			msg:   "no earlier assistant turn declared",
		},
		{
			name: "a call id on a non-tool turn",
			mutate: func(m []chatMessage) []chatMessage {
				return append(m, chatMessage{Role: "user", Content: "x", ToolCallID: "call_1"})
			},
			param: "messages[2].tool_call_id",
			msg:   `not "tool"`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := checkToolMessages(c.mutate(withCall()))
			if err == nil {
				t.Fatal("the conversation was accepted")
			}
			var pe *paramError
			if !errors.As(err, &pe) {
				t.Fatalf("error %v is not a paramError", err)
			}
			if pe.param != c.param {
				t.Errorf("param = %q, want %q", pe.param, c.param)
			}
			if !strings.Contains(pe.message, c.msg) {
				t.Errorf("message %q does not contain %q", pe.message, c.msg)
			}
		})
	}

	// A tool turn with no tool_call_id at all is allowed: some clients send the result without
	// echoing an id, and the turn is still renderable because the call precedes it.
	t.Run("a tool result with no call id is accepted", func(t *testing.T) {
		msgs := append(withCall(), chatMessage{Role: "tool", Content: "x"})
		if _, err := checkToolMessages(msgs); err != nil {
			t.Errorf("a result without a call id was refused: %v", err)
		}
	})
}

// TestResolveToolCallMessages pins the decoding of a message's tool_calls and the argument shapes.
func TestResolveToolCallMessages(t *testing.T) {
	cases := []struct {
		name  string
		raw   string
		want  string
		param string
		msg   string
	}{
		{
			name: "arguments as an object",
			raw:  `[{"id": "c", "type": "function", "function": {"name": "f", "arguments": {"city": "P"}}}]`,
			want: `{"city": "P"}`,
		},
		{
			name: "arguments as a JSON string, which is what the response writes",
			raw: `[{"id": "c", "type": "function", "function": ` +
				`{"name": "f", "arguments": "{\"city\": \"P\"}"}}]`,
			want: `{"city": "P"}`,
		},
		{
			name: "no arguments member becomes {}",
			raw:  `[{"id": "c", "type": "function", "function": {"name": "f"}}]`,
			want: `{}`,
		},
		{
			name:  "a number argument is refused",
			raw:   `[{"id": "c", "type": "function", "function": {"name": "f", "arguments": 42}}]`,
			param: "messages[0].tool_calls[0].function.arguments",
			msg:   "must be a JSON object or a string containing one",
		},
		{
			name:  "a string that is not an object is refused",
			raw:   `[{"id": "c", "type": "function", "function": {"name": "f", "arguments": "text"}}]`,
			param: "messages[0].tool_calls[0].function.arguments",
			msg:   "does not contain a JSON object",
		},
		{
			name:  "a call with no name is refused",
			raw:   `[{"id": "c", "type": "function", "function": {"arguments": {}}}]`,
			param: "messages[0].tool_calls[0].function.name",
			msg:   "names no tool",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			msgs := []chatMessage{{Role: "assistant", ToolCalls: json.RawMessage(c.raw)}}
			err := resolveToolCallMessages(msgs)
			if c.param != "" {
				if err == nil {
					t.Fatalf("%s was accepted", c.raw)
				}
				var pe *paramError
				if !errors.As(err, &pe) {
					t.Fatalf("error %v is not a paramError", err)
				}
				if pe.param != c.param {
					t.Errorf("param = %q, want %q", pe.param, c.param)
				}
				if !strings.Contains(pe.message, c.msg) {
					t.Errorf("message %q does not contain %q", pe.message, c.msg)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveToolCallMessages: %v", err)
			}
			if got := msgs[0].parsedToolCalls; len(got) != 1 || got[0].Arguments != c.want {
				t.Errorf("parsed = %+v, want arguments %q", got, c.want)
			}
		})
	}

	// tool_calls belongs on an assistant turn only; anywhere else it is refused by name.
	t.Run("tool_calls on a user turn", func(t *testing.T) {
		msgs := []chatMessage{{Role: "user", ToolCalls: json.RawMessage(
			`[{"id": "c", "type": "function", "function": {"name": "f"}}]`)}}
		err := resolveToolCallMessages(msgs)
		if err == nil {
			t.Fatal("tool_calls on a user turn was accepted")
		}
		if !strings.Contains(err.Error(), "only an assistant turn makes calls") {
			t.Errorf("error %q does not say which role may carry calls", err)
		}
	})
}

// TestToolCallGateHoldsAnUnterminatedCall covers the flush path for a stream that ends inside a
// call.
//
// Two shapes, and they differ, because what the parser does with a cut-off fragment depends on the
// rest of the reply. Beside a complete call the fragment is dropped, and releasing it here would
// put a call's bytes in front of the caller as prose immediately before the same bytes as a call.
// With no complete call in the reply at all, the parser returns the whole text as prose -- so the
// fragment is prose, and dropping it loses the tail of a streamed answer that the non-streamed body
// keeps. The old comment here said "nothing is lost ... the decision is made once, from the whole
// text", which is true of the parser and was false of the stream: the streaming path reads only
// the calls from parseToolCalls and throws its prose away.
func TestToolCallGateHoldsAnUnterminatedCall(t *testing.T) {
	t.Run("with no complete call the fragment is prose", func(t *testing.T) {
		var g toolCallGate
		g.push("before <tool_call>{\"name\": \"f\", \"argum")
		want := `before <tool_call>{"name": "f", "argum`
		if got := g.flush(); got != `<tool_call>{"name": "f", "argum` {
			t.Errorf("flush released %q, want the held fragment", got)
		}
		// And that is what the blocking path shows, which is the reason it is released: the
		// parser agrees the whole text is prose.
		prose, calls := parseToolCalls(g.text)
		if len(calls) != 0 {
			t.Fatalf("an unterminated call parsed as %+v", calls)
		}
		if prose != want {
			t.Errorf("the parser returns %q, want %q", prose, want)
		}
	})

	t.Run("beside a complete call the fragment is dropped", func(t *testing.T) {
		var g toolCallGate
		g.push(`<tool_call>{"name": "a", "arguments": {}}</tool_call> and then <tool_call>{"name": "b"`)
		// Whatever preceded the second marker is prose and was released as it arrived.
		if got := g.flush(); got != "" {
			t.Errorf("flush released %q, want nothing: the parser drops this fragment too", got)
		}
		_, calls := parseToolCalls(g.text)
		if len(calls) != 1 || calls[0].Name != "a" {
			t.Fatalf("the parser found %+v, want the one complete call", calls)
		}
	})
}

// TestJSONKind covers the helper the argument checks are built on. Its empty-input answer (0) was
// uncovered, and it is the one that decides an absent arguments member is not an object.
func TestJSONKind(t *testing.T) {
	cases := []struct {
		in   string
		want byte
	}{
		{"", 0},
		{"   \n\t ", 0},
		{`{"a": 1}`, '{'},
		{`  "text"`, '"'},
		{"42", '4'},
		{"null", 'n'},
	}
	for _, c := range cases {
		if got := jsonKind(json.RawMessage(c.in)); got != c.want {
			t.Errorf("jsonKind(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestCheckToolsRefusesMalformedConversationJSON covers the decode failure the table above does not
// reach: tool_calls that is not an array of objects at all.
func TestCheckToolsRefusesMalformedConversationJSON(t *testing.T) {
	for _, raw := range []string{`{"id": "c"}`, `["not an object"]`, `[{"function": 42}]`} {
		msgs := []chatMessage{{Role: "assistant", ToolCalls: json.RawMessage(raw)}}
		err := resolveToolCallMessages(msgs)
		if err == nil {
			t.Errorf("tool_calls %s was accepted", raw)
			continue
		}
		if !strings.Contains(err.Error(), "must be an array of") {
			t.Errorf("error for %s does not say the shape: %v", raw, err)
		}
	}
}

// TestArgumentsTextRefusesANonStringJSON covers the branch where arguments is syntactically a
// string but does not decode as one. It is unreachable from JSON input (encoding/json would have
// rejected the outer document), so it is exercised directly: the check exists so that a future
// caller passing raw bytes cannot get past it silently.
func TestArgumentsTextRefusesANonStringJSON(t *testing.T) {
	// The message has to say the value is not a JSON string, because that is the fact the
	// caller cannot see from a bool: "arguments" is a field name on the wire, and a value that
	// failed to decode needs to be told apart from one that decoded but was the wrong shape --
	// the next case below reports that one differently.
	_, err := argumentsText(json.RawMessage(`"unterminated`))
	if err == nil {
		t.Fatal("a string that does not decode was accepted as arguments")
	}
	if !strings.Contains(err.Error(), "is not a JSON string") {
		t.Errorf("error = %q, want it to name the decoding failure", err)
	}
}

// TestCheckToolChoiceRefusesNonObject covers tool_choice values that are valid JSON but neither a
// string nor an object, which is the decode-error branch.
func TestCheckToolChoiceRefusesNonObject(t *testing.T) {
	for _, raw := range []string{"[1]", "true", "42"} {
		_, err := checkToolChoice(json.RawMessage(raw), nil)
		if err == nil {
			t.Errorf("tool_choice %s was accepted", raw)
			continue
		}
		if !strings.Contains(err.Error(), "must be") {
			t.Errorf("error for %s does not describe the accepted shapes: %v", raw, err)
		}
	}
}

// TestChatRefusesAForcingToolChoice is the handler half of the tool_choice refusals.
//
// checkToolChoice parses "required" and a named function without complaint, because parsing them
// is not the same as honouring them: the handler is what decides they cannot be served, and that
// decision had no test. The refusal is the point of the whole check -- no template in the subset
// has a field that forces a call, so accepting "required" would answer with whatever the model
// chose while the caller believed the choice was forced.
func TestChatRefusesAForcingToolChoice(t *testing.T) {
	const tmpl = `{%- if tools %}TOOLS{% endif %}{%- for m in messages %}[{{ m.role }}]{% endfor %}`
	srv := newTestServerWithTemplate(t, tmpl, Config{})
	const tool = `{"type": "function", "function": {"name": "get_weather"}}`

	cases := []struct{ name, choice string }{
		{"required", `"required"`},
		{"a named function", `{"type": "function", "function": {"name": "get_weather"}}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body := `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,` +
				` "tools": [` + tool + `], "tool_choice": ` + c.choice + `}`
			resp := post(t, srv.Handler(), "/v1/chat/completions", body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != 400 {
				t.Fatalf("status = %d, want 400: a forcing tool_choice cannot be honoured",
					resp.StatusCode)
			}
			out := decodeBody[errorBody](t, resp)
			if out.Error.Param != "tool_choice" {
				t.Errorf("param = %q, want tool_choice", out.Error.Param)
			}
			if out.Error.Code != "unsupported_parameter" {
				t.Errorf("code = %q, want unsupported_parameter", out.Error.Code)
			}
			// The message has to name the reason and the alternative, or a caller cannot tell
			// whether to drop the field or change servers.
			for _, want := range []string{"forces a call", "auto", "none"} {
				if !strings.Contains(out.Error.Message, want) {
					t.Errorf("message %q does not mention %q", out.Error.Message, want)
				}
			}
		})
	}

	// And the two served modes still answer, so the refusal is specific to forcing.
	for _, choice := range []string{`"auto"`, `"none"`} {
		body := `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,` +
			` "tools": [` + tool + `], "tool_choice": ` + choice + `}`
		resp := post(t, srv.Handler(), "/v1/chat/completions", body)
		if resp.StatusCode != 200 {
			out := decodeBody[errorBody](t, resp)
			t.Errorf("tool_choice %s = %d, want 200: %s", choice, resp.StatusCode, out.Error.Message)
		}
		_ = resp.Body.Close()
	}
}

// TestChatRefusesToolsItCannotHonour is the handler-level form of the checkTools refusals.
//
// The unit test above calls checkTools directly; this asserts that the handler routes its error
// into the response rather than, say, logging it or continuing with a partially built list. The
// two cases are the ones a real client hits: a strict schema (OpenAI's structured outputs flag)
// and two tools under one name.
func TestChatRefusesToolsItCannotHonour(t *testing.T) {
	const tmpl = `{%- if tools %}TOOLS{% endif %}{%- for m in messages %}[{{ m.role }}]{% endfor %}`
	srv := newTestServerWithTemplate(t, tmpl, Config{})

	cases := []struct{ name, tools, param, code string }{
		{
			name:  "function.strict",
			tools: `[{"type": "function", "function": {"name": "f", "strict": true}}]`,
			param: "tools[0].function.strict", code: "unsupported_parameter",
		},
		{
			name: "a duplicate tool name",
			tools: `[{"type": "function", "function": {"name": "f"}},
				{"type": "function", "function": {"name": "f"}}]`,
			param: "tools[1].function.name", code: "invalid_value",
		},
		{
			name:  "a non-function tool type",
			tools: `[{"type": "retrieval", "function": {"name": "r"}}]`,
			param: "tools[0].type", code: "unsupported_parameter",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			body := `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,` +
				` "tools": ` + c.tools + `}`
			resp := post(t, srv.Handler(), "/v1/chat/completions", body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != 400 {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			out := decodeBody[errorBody](t, resp)
			if out.Error.Param != c.param {
				t.Errorf("param = %q, want %q", out.Error.Param, c.param)
			}
			if out.Error.Code != c.code {
				t.Errorf("code = %q, want %q", out.Error.Code, c.code)
			}
		})
	}
}

// TestChatRefusesAToolConversationItCannotPlace is the handler-level form of checkToolMessages:
// a tool result whose call the conversation never declared.
func TestChatRefusesAToolConversationItCannotPlace(t *testing.T) {
	const tmpl = `{%- if tools %}TOOLS{% endif %}{%- for m in messages %}[{{ m.role }}]{% endfor %}`
	srv := newTestServerWithTemplate(t, tmpl, Config{})
	body := `{"messages": [
		{"role": "user", "content": "hi"},
		{"role": "tool", "content": "21", "tool_call_id": "call_never_made"}],
		"max_tokens": 1}`
	resp := post(t, srv.Handler(), "/v1/chat/completions", body)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	out := decodeBody[errorBody](t, resp)
	if out.Error.Param != "messages[1].tool_call_id" {
		t.Errorf("param = %q, want messages[1].tool_call_id", out.Error.Param)
	}
	if !strings.Contains(out.Error.Message, "no earlier assistant turn declared") {
		t.Errorf("message %q does not say which call is missing", out.Error.Message)
	}
}

// TestHandleChatRoutesEveryToolErrorToTheResponse covers the three handler branches that were
// still uncovered: the checkToolChoice, resolveToolCallMessages and checkToolMessages failures.
//
// The point is not that each function refuses -- the unit tests above prove that -- but that the
// handler turns each refusal into the response instead of dropping it. A handler that called the
// check and ignored the error would leave every unit test green and every refusal silent.
func TestHandleChatRoutesEveryToolErrorToTheResponse(t *testing.T) {
	const tmpl = `{%- if tools %}TOOLS{% endif %}{%- for m in messages %}[{{ m.role }}]{% endfor %}`
	srv := newTestServerWithTemplate(t, tmpl, Config{})
	const tool = `{"type": "function", "function": {"name": "get_weather"}}`

	cases := []struct{ name, body, param string }{
		{
			// checkToolChoice's error path, sent with a tools list so the choice has something
			// to be checked against.
			name: "an unrecognised tool_choice",
			body: `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,` +
				` "tools": [` + tool + `], "tool_choice": "sometimes"}`,
			param: "tool_choice",
		},
		{
			// resolveToolCallMessages: malformed tool_calls JSON.
			name: "tool_calls that is not an array",
			body: `{"messages": [{"role": "assistant", "content": "", "tool_calls": {"id": "c"}}],
				"max_tokens": 1}`,
			param: "messages[0].tool_calls",
		},
		{
			// checkToolMessages: an empty tool result.
			name: "an empty tool result",
			body: `{"messages": [{"role": "user", "content": "hi"},
				{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1",
					"type": "function", "function": {"name": "f", "arguments": "{}"}}]},
				{"role": "tool", "content": "   ", "tool_call_id": "call_1"}], "max_tokens": 1}`,
			param: "messages[2].content",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/chat/completions", c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != 400 {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			out := decodeBody[errorBody](t, resp)
			if out.Error.Param != c.param {
				t.Errorf("param = %q, want %q", out.Error.Param, c.param)
			}
		})
	}
}

// TestChatWithholdsToolsForChoiceNoneThroughTheHandler is the handler-level counterpart of the
// modeNone branch: the tools must not reach the prompt.
//
// Asserted through the prompt length rather than the mode, because the mode is what the unit test
// already covers and the prompt is what a caller actually experiences. The template prints "TOOLS"
// exactly when the list is non-empty, so its presence is the observable.
func TestChatWithholdsToolsForChoiceNoneThroughTheHandler(t *testing.T) {
	// A template whose output is inspectable through the fixture's own completion is not
	// available, so this asserts on the model name path instead: both requests must be served,
	// and the prompt-token count is what differs. The fixture tokenizer makes longer prompts
	// longer in tokens, so the comparison is relative rather than absolute -- the rule AGENTS.md
	// states for fixture assertions.
	const tmpl = `{%- if tools %}TOOLS:{% for t in tools %}{{ t | tojson }}{% endfor %}{% endif %}` +
		`{%- for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}`
	srv := newTestServerWithTemplate(t, tmpl, Config{})
	const tool = `{"type": "function", "function": {"name": "get_weather",
		"description": "a long description so the token counts differ measurably"}}`

	usage := func(t *testing.T, choice string) int {
		t.Helper()
		body := `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,` +
			` "temperature": 0, "tools": [` + tool + `], "tool_choice": ` + choice + `}`
		resp := post(t, srv.Handler(), "/v1/chat/completions", body)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != 200 {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d: %s", resp.StatusCode, out.Error.Message)
		}
		return decodeBody[chatResponse](t, resp).Usage.PromptTokens
	}
	auto := usage(t, `"auto"`)
	none := usage(t, `"none"`)
	if none >= auto {
		t.Errorf("tool_choice none reported %d prompt tokens and auto %d; the tools were not "+
			"withheld from the prompt", none, auto)
	}
}

// TestToolCallGateIsIndependentOfChunking pins the property a streaming gate has to have: the
// prose it releases must not depend on how the reply happened to be split into tokens.
//
// The gate decides when bytes go out, so a difference here is a difference in what a client
// reads. Every split point is tried, not just "all at once" against "a byte at a time": the two
// shapes that broke it before the fix were a push carrying a call plus the prose after it, and a
// push carrying two calls, and neither is a split a caller can avoid -- the chunking comes from the
// tokenizer, which may emit a whole `<tool_call>{...}</tool_call>` as one token.
//
// The expected value is written out rather than taken from the gate's own answer, so a gate that
// consistently dropped or duplicated a region cannot pass by agreeing with itself.
func TestToolCallGateIsIndependentOfChunking(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "prose, a call, then prose",
			in:   `Hello there. <tool_call>{"name":"get","arguments":{"x":1}}</tool_call> Done.`,
			want: "Hello there.  Done.",
		},
		{
			name: "the whole reply is one call",
			in:   `<tool_call>{"name":"f","arguments":{}}</tool_call>`,
			want: "",
		},
		{
			name: "no marker at all",
			in:   `plain prose with no call`,
			want: `plain prose with no call`,
		},
		{
			name: "angle brackets that are not a marker",
			in:   `a < b and c > d`,
			want: `a < b and c > d`,
		},
		{
			name: "two calls with prose around and between them",
			in:   `x <tool_call>{"name":"a"}</tool_call> y <tool_call>{"name":"b"}</tool_call> z`,
			want: "x  y  z",
		},
		{
			// Neither body names a tool, so the parser returns the whole text and the gate has
			// to agree: a marker pair that does not hold a call is prose, not a call to hide.
			name: "two marker pairs that hold no calls",
			in:   `<tool_call>{"a":1}</tool_call><tool_call>{"b":2}</tool_call>`,
			want: `<tool_call>{"a":1}</tool_call><tool_call>{"b":2}</tool_call>`,
		},
		{
			name: "a marker that never completes",
			in:   `trailing <tool_ca`,
			want: `trailing <tool_ca`,
		},
		{
			// No complete call in the reply, so the parser calls the whole text prose and the
			// gate has to agree: the fragment comes out at flush.
			name: "a call that never closes",
			in:   `before <tool_call>{"a":1`,
			want: `before <tool_call>{"a":1`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := streamed(t, c.in, []int{0, len(c.in)}); got != c.want {
				t.Errorf("one push released %q, want %q", got, c.want)
			}
			// Every way of cutting the reply into two pushes has to agree with that.
			for k := 0; k <= len(c.in); k++ {
				if got := streamed(t, c.in, []int{0, k, len(c.in)}); got != c.want {
					t.Errorf("split after byte %d released %q, want %q", k, got, c.want)
				}
			}
			// And so does one byte at a time, which is the worst case a tokenizer can produce.
			cuts := make([]int, 0, len(c.in)+2)
			for k := 0; k <= len(c.in); k++ {
				cuts = append(cuts, k)
			}
			if got := streamed(t, c.in, cuts); got != c.want {
				t.Errorf("one byte at a time released %q, want %q", got, c.want)
			}
			// No path may leak a *parsed* call's bytes as prose; that is the whole point of the
			// gate. A fragment the parser refused is prose by the parser's rule, so the check is
			// scoped to the shapes that contain a complete call.
			_, calls := parseToolCalls(c.in)
			leaks := strings.Contains(c.want, toolCallOpen) || strings.Contains(c.want, toolCallClose)
			if len(calls) > 0 && leaks {
				t.Fatalf("the expected text %q carries a call marker; the case is miswritten",
					c.want)
			}

		})
	}
}

// streamed pushes in up to the given byte offsets and returns everything released, including the
// flush. Offsets are ascending and end at len(src).
func streamed(t *testing.T, src string, cuts []int) string {
	t.Helper()
	var g toolCallGate
	var got strings.Builder
	for i := 0; i+1 < len(cuts); i++ {
		got.WriteString(g.push(src[cuts[i]:cuts[i+1]]))
	}
	got.WriteString(g.flush())
	return got.String()
}

// TestStreamedToolCallsAgreeWithTheBlockingBody pins the one property the two transports owe a
// caller: for the same generated tokens, the content a client accumulates and the calls it
// receives must be the ones the non-streamed body would have returned.
//
// This is not a unit test of either half -- it drives streamOneChoice and choiceObject, which are
// the two paths a request actually takes. It has to compare rather than assert fixed strings
// because the shapes must agree with each other, and pinning one side would let the other drift to
// a defensible-looking answer that differs from it.
//
// Every name below is a shape that used to disagree, measured on the code before the fix:
//   - a stop list plus tools: the stop branch continued before the tool gate was consulted, so a
//     request carrying both streamed the whole `<tool_call>{...}</tool_call>` as content and
//     reported no tool_calls, while the non-streamed body returned the call. The two branches
//     were never alternatives -- a stop list holds bytes back and a call has to be withheld --
//     and treating them as such is what this now prevents.
//   - a call that was cut off: the gate dropped the fragment and the parser returns the whole
//     text when no complete call is in it, so the non-streamed body kept "Let me check.
//     <tool_call>{"name": "f", "argum" while the stream released only "Let me check. ".
//   - prose around a call that sits on its own lines: the stream released the newlines and the
//     parser trims them.
func TestStreamedToolCallsAgreeWithTheBlockingBody(t *testing.T) {
	type shape struct {
		name  string
		reply string
		// stop and tools are what the two branches keyed on; every combination that a caller
		// can send is run, because the defect was in the combination.
		stop  []string
		tools bool
	}
	shapes := []shape{
		{
			name:  "prose, a call, then prose, with a stop list in force",
			reply: `Hello there. <tool_call>{"name": "f"}</tool_call> Done.`,
			stop:  []string{"NEVERMATCH"},
			tools: true,
		},
		{
			name:  "a reply that is only a call, with a stop list in force",
			reply: `<tool_call>{"name": "f"}</tool_call>`,
			stop:  []string{"NEVERMATCH"},
			tools: true,
		},
		{
			name:  "a call cut off mid-argument",
			reply: `Let me check. <tool_call>{"name": "f", "argum`,
			tools: true,
		},
		{
			name:  "a call on its own lines",
			reply: "\n<tool_call>{\"name\": \"f\"}</tool_call>\n",
			tools: true,
		},
		{
			name:  "prose with no call at all",
			reply: `plain prose, no call`,
			stop:  []string{"NEVERMATCH"},
			tools: true,
		},
		{
			name:  "no tools declared, so a marker is ordinary text",
			reply: `he wrote <tool_call>{"name": "f"}</tool_call> by hand`,
			stop:  []string{"NEVERMATCH"},
		},
	}
	for _, c := range shapes {
		t.Run(c.name, func(t *testing.T) {
			srv, tok := newTestServer(t)
			ids := tok.Encode(c.reply, false)
			var specs []engine.ToolSpec
			if c.tools {
				specs = []engine.ToolSpec{{Name: "f"}}
			}
			p := generateParams{chat: true, responseID: "chatcmpl-agree", tools: specs, stop: c.stop}

			// The streaming path, through the function the handler calls.
			ch := make(chan engine.Event, len(ids)+1)
			for _, id := range ids {
				ch <- engine.Event{Token: id, HasToken: true}
			}
			ch <- engine.Event{Done: true, FinishReason: "stop"}
			close(ch)
			var content, reason string
			var streamedCalls []toolCall
			if _, _, ok := srv.streamOneChoice(ch, 0, p, func(it streamItem) bool {
				content += it.delta
				if it.reason != "" {
					reason = it.reason
				}
				streamedCalls = append(streamedCalls, it.calls...)
				return true
			}); !ok {
				t.Fatal("streamOneChoice reported a failed write")
			}

			// The blocking path, on the same tokens.
			obj := srv.choiceObject(&choiceResult{out: ids, reason: "stop"}, 0, p)
			msg, ok := obj["message"].(map[string]any)
			if !ok {
				t.Fatalf("the blocking choice has no message object: %v", obj["message"])
			}
			blocked, _ := msg["content"].(string)
			blockedReason, _ := obj["finish_reason"].(string)
			var blockedCalls []toolCall
			if raw, ok := msg["tool_calls"].([]map[string]any); ok {
				for _, m := range raw {
					fn, _ := m["function"].(map[string]string)
					blockedCalls = append(blockedCalls, toolCall{
						ID: m["id"].(string), Name: fn["name"], Arguments: fn["arguments"]})
				}
			}

			// Surrounding whitespace is compared trimmed, and that is the one difference that
			// cannot be closed. parseToolCalls trims the prose around the calls it removes, and
			// the decision is made over the whole reply: whitespace before a call survives when
			// prose follows it, so nothing can be released the moment the marker appears. A
			// stream that has already sent a reply's opening newline cannot take it back once a
			// call turns up later. Every other byte has to match, which trimming the ends still
			// proves -- two strings that agree after TrimSpace differ only in those runs.
			if got, want := strings.TrimSpace(content), strings.TrimSpace(blocked); got != want {
				t.Errorf("the stream reassembled to %q and the blocking body returned %q; a client "+
					"that switches transports must not see different text", got, want)
			}
			if reason != blockedReason {
				t.Errorf("the stream reported finish_reason %q and the blocking body %q",
					reason, blockedReason)
			}
			if len(streamedCalls) != len(blockedCalls) {
				t.Fatalf("the stream carried %d calls and the blocking body %d",
					len(streamedCalls), len(blockedCalls))
			}
			for i := range streamedCalls {
				// The ids are assigned per response and so differ; everything else is the
				// call the model wrote, and has to be identical.
				if streamedCalls[i].Name != blockedCalls[i].Name ||
					streamedCalls[i].Arguments != blockedCalls[i].Arguments {
					t.Errorf("call %d streamed as %+v and blocked as %+v",
						i, streamedCalls[i], blockedCalls[i])
				}
			}
			// A *parsed* call's bytes must never reach the client as content on either
			// transport. The markers of a fragment the parser refused are content, by the
			// parser's own rule, so the check is scoped to replies that hold a complete call.
			if _, parsed := parseToolCalls(c.reply); len(parsed) > 0 && c.tools {
				if strings.Contains(content, toolCallOpen) || strings.Contains(content, toolCallClose) {
					t.Errorf("the client received a call marker as content: %q", content)
				}
			}
		})
	}
}

// TestLogprobsDescribeTheTextTheResponseShows pins the rule the logprob lists exist to follow: an
// entry may only describe bytes the caller receives.
//
// The rule was already written down -- keptTokens says a stop string's removed text gets no entries
// because "a logprob for them would describe something the caller cannot see" -- but only the stop
// cut honoured it. Parsing a tool call cuts a hole out of the *middle* of a reply, and the lists were
// built from the length of the message's content, which a hole makes meaningless. Measured on
// "Hi. <tool_call>{...}</tool_call> Done.": message.content was "Hi.  Done." and logprobs.content
// held 45 entries joining to the whole reply including the call's own bytes, while the streamed
// response carried the 9 entries that matched. A caller reading the logprob list to score the answer
// was scoring a JSON blob that was never in the answer.
//
// The shapes below are the ones that differ in where the prose is: beside a call, between two
// calls, nowhere, and the whole reply unparsed.
func TestLogprobsDescribeTheTextTheResponseShows(t *testing.T) {
	cases := []struct {
		name  string
		reply string
	}{
		{"prose around one call", `Hi. <tool_call>{"name": "f"}</tool_call> Done.`},
		{"the whole reply is a call", `<tool_call>{"name": "f"}</tool_call>`},
		{"prose only, no call", `plain prose`},
		{"two calls with prose around and between", `x <tool_call>{"name": "f"}</tool_call> y ` +
			`<tool_call>{"name": "g"}</tool_call> z`},
		{"a call cut off mid-argument", `Let me check. <tool_call>{"name": "f", "argum`},
		{"a call on its own lines", "\n<tool_call>{\"name\": \"f\"}</tool_call>\n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv, tok := newTestServer(t)
			ids := tok.Encode(c.reply, false)
			samples := make([]sampledToken, len(ids))
			for i, id := range ids {
				samples[i] = sampledToken{id: id, logprob: -1}
			}
			p := generateParams{chat: true, responseID: "chatcmpl-logprobs", logprobs: true,
				tools: []engine.ToolSpec{{Name: "f"}}}
			obj := srv.choiceObject(&choiceResult{out: ids, samples: samples, reason: "stop"}, 0, p)
			msg, ok := obj["message"].(map[string]any)
			if !ok {
				t.Fatalf("no message object: %v", obj["message"])
			}
			content, _ := msg["content"].(string)

			// The entries, in order, as a caller reading them would concatenate them.
			var joined strings.Builder
			entries := decodedLogprobEntries(t, obj["logprobs"])
			for _, e := range entries {
				joined.WriteString(e)
			}
			if got := joined.String(); got != content {
				t.Errorf("the logprob entries join to %q but message.content is %q; an entry may "+
					"only describe text the caller receives", got, content)
			}
		})
	}
}

// decodedLogprobEntries renders a chat logprobs object's per-token texts, in order.
func decodedLogprobEntries(t *testing.T, lp any) []string {
	t.Helper()
	raw, err := json.Marshal(lp)
	if err != nil {
		t.Fatalf("marshal logprobs: %v", err)
	}
	var obj struct {
		Content []struct {
			Token string `json:"token"`
		} `json:"content"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("logprobs is not the chat shape (%s): %v", raw, err)
	}
	out := make([]string, len(obj.Content))
	for i, e := range obj.Content {
		out[i] = e.Token
	}
	return out
}

// TestStreamedLogprobsCoverExactlyTheStreamedText pins the property that makes a streamed logprob
// list usable: the entries a client accumulates have to describe the text it accumulates, token for
// token and in order.
//
// The tool gate makes this non-trivial, because it withholds text that it has not yet decided
// about. A '<' may begin a call or may be prose, and the decision comes on a later push. The gate's
// text was released correctly either way; what went wrong was the log probability, which was
// attached to the chunk at the moment the token was pushed -- so a token whose text was held and
// then released as prose lost its entry. Measured on "a < b tail" with tools declared: the client
// received all 11 bytes and 8 logprob entries for 9 tokens, the missing one being the '<'.
//
// Both outcomes are covered, because a fix that only ever attaches entries would keep describing
// bytes that turned out to be a call: the reply with a real call must report entries for the prose
// alone, and one whose '<' never became a call must report its entry.
func TestStreamedLogprobsCoverExactlyTheStreamedText(t *testing.T) {
	cases := []struct {
		name  string
		reply string
	}{
		{"a '<' that turns out to be prose", `a < b tail`},
		{"a reply with no angle bracket at all", `Hi. there.`},
		{"prose, a call, and prose", `x <tool_call>{"name": "f"}</tool_call> y`},
		{"the whole reply is a call", `<tool_call>{"name": "f"}</tool_call>`},
		{"a marker that never completes", `trailing <tool_ca`},
		{"a call that never closes", `prose <tool_call>{"name": "f"}`},
		{"two calls with prose around and between",
			`a <tool_call>{"name": "f"}</tool_call> b <tool_call>{"name": "g"}</tool_call> c`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv, tok := newTestServer(t)
			ids := tok.Encode(c.reply, false)
			ch := make(chan engine.Event, len(ids)+1)
			for i, id := range ids {
				ch <- engine.Event{Token: id, HasToken: true, HasLogprob: true,
					Logprob: float32(-0.5 - float64(i)/100)}
			}
			ch <- engine.Event{Done: true, FinishReason: "stop"}
			close(ch)
			p := generateParams{chat: true, responseID: "chatcmpl-logprobs", logprobs: true,
				tools: []engine.ToolSpec{{Name: "f"}}}

			var streamed strings.Builder
			var entries strings.Builder
			count := 0
			if _, _, ok := srv.streamOneChoice(ch, 0, p, func(it streamItem) bool {
				streamed.WriteString(it.delta)
				for _, e := range decodedLogprobEntries(t, it.logprobs) {
					entries.WriteString(e)
					count++
				}
				return true
			}); !ok {
				t.Fatal("streamOneChoice reported a failed write")
			}

			if got, want := entries.String(), streamed.String(); got != want {
				t.Errorf("the logprob entries join to %q but the client received %q; an entry may "+
					"only describe text that reaches the client", got, want)
			}
			// The call's own bytes must be in neither stream. A reply that holds no complete
			// call is prose by the parser's rule, so its markers are legitimately content.
			if _, parsed := parseToolCalls(c.reply); len(parsed) > 0 {
				if strings.Contains(streamed.String(), toolCallOpen) ||
					strings.Contains(entries.String(), toolCallOpen) {
					t.Errorf("a call marker reached the client: text %q entries %q",
						streamed.String(), entries.String())
				}
			}
			if count == 0 && streamed.Len() > 0 {
				t.Error("the stream carried text but no logprob entries at all")
			}
		})
	}
}

// TestStreamedToolCallsSurviveRandomChunkings is the same invariant as the agreement tests above,
// checked over generated replies rather than hand-listed ones.
//
// The hand-written cases came from shapes that were observed to break, which makes them good
// regressions and weak discovery: the next defect will be at a boundary nobody thought to write
// down. The pieces below are the ones a tool-calling reply is built from -- both markers, their
// halves, the JSON punctuation, and prose that contains a stray '<' -- and the generator assembles
// them into replies whose marker placement is arbitrary. A fixed seed keeps a failure reproducible;
// the reply is printed with it, so a case that goes wrong can be added to the table above.
func TestStreamedToolCallsSurviveRandomChunkings(t *testing.T) {
	pieces := []string{"Hi", " ", "<", "tool_call", ">", "</", "{", `"name"`, ":", `"f"`,
		"}", "x", "y", "\n", "a < b", toolCallOpen, toolCallClose}
	rng := rand.New(rand.NewSource(20261128))
	for iter := range 60 {
		var reply strings.Builder
		for range rng.Intn(8) + 1 {
			reply.WriteString(pieces[rng.Intn(len(pieces))])
		}
		src := reply.String()
		t.Run(fmt.Sprintf("%02d", iter), func(t *testing.T) {
			srv, tok := newTestServer(t)
			ids := tok.Encode(src, false)
			if len(ids) == 0 {
				t.Skip("the fixture vocabulary encodes this reply to nothing")
			}
			p := generateParams{chat: true, responseID: "chatcmpl-random", logprobs: true,
				tools: []engine.ToolSpec{{Name: "f"}}}

			// The streamed path: what a client accumulates from the deltas, and from the entries.
			ch := make(chan engine.Event, len(ids)+1)
			for i, id := range ids {
				ch <- engine.Event{Token: id, HasToken: true, HasLogprob: true,
					Logprob: float32(-0.5 - float64(i)/100)}
			}
			ch <- engine.Event{Done: true, FinishReason: "stop"}
			close(ch)
			var content, entries strings.Builder
			var streamedCalls []toolCall
			if _, _, ok := srv.streamOneChoice(ch, 0, p, func(it streamItem) bool {
				content.WriteString(it.delta)
				for _, e := range decodedLogprobEntries(t, it.logprobs) {
					entries.WriteString(e)
				}
				streamedCalls = append(streamedCalls, it.calls...)
				return true
			}); !ok {
				t.Fatal("streamOneChoice reported a failed write")
			}

			// The blocking path on the same tokens, for the same reasons as the table test.
			obj := srv.choiceObject(&choiceResult{out: ids, reason: "stop"}, 0, p)
			msg, ok := obj["message"].(map[string]any)
			if !ok {
				t.Fatalf("no message object for %q", src)
			}
			blocked, _ := msg["content"].(string)
			var blockedCalls int
			if raw, ok := msg["tool_calls"].([]map[string]any); ok {
				blockedCalls = len(raw)
			}

			if got, want := entries.String(), content.String(); got != want {
				t.Errorf("reply %q: the logprob entries join to %q but the client received %q",
					src, got, want)
			}
			// Surrounding whitespace is the one difference the transports cannot close; see
			// TestStreamedToolCallsAgreeWithTheBlockingBody for why.
			if got, want := strings.TrimSpace(content.String()), strings.TrimSpace(blocked); got != want {
				t.Errorf("reply %q: streamed %q, blocking %q", src, got, want)
			}
			if len(streamedCalls) != blockedCalls {
				t.Errorf("reply %q: the stream carried %d calls, the blocking body %d",
					src, len(streamedCalls), blockedCalls)
			}
		})
	}
}

// TestChatLogprobsDescribeTheReplyAfterTheLeadingSpaceTrim pins the rule the blocking path has to
// follow to keep its entries aligned with the text: the entries have to describe exactly what
// DecodeCompletion renders, no more and no less.
//
// A vocabulary that spells a space as its own token makes the encoder emit one whether or not the
// caller wrote one, and the decoder drops that token's space again. So the pieces a reply is built
// from can run one space longer than the decoded text, or two, depending on what was encoded --
// measured on this fixture, Encode(" Hi") is [space space H i], whose pieces join to "  Hi" while
// Decode returns " Hi" and DecodeCompletion "Hi". The old rule dropped the leading space of the
// first token that had any bytes, which for that shape emptied the first bare-space token and left
// the second space standing: every entry then described text one byte longer than the content, and
// the whole list read as the reply shifted right by one character.
//
// The count is taken from the tokenizer rather than guessed, so the cases below cover zero, one,
// two and three leading spaces -- the shapes where a constant would be wrong for some of them.
func TestChatLogprobsDescribeTheReplyAfterTheLeadingSpaceTrim(t *testing.T) {
	cases := []string{
		"no leading space",
		" one leading space",
		"  two leading spaces",
		"   three leading spaces",
		` "quoted with a space`,
	}
	for _, reply := range cases {
		t.Run(reply, func(t *testing.T) {
			srv, tok := newTestServer(t)
			ids := tok.Encode(reply, false)
			if len(ids) == 0 {
				t.Fatal("the fixture encodes the reply to nothing")
			}
			samples := make([]sampledToken, len(ids))
			for i, id := range ids {
				samples[i] = sampledToken{id: id, logprob: -1}
			}
			p := generateParams{chat: true, responseID: "chatcmpl-trim", logprobs: true}
			obj := srv.choiceObject(&choiceResult{out: ids, samples: samples, reason: "length"}, 0, p)
			msg, ok := obj["message"].(map[string]any)
			if !ok {
				t.Fatalf("no message object: %v", obj["message"])
			}
			content, _ := msg["content"].(string)

			var joined strings.Builder
			for _, e := range decodedLogprobEntries(t, obj["logprobs"]) {
				joined.WriteString(e)
			}
			if got := joined.String(); got != content {
				t.Errorf("the entries join to %q but the content is %q; every entry describes the "+
					"reply shifted by the untrimmed space", got, content)
			}
			// The entries may not be longer than the content either: a trailing space the
			// tokenizer kept is part of the reply, and describing it is correct.
			if len(joined.String()) < len(content) {
				t.Errorf("the entries are shorter than the content: %q against %q",
					joined.String(), content)
			}
		})
	}
}

// TestStreamedLogprobsCoverTheTextWhenAStopStringIsAMarkerPrefix covers the case where the stop
// gate withholds the very characters that make up a tool call's marker, and then matches them.
//
// Both gates hold text independently. When the stop string is `<tool` and the model writes
// `<tool_call>`, the stop gate withholds each leading character on the chance it begins the stop
// string, and then matches: the client receives "x" and the call never appears. The streamed
// entries used to describe those withheld characters as they were pushed, one empty-delta chunk
// each, so the list joined to "x</too" against a content of "x". Measured over 1200 generated
// replies before the fix: 82 streamed mismatches; after it: 2, both of the same shape, where a
// character is withheld and released within one push.
//
// The empty-text case is covered too: those tokens' entries have to go out immediately, because
// the gate holds nothing on their behalf and no later release will carry them.
func TestStreamedLogprobsCoverTheTextWhenAStopStringIsAMarkerPrefix(t *testing.T) {
	cases := []struct {
		name  string
		reply string
		stop  []string
	}{
		{"the stop string cuts a call's opening marker", `x</tool_call>`, []string{"</tool"}},
		{"the stop string cuts the call and prose follows", `a <tool_call>{} b`, []string{"<tool"}},
		{"the prefix never completes, so it is prose", `a < b`, []string{"<tool"}},
		{"two prefixes, the second completing", `x <tool y <tool z`, []string{"<tool"}},
		{"no marker at all", `a b c`, []string{"<tool"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv, tok := newTestServer(t)
			ids := tok.Encode(c.reply, false)
			if len(ids) == 0 {
				t.Fatal("the fixture encodes the reply to nothing")
			}
			p := generateParams{chat: true, responseID: "chatcmpl-stop-marker", logprobs: true,
				tools: []engine.ToolSpec{{Name: "f"}}, stop: c.stop}
			ch := make(chan engine.Event, len(ids)+1)
			for i, id := range ids {
				ch <- engine.Event{Token: id, HasToken: true, HasLogprob: true,
					Logprob: float32(-0.5 - float64(i)/100)}
			}
			ch <- engine.Event{Done: true, FinishReason: "length"}
			close(ch)

			var content, entries strings.Builder
			if _, _, ok := srv.streamOneChoice(ch, 0, p, func(it streamItem) bool {
				content.WriteString(it.delta)
				for _, e := range decodedLogprobEntries(t, it.logprobs) {
					entries.WriteString(e)
				}
				return true
			}); !ok {
				t.Fatal("streamOneChoice reported a failed write")
			}
			if got, want := entries.String(), content.String(); got != want {
				t.Errorf("the entries join to %q but the client received %q; an entry may only "+
					"describe text that reaches the client", got, want)
			}
		})
	}
}
