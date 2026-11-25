package server

import (
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
