package server

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"testing"
)

// FuzzChatContentUnmarshal checks that no request body makes the content parser panic.
//
// chatContent is the server's only hand-written unmarshaler, and it is the path every chat
// message body takes: a string, an array of text parts, or something the caller got wrong. It
// inspects the first byte to decide which, then hands the rest to the decoder, so its inputs are
// whatever bytes a client sends.
//
// The property is that it always returns: either the content or an error naming what is wrong. A
// panic here would take down the serving goroutine on a malformed request, which is the one input
// a server cannot choose.
func FuzzChatContentUnmarshal(f *testing.F) {
	seeds := []string{
		`"hello"`,
		`""`,
		`null`,
		`["a","b"]`,
		`[{"type":"text","text":"hi"}]`,
		// Wrong shapes: each should be refused rather than crash.
		`[{"type":"text"}]`,
		`[{"text":"hi"}]`,
		`[{"type":"image_url","image_url":{"url":"x"}}]`,
		`[1,2,3]`,
		`{"type":"text","text":"hi"}`,
		`123`,
		`true`,
		`[`,
		`"unterminated`,
		``,
	}
	for _, s := range seeds {
		f.Add([]byte(s))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		var c chatContent
		err := c.UnmarshalJSON(data)
		if err != nil {
			// Any error is acceptable here, and the reason is worth stating because the first
			// version of this asserted two concrete types and the fuzzer immediately produced a
			// third. UnmarshalJSON is an inner component: writeDecodeError is what classifies a
			// failure, mapping *typeMismatchError and *paramError to their own codes and
			// everything else -- a *json.SyntaxError from a truncated body, say -- to
			// invalid_json. So the property this layer owes is that it returns an error carrying
			// a message, not that it returns one of a particular shape.
			if err.Error() == "" {
				t.Fatal("UnmarshalJSON returned an error with no message")
			}
			return
		}
		// Accepted, so the bytes described a content value. Nothing further is asserted about
		// the text -- it is whatever the input said -- only that the call returned a usable
		// value rather than leaving the receiver in a state a caller would trip over.
		_ = string(c)
	})
}

// TestChatContentRoundTripsItsAcceptedForms keeps the fuzz's accept branch reachable.
//
// If every input were refused the fuzz above would only exercise the error paths, and the success
// path it claims to cover would never run. That is what happened to the llama fuzz in an earlier
// round; this test pins the accepted forms so a change that started refusing them all would fail
// here rather than silently hollow out the fuzz.
func TestChatContentRoundTripsItsAcceptedForms(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{`"hello"`, "hello"},
		{`""`, ""},
		{`null`, ""},
		{`[{"type":"text","text":"hi"}]`, "hi"},
		{`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`, "ab"},
		{`[]`, ""},
		// A bare string inside the array is refused: the parts are typed objects, and reading
		// its text anyway would be guessing. Asserted below rather than here.
	}
	for _, c := range cases {
		var got chatContent
		if err := got.UnmarshalJSON([]byte(c.in)); err != nil {
			t.Errorf("UnmarshalJSON(%s): %v", c.in, err)
			continue
		}
		if string(got) != c.want {
			t.Errorf("UnmarshalJSON(%s) = %q, want %q", c.in, got, c.want)
		}
	}

	// And the encoder, so the value survives a round trip through encoding/json.
	var c chatContent
	if err := json.Unmarshal([]byte(`[{"type":"text","text":"x"},{"type":"text","text":"y"}]`), &c); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if string(c) != "xy" {
		t.Errorf("through the decoder = %q, want %q", c, "xy")
	}
	// The refusal the table above left out: an array of bare strings. Its error is the type
	// mismatch, because each element is decoded into an object.
	var bad chatContent
	err := json.Unmarshal([]byte(`["x","y"]`), &bad)
	if err == nil {
		t.Fatal("an array of bare strings was accepted")
	}
	if _, ok := err.(*typeMismatchError); !ok {
		t.Errorf("the refusal is %T, want *typeMismatchError", err)
	}
}

// FuzzDecodeJSON checks that no request body can take the decode path down or leave it returning
// a blank answer.
//
// The existing target covers chatContent, which is one field of one request. This covers the
// envelope: decodeJSON is what every POST on both endpoints goes through before anything else
// looks at the body, so its inputs are whatever bytes a client sends -- including the ones the
// decoder has to give up on. Three properties, each matching something the layer below can
// violate in a way the caller would trip over:
//
//   - it returns, so a malformed body cannot take the serving goroutine down;
//   - an error always carries a message, because writeDecodeError surfaces it to the caller and
//     an empty one produces an envelope with nothing in it;
//   - a nil error leaves the destination usable, since the handlers read it immediately.
//
// The destination is a chatRequest rather than `any` so the fuzzer reaches the fields that have
// their own unmarshalers -- messages, tool_choice, the numeric limits -- instead of stopping at
// "this is valid JSON".
func FuzzDecodeJSON(f *testing.F) {
	seeds := []string{
		`{}`,
		`{"messages":[{"role":"user","content":"hi"}]}`,
		`{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
		`{"messages":[],"max_tokens":8,"temperature":0.5,"top_p":0.9,"n":2}`,
		`{"model":"m","stream":true,"stream_options":{"include_usage":true}}`,
		`{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function",` +
			`"function":{"name":"f","parameters":{"type":"object"}}}]}`,
		`{"messages":[{"role":"user","content":"hi"}],"tool_choice":"required"}`,
		`{"prompt":"hi","suffix":null,"echo":true,"best_of":2}`,
		// Wrong shapes and truncations: each should be refused rather than crash.
		`{"messages":"x"}`,
		`{"messages":[1]}`,
		`{"messages":[{"role":1}]}`,
		`{"max_tokens":"8"}`,
		`{"max_tokens":-1}`,
		`{"temperature":true}`,
		`{"messages":[{"role":"user","content":`,
		`null`,
		`[]`,
		``,
	}
	for _, s := range seeds {
		f.Add([]byte(s))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		var req chatRequest
		w := httptest.NewRecorder()
		r := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(data))
		err := decodeJSON(w, r, &req, 1<<20)
		if err != nil {
			if err.Error() == "" {
				t.Fatal("decodeJSON returned an error with no message")
			}
			return
		}
		// Accepted, so the handlers read the destination. Touching the fields that have their own
		// unmarshalers is what would surface a receiver left half-built.
		_ = len(req.Messages)
		for _, m := range req.Messages {
			_ = string(m.Content)
		}
		_ = req.MaxTokens
	})
}
