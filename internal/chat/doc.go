// Package chat renders a conversation with a model's own chat template.
//
// A checkpoint's chat format is part of the model, not a presentation choice. Qwen2 was
// trained on "<|im_start|>user" turns and Llama 3 on "<|start_header_id|>" headers; a
// neutral transcript the model has never seen moves it off its training distribution,
// and the damage is quiet -- the model answers, badly, from a prompt it does not
// recognise. GGUF carries the format as a Jinja2 string under
// tokenizer.chat_template, so reading it is the difference between serving the model
// that was trained and serving something adjacent to it.
//
// This package is not a Jinja implementation. It evaluates a subset chosen from what
// published templates actually use:
//
//   - {% if %} / {% elif %} / {% else %} / {% endif %} with ==, !=, not, and, or,
//     in, and is defined
//   - {% for %} with loop.index0, loop.index, loop.first, loop.last, and loop.length
//   - {% set %}, including assigning to a name another statement reads
//   - {{ expression }} with string and variable operands, + concatenation, and the
//     trim, tojson and length filters
//   - whitespace control ({%- and -%}), which templates use to keep output tight
//
// Anything outside that set makes Parse report an error naming the construct it does
// not understand. That is the important property: a partial evaluator that quietly
// emitted half a template would produce a prompt that looks plausible and is wrong,
// which is exactly the failure mode this package exists to remove. Callers fall back to
// a neutral transcript when a template cannot be parsed, so an unsupported checkpoint
// degrades to the old behaviour rather than to a broken prompt.
//
// One gap in that guarantee is worth stating. A field read from a mapping that is not
// there evaluates to empty rather than to an error, because templates disagree about how
// to ask: Qwen2 writes `not message.tool_calls` and Llama 3 writes `'tool_calls' in
// message`, and both mean "no tool calls" only if a missing field is empty *and* absent.
// The rule is right for message fields and it also covers the loop object, so
// `{{ loop.depth }}` -- a Jinja field this package does not implement -- renders as the
// empty string instead of failing. The loop fields that do exist are the five listed
// above; a template using another one produces output one token short rather than an
// error, which is the failure mode the paragraph above promises not to have. It has not
// come up in a published template yet, which is why it is documented rather than fixed:
// rejecting unknown loop fields specifically would mean the loop object is validated
// while message mappings are not.
//
// The evaluation is deliberately eager and side-effect free: rendering walks the parsed
// tree once, with no user-visible state and no loops beyond for, so the cost is
// proportional to the template size and a malformed template cannot hang the server.
package chat
