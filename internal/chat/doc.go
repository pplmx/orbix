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
//     in, and the is-tests defined, none, mapping and iterable
//   - {% for %} with loop.index0, loop.index, loop.first, loop.last, loop.length,
//     loop.revindex, and loop.revindex0
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
// The is-tests cover four of Jinja2's, and three of them answer differently from it. A dict
// and a string are both iterable in Jinja2; here only a sequence is, because that is the only
// thing {% for %} can walk. And an absent field reads as empty rather than as Undefined, so
// `field is none` is true here where Jinja2 says false. The answers are pinned by tests rather
// than left to drift.
//
// The difference is not dormant: llama 3.2's template guards a tool message with
// `message.content is mapping or message.content is iterable`, and message.content is a
// string. Jinja2 takes the first branch and emits {{ content | tojson }}, which quotes it;
// here the guard is false and the content is printed as it stands. For a tool message carrying
// JSON -- the case the branch exists for -- printing it unquoted is the more useful answer, so
// this reading is preferable to Jinja2's even though it was arrived at by a rule about
// sequences rather than by intent. An earlier version of this paragraph said no published
// template used any of the three, which the llama 3.2 file in testdata disproves.
//
// One gap in that guarantee is worth stating. A field read from a mapping that is not
// there evaluates to empty rather than to an error, because templates disagree about how
// to ask: Qwen2 writes `not message.tool_calls` and Llama 3 writes `'tool_calls' in
// message`, and both mean "no tool calls" only if a missing field is empty *and* absent.
// The rule is right for message fields and it also covers the loop object, so
// `{{ loop.depth }}` -- a Jinja field this package does not implement -- renders as the
// empty string instead of failing. The loop fields that do exist are the seven listed
// above; a template using another one produces output one token short rather than an
// error, which is the failure mode the paragraph above promises not to have. It has not
// come up in a published template yet, which is why it is documented rather than fixed:
// rejecting unknown loop fields specifically would mean the loop object is validated
// while message mappings are not.
//
// The evaluation is deliberately eager and side-effect free: rendering walks the parsed
// tree once, with no user-visible state and no loop construct beyond for. There is no while,
// so no template can fail to terminate on its own.
//
// Cost is another matter, and this paragraph used to claim the two together -- "proportional
// to the template size", with a malformed template unable to hang the server. Nesting is
// multiplicative, so that is not true. Measured over a list of 8 messages: 5 nested for
// bodies took 53ms, 6 took 389ms, 7 took 2.6s from a 246-byte template, and 10 still had not
// returned after 10 seconds. A template is model-file content, so the depth is a property of
// the file an operator chose to serve rather than of a request.
//
// What a request can influence is the list length, and only through a template that already
// nests: the published templates here sit at depth 1, where cost stays linear in the message
// count. A server that must not be slowed by its own checkpoint needs the render to be bounded
// rather than the recursion to be forbidden, and that is what maxRenderNodes does -- it stops
// the exponential shape in milliseconds rather than seconds, far above anything a real
// template reaches (a 40-message conversation over a single-level loop evaluates about 200
// nodes; the budget is 100,000). maxRenderBytes is the second line of defence, for a template
// that stays within the node count and still emits something enormous.
package chat
