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
//
//   - {% for %} with loop.index0, loop.index, loop.first, loop.last, loop.length,
//     loop.revindex, and loop.revindex0
//
//   - {% set %}, including assigning to a name another statement reads
//
//   - {{ expression }} with string and variable operands, + concatenation, - subtraction and
//     % modulo between integers, and the trim, tojson and length filters. + always concatenates,
//     including when both operands are integers: {{ 1 + 2 }} renders "12" here and "3" in
//     Jinja2. No template this package renders adds numbers, and the alternative -- deciding at
//     render time whether a value came from a literal or a field -- would make + mean two
//     things depending on where its operands came from.
//
//   - raise_exception, which is the one global implemented
//
//   - "x is string", alongside the is tests above, because Gemma 3's template uses it to tell a
//     plain content string from the multimodal list of parts. "x is iterable" answers True only
//     for a list, which is narrower than Jinja2 (where a string and a dict count too) and is what
//     the two real templates that use it depend on: Llama 3.2 writes
//     `content is mapping or content is iterable` to mean "encode this as JSON", and a string
//     matching there would come out quoted. "x is none" answers True for the none literal and for
//     a missing field, and False for the empty string -- Jinja2's rule too, and the two are easy
//     to conflate because both render as nothing.
//
//   - "a if cond else b", Jinja's inline conditional, which Gemma 3 writes inside parentheses to
//     emit its system prefix on the first turn only. The shorter "a if cond" yields an undefined
//     value and is refused rather than guessed at: its job is to abort with the
//     message the template wrote, and templates use it to assert what a conversation may be
//
//   - whitespace control ({%- and -%}), which templates use to keep output tight
//
//   - the variables a caller passes in Inputs.Variables, which is how a checkpoint's own
//     switches reach its template: Qwen3 branches on enable_thinking, and a name nobody binds
//     takes the else branch without saying so. The values are what a JSON body carries, and
//     they render as Jinja2 would render what json.loads makes of the same bytes -- including
//     the detail that an integer prints "3" while a fraction prints "2.5", which this package
//     decides from the value because encoding/json hands it a float64 for both. Template.Reads
//     answers whether a name is read anywhere in the tree, so a caller can refuse a variable
//     that would do nothing.
//
// Anything outside that set makes Parse report an error naming the construct it does
// not understand. That is the important property: a partial evaluator that quietly
// emitted half a template would produce a prompt that looks plausible and is wrong,
// which is exactly the failure mode this package exists to remove. Callers fall back to
// a neutral transcript when a template cannot be parsed, so an unsupported checkpoint
// degrades to the old behaviour rather than to a broken prompt.
//
// The is-tests cover four of Jinja2's, and two of the four answer differently from it: iterable
// and none. "is mapping" and "is string" agree. A dict and a string are both iterable in Jinja2;
// here only a sequence is, because that is the only thing {% for %} can walk. And an absent field
// reads as empty rather than as Undefined, so `field is none` is true here where Jinja2 says
// false. The answers are pinned by tests rather than left to drift, and jinja_diff.py compares
// the ones that agree against the reference while asserting the divergent pair produces its
// documented answer.
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
