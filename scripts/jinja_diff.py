"""Check the chat template subset against real Jinja2.

This is a cross-validation tool, not part of the engine. The Go package in internal/chat
evaluates a subset of Jinja2 chosen from what published chat templates use; its own tests
compare against strings typed by hand, which can share a blind spot with the code. Real
Jinja2 is the independent side, so this script renders the same templates with the same
inputs through both and reports the disagreements.

Run it with uv, which provides Jinja2 without touching the system Python:

    uv run --with jinja2 python3 scripts/jinja_diff.py            # built-in corpus
    uv run --with jinja2 python3 scripts/jinja_diff.py --verbose  # list every case and its verdict

    uv run --with jinja2 --with gguf python3 scripts/jinja_diff.py \
        --template-file /tmp/llama32.gguf                         # a checkpoint's own template

The built-in corpus holds hand-written approximations of published templates, which makes it
a check on the constructs this package implements. `--template-file` replaces it with the
template read out of a GGUF file's `tokenizer.chat_template`, which is the different question
of whether the package can evaluate what a model actually ships. The gap is not small: the
`llama3` entry below is one line, while the Llama 3.2 checkpoint's template is 3827 characters
that branch on tools, a date and a system message. Both modes are worth running -- the corpus
names a broken construct precisely, and the checkpoint proves the shape of a shipped template.

It builds a small Go program that renders each case through the package under test, runs
both sides, and diffs the text. Exit status is 1 if anything disagrees, so it can be used as
a gate by hand.

A disagreement is one of two things, and the output says which:

  - a bug in the subset -- the Go side produced something Jinja2 does not;
  - an unsupported construct -- the package documents a list of Jinja2 features it does
    not implement, and a template using one is expected to be refused rather than rendered.
    Some are refused at parse time (a macro, a filter block) and some at render time (printing
    a message dict, iterating a string), and both count as designed behaviour. Those cases are
    reported separately, because "refused" is the design and "rendered differently" is not.

The corpus is in this file rather than in testdata: it belongs with the tool that reproduces
it, and a template corpus without its reference output is a list of strings with no claim
attached.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import jinja2

MESSAGES = {
    "user": [{"role": "user", "content": "What is 2+2?"}],
    "system_user": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ],
    "multi": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ],
    "trimmy": [{"role": "user", "content": "  spaced  "}],
    # Content the ASCII corpora above cannot catch. The length filter returned a byte
    # count for two years' worth of these cases because every message was ASCII; a
    # template that branches on a length, or a renderer that slices text, shows it here.
    "cjk": [{"role": "user", "content": "\u4f60\u597d\u4e16\u754c"}],
    "emoji": [{"role": "user", "content": "hi \U0001f389"}],
    "empty": [{"role": "user", "content": ""}],
    "newline": [{"role": "user", "content": "a\nb"}],
    "quoted": [{"role": "user", "content": 'he said "hi"'}],
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

# Templates that ship with checkpoints, abbreviated only where they use a construct the
# package documents as unsupported.
TEMPLATES = {
    "qwen2": """{{- '<|im_start|>system\\n' }}
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
{{- 'You are a helpful assistant.' }}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- for message in messages %}
{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- elif message.role == "assistant" %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- endif %}""",
    "llama3": """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}""",
    "chatml": """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}""",
    "gemma": """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\\n\\n' }}{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if loop.first %}{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% else %}{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}""",
    "mistral": """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] | trim + eos_token }}{% endif %}{% endfor %}""",
    "phi3": """{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\\n' + message['content'] + '<|end|>\\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\\n' + message['content'] + '<|end|>\\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\\n' + message['content'] + '<|end|>\\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<|assistant|>\\n'}}{% endif %}""",
    "deepseek": """{% for message in messages %}{% if loop.first %}{{ bos_token }}{% endif %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}""",
}

# One construct each, so a disagreement names the construct.
SUBSET = {
    "if_eq": """{% if messages[0]['role'] == 'user' %}yes{% else %}no{% endif %}""",
    "if_ne": """{% if messages[0]['role'] != 'system' %}not-system{% endif %}""",
    "if_not": """{% if not add_generation_prompt %}nogen{% else %}gen{% endif %}""",
    "if_and": """{% if add_generation_prompt and messages %}both{% else %}one{% endif %}""",
    "if_or": """{% if not add_generation_prompt or messages %}sat{% else %}unsat{% endif %}""",
    "if_in": """{% if 'user' in messages[0]['role'] %}has-user{% endif %}""",
    # A bool and a str are never equal in Jinja2, which is how the Go side got this wrong: it
    # compared truth values, and every non-empty string is truthy.
    "if_eq_bool_str": """{% if true == 'false' %}T{% else %}F{% endif %}""",
    "if_eq_bool_str2": """{% if true == 'true' %}T{% else %}F{% endif %}""",
    "if_eq_bool_bool": """{% if true == true %}T{% else %}F{% endif %}""",
    "if_isdefined": """{% if bos_token is defined %}bos-set{% else %}bos-absent{% endif %}""",
    "for_index": """{% for m in messages %}{{ loop.index }}{{ m['role'] }};{% endfor %}""",
    "for_index0": """{% for m in messages %}{{ loop.index0 }}{% endfor %}""",
    "for_first_last": """{% for m in messages %}{{ loop.first }}{{ loop.last }};{% endfor %}""",
    "for_bool_tojson": """{% for m in messages %}{{ loop.first|tojson }};{% endfor %}""",
    # The loop attributes doc.go lists beside first/last. They were implemented and asserted
    # against hand-written expectations, but nothing compared them with Jinja2, so a rule that
    # agreed with the Go tests and not with the reference would have gone unnoticed.
    "for_length": """{% for m in messages %}{{ loop.length }};{% endfor %}""",
    "for_revindex": """{% for m in messages %}{{ loop.revindex }};{% endfor %}""",
    "for_revindex0": """{% for m in messages %}{{ loop.revindex0 }};{% endfor %}""",
    # is mapping / iterable / none / string are the four doc.go names explicitly. Real templates
    # branch on them: "x is string" is how Gemma 3 tells a plain content string from the
    # multimodal list of parts, and the other three appear in Qwen's tool templates.
    "is_mapping": """{{ messages[0] is mapping }}""",
    "is_mapping_false": """{{ messages is mapping }}""",
    "is_iterable": """{{ messages is iterable }}""",
    "is_none": """{{ none is none }}""",
    "is_none_false": """{{ messages[0]['content'] is none }}""",
    "is_string": """{{ messages[0]['content'] is string }}""",
    "is_string_false": """{{ messages[0] is string }}""",
    "set_var": """{% set x = 'hello' %}{{ x }}""",
    "set_concat": """{% set x = 'a' + 'b' %}{{ x }}""",
    "set_reassign": """{% set x = 'a' %}{% set x = 'b' %}{{ x }}""",
    "set_in_loop": """{% for m in messages %}{% set c = m['content'] %}{{ c }}{% endfor %}""",
    "expr_concat": """{{ 'a' + 'b' + 'c' }}""",
    "expr_field": """{{ messages[0]['content'] }}""",
    "expr_dot": """{{ messages[0].role }}""",
    "expr_trim": """[{{ '  x  ' | trim }}]""",
    "expr_trim_field": """[{{ messages[0]['content'] | trim }}]""",
    "expr_tojson_msg": """{{ messages[0] | tojson }}""",
    "expr_tojson_list": """{{ messages | tojson }}""",
    "expr_tojson_scalar": """{{ 'a' | tojson }}""",
    # Both spellings of the indent argument. Jinja2 renders them identically, and the Go side
    # dropped the positional one for a while: it was parsed and then ignored.
    "expr_tojson_indent_kw": """{{ messages[0] | tojson(indent=2) }}""",
    "expr_tojson_indent_pos": """{{ messages[0] | tojson(2) }}""",
    # The two arithmetic operators doc.go lists beside +. They were implemented for one real
    # template each -- Gemma's alternation check is `loop.index0 % 2 == 0` -- and neither was
    # compared against Jinja2.
    "expr_sub": """{{ 5 - 2 }}""",
    "expr_sub_negative": """{{ 2 - 5 }}""",
    "expr_sub_zero": """{{ 3 - 3 }}""",
    "expr_mod": """{{ 7 % 3 }}""",
    "expr_mod_zero_operand": """{{ 7 % 7 }}""",
    "expr_neg_literal": """{{ -3 }}""",
    "expr_len": """{{ messages | length }}""",
    "expr_nl": """{{ 'a\\nb' }}""",
    "ws_control": """a  {{- 'b' -}}  c""",
    "ws_left": """a  {{- 'b' }}  c""",
    "ws_right": """a  {{ 'b' -}}  c""",
    "ws_block": """a  {%- if true -%}  b  {%- endif -%}  c""",
    "ws_newline": "a\n  {{- 'b' }}\n  c",
    "truthy_empty_str": """{% if '' %}T{% else %}F{% endif %}""",
    "truthy_nonempty_str": """{% if 'x' %}T{% else %}F{% endif %}""",
    "truthy_undefined": """{% if nosuchthing %}T{% else %}F{% endif %}""",
    "truthy_flag": """{% if add_generation_prompt %}T{% else %}F{% endif %}""",
    "defined_guard": """{% if not tools is defined %}NO_TOOLS{% endif %}""",
    "elif_false": """{% if false %}A{% elif true %}B{% else %}C{% endif %}""",
    "no_else": """{% if false %}A{% endif %}END""",
    # The inline conditional, in both arms plus a condition over the conversation. It moved
    # here from UNSUPPORTED when Gemma 3's template needed it: this tool kept expecting a
    # refusal for months after the construct was implemented, so every run exited 1 on a case
    # that was working. An expectation table that is not updated with the feature is a gate
    # that reports a regression that is not there -- and, worse, hides the real one behind it.
    "inline_if_true": """{{ 'a' if add_generation_prompt else 'b' }}""",
    "inline_if_false": """{{ 'a' if not add_generation_prompt else 'b' }}""",
    "inline_if_field": """{{ messages[0]['role'] if messages else 'empty' }}""",
    # Precedence: the conditional binds looser than comparison and concatenation, so every
    # one of these is a single expression rather than a parse error.
    "inline_if_precedence": """{{ 'x' if messages and messages[0]['role'] == 'user' else 'y' }}""",
    "inline_if_nested": """{{ 'a' if messages else ('b' if add_generation_prompt else 'c') }}""",
    # The constructs Qwen3's template needed, one case each so a disagreement names the one
    # that broke. They were added to the package together, and without corpus entries the only
    # evidence that they agree with Jinja2 is the whole-template comparison -- which reports
    # "these two 4 KB strings differ" and leaves the reader to find where.
    #
    # namespace() is the one piece of mutable state in the language, and the for-body case is
    # why it exists: a plain {% set %} there binds in the body's frame and is gone by the time
    # the template reads it. The field assignment outside a loop is the same rule seen from the
    # other side -- there is no frame to escape, so the plain set would have worked, and a
    # renderer that only handled the loop case would pass the interesting test and fail this.
    "namespace_field": """{% set ns = namespace(x=1) %}{{ ns.x }}""",
    "namespace_out_of_loop": """{% set ns = namespace(last=-1) %}"""
                             """{% for m in messages %}{% set ns.last = loop.index0 %}"""
                             """{% endfor %}{{ ns.last }}""",
    # Python's strip family takes a character set, not a substring: 'xyhixy'.strip('xy') is
    # "hi", while a TrimPrefix reading gives "hixy" and a Trim reading gives "hi" by accident.
    # Measured: changing strings.Trim to strings.TrimPrefix in the evaluator turns seven cases
    # red, and this is the one that names the rule. The no-argument case is separate because it
    # defaults to whitespace, and lstrip/rstrip are separate from strip so a rule applied on the
    # wrong side is caught.
    "method_strip_charset": """{{ 'xyhixy'.strip('xy') }}""",
    "method_strip_no_arg": """[{{ '  pad  '.strip() }}]""",
    "method_lstrip": """[{{ 'xxhixx'.lstrip('x') }}]""",
    "method_rstrip": """[{{ 'xxhixx'.rstrip('x') }}]""",
    "method_split": """{{ 'a,b,c'.split(',') | length }}""",
    "method_split_index": """{{ 'a,b,c'.split(',')[1] }}""",
    # A separator that is more than one character, and none at all. Found by injection:
    # replacing the separator with "split on its first rune" left the gate green, because every
    # case above splits on a single-character comma. Qwen3 splits a reply on a multi-character
    # marker, where the first-rune reading produces fragments that look plausible.
    "method_split_multichar": """{{ 'a--b'.split('--') | length }}""",
    "method_split_no_arg": """{{ '  a  b '.split() | length }}""",
    # Slices on a string and on a sequence. Jinja2 spells the open end by leaving it out, and
    # a parser that reads "[:n]" as "the first index expression" has to treat the missing
    # bound as an absent one rather than as zero -- for [2:] those are not the same.
    "slice_string_head": """{{ 'abcdef'[:3] }}""",
    "slice_string_tail": """{{ 'abcdef'[2:] }}""",
    "slice_list_tail": """{{ messages[1:] | length }}""",
    # Negative bounds and out-of-range ones, which clamp rather than fail in Python. Found by
    # injection: removing the `start += n` that turns a negative start into an offset left the
    # gate green, because every case above uses a non-negative bound. A template that slices
    # from the end (`current_content[-8:]`) would then render the whole string instead.
    "slice_negative_start": """{{ 'abcdef'[-2:] }}""",
    "slice_negative_end": """{{ 'abcdef'[:-2] }}""",
    "slice_past_end": """{{ 'abcdef'[2:99] }}""",
    "slice_past_start": """{{ 'abcdef'[-99:] }}""",
    # The ordering comparisons. They were missing entirely: the parser stopped at the "<" and
    # reported it as unexpected, so a template branching on `start_pos < 0` was refused whole.
    # Numbers compare numerically and text lexically, which is two cases rather than one.
    "cmp_lt": """{{ 1 < 2 }}""",
    "cmp_gt": """{{ 2 > 3 }}""",
    "cmp_le": """{{ 2 <= 2 }}""",
    "cmp_ge": """{{ 3 >= 4 }}""",
    # Both comparisons at the equality boundary, which is the only place a strict operator
    # differs from a non-strict one. Found by injection: changing "<" to "<=" in the evaluator
    # left the gate green, because every case above compares unequal operands. The same kind of
    # gap it exposes is why each of these four has its own case rather than one shared shape.
    "cmp_lt_equal": """{{ 2 < 2 }}""",
    "cmp_gt_equal": """{{ 2 > 2 }}""",
    "cmp_string": """{% if 'a' < 'b' %}lt{% endif %}""",
    # "is true" tests that a value *is* the boolean, which is why the first case is False: an
    # integer 1 is truthy but is not True. A package that answered with a truth test would
    # pass the three obvious cases and get this one wrong.
    "is_true_of_int": """{{ 1 is true }}""",
    "is_true_of_true": """{{ true is true }}""",
    "is_false_of_false": """{{ false is false }}""",
    "is_false_of_zero": """{{ 0 is false }}""",
    # Mistral's published templates open with this chain, and until selectattr existed they did
    # not parse at all. SUBSET takes the source alone: the expected output is whatever jinja2
    # renders it to, computed at run time, so these cannot agree with a wrong expectation the way
    # a hand-written one can -- the first version of this block carried hard-coded values that
    # were wrong because the corpus has two user turns, not one.
    "selectattr_equalto":
        """{{ messages | selectattr('role', 'equalto', 'user') | list | length }}""",
    "selectattr_no_match":
        """{{ messages | selectattr('role', 'equalto', 'nobody') | list | length }}""",
    # One argument defaults the test to `boolean`, which asks whether the field is truthy.
    "selectattr_truthy": """{{ messages | selectattr('content') | list | length }}""",
    "list_splits_a_string": """{{ "abc" | list | length }}""",

}

# Templates using constructs the package documents as unsupported. The expected Go result is
# a parse error, which is what makes "unsupported" a designed refusal rather than a wrong
# render. Jinja2 renders all of them.
# DIVERGENT holds templates whose Go output is deliberately not what Jinja2 produces. Each entry
# names what this package does and why, and the script asserts both halves: that the output is the
# documented one, and that it still differs from Jinja2. Without the second assertion a change that
# accidentally made the two agree would be reported as an improvement, when it would mean the
# documented behaviour had changed.
#
# None of these is about + any more. Two integers add on both sides now, and the one mixed-pair
# case -- "a" + 1, which this package renders as "a1" and Jinja2 refuses with TypeError -- cannot
# live in this table: a DIVERGENT entry is compared against Jinja2's output, and Jinja2 produces
# none for it. It is asserted by TestPlusAddsNumbersAndJoinsEverythingElse in template_test.go
# instead, and the skip this table used to generate ("2 cases skipped because Jinja2 itself
# failed") is the reason: a case that runs neither side reads as coverage and is the opposite.
DIVERGENT = {
    # A message dict counts as iterable in Jinja2 and not here. doc.go records the reason: Llama
    # 3.2 writes `content is mapping or content is iterable` to mean "encode this as JSON", and a
    # dict matching there would take the record's keys apart instead.
    "is_iterable_dict": ("""{{ messages[0] is iterable }}""", "False"),
    # An absent field is none here and Undefined in Jinja2, where reading a missing key yields
    # something that is not none. Measured on the reference: False for an absent key, True for an
    # explicit null. doc.go records which templates depend on it -- Qwen2 writes
    # `not message.tool_calls` and Llama 3 writes `'tool_calls' in message`, so neither can be
    # answered by raising.
    "is_none_absent": ("""{{ messages[0].nosuchfield is none }}""", "True"),
    # The same divergence, seen through another filter. Jinja2's Undefined stringifies to "" and
    # its None to "None"; this package treats an absent field as none for the reason above, so
    # string gives "None" where Jinja2 gives "". Kept here rather than in SUBSET so the pair stays
    # together: an editor fixing one without the other would make the two disagree.
    "string_of_absent_field": ("""{{ messages[0].nosuchfield | string }}""", "None"),
}


UNSUPPORTED = {
    # Jinja's short form, which yields Undefined when the condition is false. Jinja2 renders it
    # as the empty string; here it is refused at parse time, which is the documented behaviour:
    # an undefined result that reached output would put a hole in the prompt, and no published
    # template relies on it. The refusal names the construct rather than rendering something
    # inert, so a reader learns which form to write instead.
    "inline_if_no_else": """{{ 'a' if add_generation_prompt }}""",
    "filter_block": """{% filter upper %}x{% endfilter %}""",
    # The bag itself, not a field of it. Printing a namespace is a template bug, and this
    # package refuses it rather than emitting Go's map formatting: Jinja2 prints a repr that
    # no template wants, so there is nothing to agree with. Each field read back is in SUBSET
    # -- this entry stayed behind when namespace() moved there, which is how the gate spent a
    # release reporting a refusal for a construct that was implemented.
    "print_a_namespace": """{% set ns = namespace(x=1) %}{{ ns }}""",
    "macro": """{% macro m() %}x{% endmacro %}{{ m() }}""",
    "for_else": """{% for m in messages %}{{ m }}{% else %}none{% endfor %}""",
    "tuple_unpack": """{% for k, v in tools %}{{ k }}{% endfor %}""",
    "dict_literal": """{{ {'a': 1} }}""",
    "print_a_message": """{{ messages[0] }}""",
    # Jinja2 iterates a string character by character; this subset does not implement that and
    # refuses rather than yielding no turns. The refusal is at render time, which the tool treats
    # the same as a parse-time one -- both mean "this construct is not supported".
    "for_over_string": """{% for c in 'abc' %}{{ c }}{% endfor %}""",
}


class JinjaTemplateError(Exception):
    """Raised by the raise_exception global, mirroring Jinja2's own exception type.

    Jinja2 does not define raise_exception itself; the environment's caller does. A template
    that calls it is asserting something about the conversation, so the message it wrote is the
    useful part and this carries it rather than replacing it with a generic failure.
    """


def oracle(template, messages, tools, add_gen, bos, eos):
    env = jinja2.Environment(keep_trailing_newline=True)
    # raise_exception is the one global a published template calls, and the caller is what
    # supplies it -- transformers' renderer binds it, and so does this package. Without it in
    # the reference environment Gemma 2's "System role not supported" guard raises
    # UndefinedError instead, which reads as a jinja2 failure rather than as the template
    # refusing the conversation. That made every gemma2 system-prompt case a skip.
    def raise_exception(msg=""):
        raise JinjaTemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    ctx = {
        "messages": messages,
        "add_generation_prompt": add_gen,
        "bos_token": bos,
        "eos_token": eos,
    }
    if tools is not None:
        ctx["tools"] = tools
    try:
        return True, env.from_string(template).render(**ctx)
    except Exception as e:  # noqa: BLE001 - the message is what matters
        return False, f"{type(e).__name__}: {e}"


def asset_cases(repo_root):
    """Cases from the templates committed under internal/chat/testdata.

    These are not approximations: each is a byte-for-byte copy of a checkpoint's
    `tokenizer.chat_template`, and the repository's TestTemplateAssetsAreExact pins the hash. So
    the question here is the strong one -- can this package evaluate what that model ships --
    without needing the checkpoint on disk.

    They earn their place in a way the hand-written corpus cannot. Qwen3 0.6B's template uses
    `messages[::-1]` and `.startswith()`, added for it; the construct cases in SUBSET did not
    catch their absence because they never render the template that needed them. The whole
    template is what fails when a construct is missing, and that is the failure this reports.

    A missing testdata directory is not an error: the tool is documented to work on a partial
    checkout too, and the built-in corpus still runs.
    """
    root = os.path.join(repo_root, "internal", "chat", "testdata")
    if not os.path.isdir(root):
        return []
    cases = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jinja"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as f:
            tmpl = f.read()
        for msgname, msgs in MESSAGES.items():
            cases.append({
                "name": f"asset/{name}/{msgname}", "template": tmpl, "messages": msgs,
                "add_generation_prompt": True, "bos_token": "", "eos_token": "",
                "kind": "asset",
            })
        # The Go side takes a tool's parameters as text and prints them verbatim -- documented
        # in render.go, and deliberate: re-encoding would reorder the model's own keys and
        # change the prompt. Jinja2 re-serialises the dict it was handed, so the two only agree
        # if this side supplies the text json.dumps would produce. Supplying the compact
        # original instead reported the spacing of the input as a rendering difference.
        cases.append({
            "name": f"asset/{name}/tools", "template": tmpl, "messages": MESSAGES["multi"],
            "tools": TOOLS, "add_generation_prompt": True,
            "bos_token": "", "eos_token": "", "kind": "asset",
        })
    return cases


# ASSET_DIVERGENT names the asset cases whose Go output is deliberately not what Jinja2
# produces, with the reason the two cannot agree rather than a note that they do not.
#
# There is one, and it is a data-type difference rather than a rendering bug: Llama 3.2's
# template prints its tools with `tojson(indent=4)`, which re-serialises the whole structure.
# The Go side holds a tool's parameters as text -- so that a client's schema reaches the model
# byte for byte instead of being re-encoded and reordered -- and text cannot be re-indented, so
# the parameters object stays on one line inside an otherwise indented document. Jinja2, having
# parsed the parameters into a dict, indents them too.
#
# The alternative fix is to parse and re-encode every schema, which would change the prompt for
# every model to make one model's debug formatting match. That is the wrong trade, so the
# difference is recorded here: this table is what makes "we know" different from "we missed it".
ASSET_DIVERGENT = {
    "asset/llama32-instruct.jinja/tools": (
        "a tool's parameters are text and keep their original spacing, so tojson(indent=4) "
        "cannot indent them; see chat.Tool.Parameters"
    ),
}


def build_cases(repo_root="."):
    cases = asset_cases(repo_root)
    for name, tmpl in TEMPLATES.items():
        for msgname, msgs in MESSAGES.items():
            for add_gen in (False, True):
                for bos, eos in (("", ""), ("<|begin_of_text|>", "<|eot_id|>")):
                    cases.append({
                        "name": f"{name}/{msgname}/gen={add_gen}/bos={bool(bos)}",
                        "template": tmpl, "messages": msgs,
                        "add_generation_prompt": add_gen, "bos_token": bos, "eos_token": eos,
                        "kind": "template",
                    })
        cases.append({
            "name": f"{name}/tools", "template": tmpl, "messages": MESSAGES["multi"],
            "tools": TOOLS, "add_generation_prompt": True,
            "bos_token": "<s>", "eos_token": "</s>", "kind": "template",
        })
    for name, tmpl in SUBSET.items():
        # Every message shape, not just the two ASCII ones: the length filter counted bytes
        # instead of characters, and the only case that shows it is 'expr_len' over a
        # non-ASCII message -- which the two-shape loop never built.
        for msgname in ("user", "multi", "cjk", "emoji", "empty", "newline", "quoted"):
            cases.append({
                "name": f"subset/{name}/{msgname}", "template": tmpl,
                "messages": MESSAGES[msgname], "add_generation_prompt": True,
                "bos_token": "<s>", "eos_token": "</s>", "kind": "subset",
            })
    for name, (tmpl, want) in DIVERGENT.items():
        cases.append({
            "name": f"divergent/{name}", "template": tmpl, "want": want,
            "messages": MESSAGES["multi"], "tools": TOOLS, "add_generation_prompt": True,
            "bos_token": "<s>", "eos_token": "</s>", "kind": "divergent",
        })
    for name, tmpl in UNSUPPORTED.items():
        cases.append({
            "name": f"unsupported/{name}", "template": tmpl,
            "messages": MESSAGES["multi"], "tools": TOOLS, "add_generation_prompt": True,
            "bos_token": "<s>", "eos_token": "</s>", "kind": "unsupported",
        })
    return with_oracle(cases)


def with_oracle(cases):
    """Render every case with real Jinja2 and record the answer.

    Kept separate from corpus construction so cases that come from a checkpoint go through the
    same reference implementation as the built-in ones. A case Jinja2 itself refuses keeps
    oracle_ok False and is reported rather than compared: there is no reference answer to
    compare against, and inventing one would turn a gap in the corpus into a passing test.
    """
    for c in cases:
        ok, out = oracle(c["template"], c["messages"], c.get("tools"),
                         c["add_generation_prompt"], c["bos_token"], c["eos_token"])
        c["oracle"] = out if ok else ""
        c["oracle_ok"] = ok
        # The reason travels with the case so the skip list names it. Before this the report
        # printed an empty pair of parentheses, which says "jinja2 failed" and nothing about
        # why -- and for the gemma2 asset the why is the interesting part: its template refuses
        # a system turn outright, so there is no prompt for the two sides to disagree about.
        c["oracle_error"] = "" if ok else out
    return cases


GO_PROGRAM = r'''
package main

import (
	"encoding/json"
	"os"

	"github.com/pplmx/orbix/internal/chat"
)

// openAITool is the shape the case payload carries, which is the same one a client sends:
// the function nested under "function". chat.Tool is flat, because the template only ever
// reads name, description and parameters, so the program flattens rather than making the
// evaluator's type carry a nesting it does not use.
//
// Decoding straight into chat.Tool silently produced three empty strings -- its fields have no
// JSON tags, so "name" never matched "Name" -- and the corpus reported the tools cases as
// agreeing while both sides rendered a tool with no name. The asset cases are what exposed it:
// their templates print the parameters, so an empty tool stops matching Jinja2's answer.
type openAITool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		// Parameters is text in this payload rather than an object: the program passes it to
		// chat.Tool verbatim, and the corpus supplies the form json.dumps would write. A
		// RawMessage here would re-encode an object and lose exactly the spacing the
		// comparison is about.
		Parameters string `json:"parameters"`
	} `json:"function"`
}

func (t openAITool) flat() chat.Tool {
	return chat.Tool{
		Name:        t.Function.Name,
		Description: t.Function.Description,
		Parameters:  t.Function.Parameters,
	}
}

type tc struct {
	Name     string         `json:"name"`
	Template string         `json:"template"`
	Messages []chat.Message `json:"messages"`
	Tools    []openAITool   `json:"tools"`
	AddGen   bool           `json:"add_generation_prompt"`
	BOS      string         `json:"bos_token"`
	EOS      string         `json:"eos_token"`
}

func main() {
	var cases []tc
	if err := json.NewDecoder(os.Stdin).Decode(&cases); err != nil {
		panic(err)
	}
	type res struct {
		Text  string `json:"text"`
		Error string `json:"error"`
	}
	out := make([]res, len(cases))
	for i, c := range cases {
		tmpl, err := chat.Parse(c.Template)
		if err != nil {
			out[i] = res{Error: err.Error()}
			continue
		}
		tools := make([]chat.Tool, len(c.Tools))
		for j, t := range c.Tools {
			tools[j] = t.flat()
		}
		text, err := tmpl.Render(c.Messages, chat.Inputs{
			AddGenerationPrompt: c.AddGen,
			BOSToken:            c.BOS,
			EOSToken:            c.EOS,
			Tools:               tools,
		})
		if err != nil {
			out[i] = res{Error: err.Error()}
			continue
		}
		out[i] = res{Text: text}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(out); err != nil {
		panic(err)
	}
}
'''


def dumped_tools(tools):
    """Convert a case's tools into the shape the Go program decodes.

    Each parameters object becomes the text json.dumps would write for it, which is what the
    Go side expects: chat.Tool.Parameters is text, emitted verbatim so the model sees the
    schema the client sent rather than a re-encoding of it.

    Doing this at the payload boundary keeps both sides honest. Converting inside the Go
    program would mean the comparison no longer exercises the code that a server reaches, and
    converting the case would hide the difference rather than account for it.
    """
    out = []
    for t in tools:
        fn = dict(t.get("function", {}))
        if "parameters" in fn:
            fn["parameters"] = json.dumps(fn["parameters"], sort_keys=True)
        out.append({"type": t.get("type", "function"), "function": fn})
    return out


def run_go(cases, repo_root):
    """Render every case through the Go package and return its results.

    The program is written inside the module, because internal/chat is an internal package:
    a file outside the module tree may not import it.
    """
    # The parent has to exist: mkdtemp creates one level and raises FileNotFoundError if
    # .cache/ is missing. It is ignored scratch space, so `just clean` removes it -- which
    # would otherwise leave this script unable to run until someone recreated the directory.
    cache_dir = os.path.join(repo_root, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="jinja-diff-", dir=cache_dir)
    try:
        prog = os.path.join(tmp, "main.go")
        with open(prog, "w") as f:
            f.write(GO_PROGRAM)
        payload = [
            {
                "name": c["name"], "template": c["template"], "messages": c["messages"],
                # The Go side takes a tool's parameters as text and prints them verbatim,
                # while Jinja2 re-serialises the object it is handed. Both are right for their
                # own caller: a server passes through the schema the client sent, and Jinja2
                # has only the parsed value. The conversion happens here, at the boundary,
                # rather than in either implementation.
                "tools": dumped_tools(c.get("tools", [])),
                "add_generation_prompt": c["add_generation_prompt"],
                "bos_token": c["bos_token"], "eos_token": c["eos_token"],
            }
            for c in cases
        ]
        proc = subprocess.run(
            ["go", "run", prog],
            input=json.dumps(payload), capture_output=True, text=True, cwd=repo_root,
        )
        if proc.returncode != 0:
            sys.stderr.write("go run failed:\n" + proc.stderr + "\n")
            sys.exit(2)
        return json.loads(proc.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cases_from_checkpoint(path):
    """Build cases from the template a checkpoint actually ships.

    The corpus below is a set of hand-written approximations: `llama3` here is one line, while
    the Llama 3.2 file carries 3827 characters that branch on tools, a date and a system
    message. Comparing against an approximation answers "does this subset handle the shape I
    typed"; it cannot answer "does it handle the template a model ships", which is what the
    documentation claims.

    The template is read out of the GGUF metadata through the `gguf` package, so this side does
    not share the Go loader either.
    """
    import gguf

    if not os.path.exists(path):
        sys.stderr.write(f"{path}: no such file\n")
        return []
    try:
        reader = gguf.GGUFReader(path)
    except Exception as e:  # noqa: BLE001 - the package raises bare ValueError/struct.error
        # A traceback here reads as a bug in this script. The caller passed the wrong path, and
        # "not a GGUF file" is the fact they can act on.
        sys.stderr.write(f"{path}: not a readable GGUF file ({e})\n")
        return []
    template = None
    for field in reader.fields.values():
        if field.name != "tokenizer.chat_template":
            continue
        # GGUF metadata strings are stored as arrays of single bytes.
        template = bytes(field.parts[field.data[0]]).decode("utf-8")
        break
    if template is None:
        # Not an error in this package: a base model or a fine-tune may ship without one. It
        # is still a failed run, because zero cases compared is not a pass -- the same
        # distinction the stress script draws.
        sys.stderr.write(f"{path}: declares no tokenizer.chat_template, so there is nothing "
                         f"to compare; pick a checkpoint whose file carries one\n")
        return []

    print(f"template from {path}: {len(template)} characters")
    cases = []
    for msgname, msgs in MESSAGES.items():
        for add_gen in (False, True):
            cases.append({
                "name": f"{os.path.basename(path)}/{msgname}/gen={add_gen}",
                "template": template, "messages": msgs,
                "add_generation_prompt": add_gen,
                # A checkpoint's own tokens: the template interpolates bos_token, and an
                # empty one would compare nulls on both sides.
                "bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>",
                "kind": "template",
            })
    return with_oracle(cases)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="list every case and its verdict")
    ap.add_argument("--repo", default=".", help="repository root (default: .)")
    ap.add_argument("--template-file", metavar="MODEL.gguf",
                    help="compare the chat template this checkpoint ships instead of the "
                         "built-in corpus; needs the gguf package")
    args = ap.parse_args()

    if not shutil.which("go"):
        sys.stderr.write("go not found on PATH\n")
        return 2

    cases = (cases_from_checkpoint(args.template_file) if args.template_file
             else build_cases(args.repo))
    if not cases:
        return 1
    no_oracle = [c for c in cases if not c["oracle_ok"]]
    cases = [c for c in cases if c["oracle_ok"]]
    # Zero comparable cases is a failed run rather than a vacuous pass. Jinja2 is the reference
    # here, so a case it cannot render has nothing to compare against -- and if that is all of
    # them, the script would print "0 cases compared, 0 agree" and exit 0 having checked
    # nothing. Measured by making every oracle fail: that is exactly what happened.
    if not cases:
        print("0 cases could be compared: Jinja2 failed to render every one", file=sys.stderr)
        return 1
    results = run_go(cases, args.repo)

    agree = 0
    wrong = []
    refused_ok = []
    refused_bad = []
    diverged = []
    diverged_bad = []
    for c, r in zip(cases, results):
        if c["kind"] == "divergent":
            # The documented output, and that it is still not what Jinja2 says. Both halves:
            # agreeing with the reference here would mean the documented behaviour changed.
            if r["error"]:
                diverged_bad.append((c["name"], r["error"]))
            elif r["text"] != c["want"]:
                diverged_bad.append((c["name"], f"got {r['text']!r}, documented {c['want']!r}"))
            elif r["text"] == c["oracle"]:
                diverged_bad.append((c["name"], f"now agrees with Jinja2 ({r['text']!r}); the "
                                               "documented divergence is gone"))
            else:
                diverged.append(c["name"])
            continue
        if c["kind"] == "asset" and c["name"] in ASSET_DIVERGENT:
            # Compared for non-equality as well: a change that made the two agree would mean
            # the verbatim-parameters decision had been reversed, which is worth being told.
            if r["error"]:
                diverged_bad.append((c["name"], r["error"]))
            elif r["text"] == c["oracle"]:
                diverged_bad.append((c["name"], "now agrees with Jinja2; the documented "+
                                     "difference is gone"))
            else:
                diverged.append(c["name"])
            continue
        if c["kind"] == "unsupported":
            if r["error"]:
                refused_ok.append(c["name"])
            else:
                refused_bad.append((c["name"], r["text"]))
            continue
        if r["error"]:
            wrong.append((c["name"], f"error: {r['error']}", c["oracle"]))
            continue
        if r["text"] == c["oracle"]:
            agree += 1
        else:
            wrong.append((c["name"], r["text"], c["oracle"]))

    print(f"{len(cases)} cases compared, {agree} agree with Jinja2")
    if diverged:
        print(f"{len(diverged)} diverge by design, each producing its documented output")
    if refused_ok:
        print(f"{len(refused_ok)} unsupported constructs refused, as designed")
    if no_oracle:
        # Named rather than counted. A skip is a case that ran neither side -- the Go render never
        # happens and nothing is compared -- so a number without the names reads as coverage when
        # it is the opposite. Templates that are refused on purpose land in refused_ok instead;
        # this list is only for cases where the oracle itself could not render.
        print(f"{len(no_oracle)} cases skipped because Jinja2 itself failed:")
        for c in no_oracle:
            print(f"  {c['name']}  ({c['oracle_error']})")

    failed = False
    if wrong:
        failed = True
        print(f"\n{len(wrong)} DISAGREE:")
        for name, got, want in wrong:
            print(f"  {name}\n    go     {got!r}\n    jinja2 {want!r}")
    if refused_bad:
        failed = True
        print(f"\n{len(refused_bad)} unsupported constructs were RENDERED instead of refused:")
        for name, got in refused_bad:
            print(f"  {name}\n    go {got!r}")
    if diverged_bad:
        failed = True
        print(f"\n{len(diverged_bad)} divergent cases did not behave as documented:")
        for name, why in diverged_bad:
            print(f"  {name}\n    {why}")

    if args.verbose:
        print("\nagreements:")
        for c, r in zip(cases, results):
            if c["kind"] != "unsupported" and not r["error"] and r["text"] == c["oracle"]:
                print(f"  {c['name']}")
        # The summary line says nine constructs were refused as designed, and without this
        # list a reader can only take that number on trust: a template that was refused for
        # the wrong reason -- a typo in the case, an unrelated parse error -- would be
        # counted in the same nine. Naming them makes "as designed" checkable.
        print("\nrefused (unsupported constructs, expected):")
        for c, r in zip(cases, results):
            if c["kind"] == "unsupported" and r["error"]:
                print(f"  {c['name']}")
                print(f"    {r['error'].splitlines()[0]}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
