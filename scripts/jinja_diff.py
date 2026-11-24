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
}

# Templates using constructs the package documents as unsupported. The expected Go result is
# a parse error, which is what makes "unsupported" a designed refusal rather than a wrong
# render. Jinja2 renders all of them.
UNSUPPORTED = {
    # Jinja's short form, which yields Undefined when the condition is false. Jinja2 renders it
    # as the empty string; here it is refused at parse time, which is the documented behaviour:
    # an undefined result that reached output would put a hole in the prompt, and no published
    # template relies on it. The refusal names the construct rather than rendering something
    # inert, so a reader learns which form to write instead.
    "inline_if_no_else": """{{ 'a' if add_generation_prompt }}""",
    "filter_block": """{% filter upper %}x{% endfilter %}""",
    "namespace": """{% set ns = namespace(x=1) %}{{ ns.x }}""",
    "macro": """{% macro m() %}x{% endmacro %}{{ m() }}""",
    "for_else": """{% for m in messages %}{{ m }}{% else %}none{% endfor %}""",
    "tuple_unpack": """{% for k, v in tools %}{{ k }}{% endfor %}""",
    "selectattr": """{{ messages | selectattr('role', 'equalto', 'user') | list }}""",
    "dict_literal": """{{ {'a': 1} }}""",
    "print_a_message": """{{ messages[0] }}""",
    # Jinja2 iterates a string character by character; this subset does not implement that and
    # refuses rather than yielding no turns. The refusal is at render time, which the tool treats
    # the same as a parse-time one -- both mean "this construct is not supported".
    "for_over_string": """{% for c in 'abc' %}{{ c }}{% endfor %}""",
}


def oracle(template, messages, tools, add_gen, bos, eos):
    env = jinja2.Environment(keep_trailing_newline=True)
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


def build_cases():
    cases = []
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
    return cases


GO_PROGRAM = r'''
package main

import (
	"encoding/json"
	"os"

	"github.com/pplmx/orbix/internal/chat"
)

type tc struct {
	Name     string        `json:"name"`
	Template string        `json:"template"`
	Messages []chat.Message `json:"messages"`
	Tools    []chat.Tool    `json:"tools"`
	AddGen   bool          `json:"add_generation_prompt"`
	BOS      string        `json:"bos_token"`
	EOS      string        `json:"eos_token"`
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
		text, err := tmpl.Render(c.Messages, chat.Inputs{
			AddGenerationPrompt: c.AddGen,
			BOSToken:            c.BOS,
			EOSToken:            c.EOS,
			Tools:               c.Tools,
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
                "tools": c.get("tools", []), "add_generation_prompt": c["add_generation_prompt"],
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

    cases = cases_from_checkpoint(args.template_file) if args.template_file else build_cases()
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
    for c, r in zip(cases, results):
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
    if refused_ok:
        print(f"{len(refused_ok)} unsupported constructs refused, as designed")
    if no_oracle:
        print(f"{len(no_oracle)} cases skipped because Jinja2 itself failed")

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
