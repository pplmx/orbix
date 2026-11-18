"""Check the chat template subset against real Jinja2.

This is a cross-validation tool, not part of the engine. The Go package in internal/chat
evaluates a subset of Jinja2 chosen from what published chat templates use; its own tests
compare against strings typed by hand, which can share a blind spot with the code. Real
Jinja2 is the independent side, so this script renders the same templates with the same
inputs through both and reports the disagreements.

Run it with uv, which provides Jinja2 without touching the system Python:

    uv run --with jinja2 python3 scripts/jinja_diff.py            # built-in corpus
    uv run --with jinja2 python3 scripts/jinja_diff.py --verbose  # show agreements too

It builds a small Go program that renders each case through the package under test, runs
both sides, and diffs the text. Exit status is 1 if anything disagrees, so it can be used as
a gate by hand.

A disagreement is one of two things, and the output says which:

  - a bug in the subset -- the Go side produced something Jinja2 does not;
  - an unsupported construct -- the package documents a list of Jinja2 features it does
    not implement, and a template using one is expected to be refused at parse time rather
    than rendered. Those cases are reported separately, because "refused" is the designed
    behaviour and "rendered differently" is not.

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
    "llama3": """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}""",
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
}

# Templates using constructs the package documents as unsupported. The expected Go result is
# a parse error, which is what makes "unsupported" a designed refusal rather than a wrong
# render. Jinja2 renders all of them.
UNSUPPORTED = {
    "inline_if": """{{ 'a' if add_generation_prompt else 'b' }}""",
    "filter_block": """{% filter upper %}x{% endfilter %}""",
    "namespace": """{% set ns = namespace(x=1) %}{{ ns.x }}""",
    "macro": """{% macro m() %}x{% endmacro %}{{ m() }}""",
    "for_else": """{% for m in messages %}{{ m }}{% else %}none{% endfor %}""",
    "tuple_unpack": """{% for k, v in tools %}{{ k }}{% endfor %}""",
    "selectattr": """{{ messages | selectattr('role', 'equalto', 'user') | list }}""",
    "dict_literal": """{{ {'a': 1} }}""",
    "print_a_message": """{{ messages[0] }}""",
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
        for msgname in ("user", "multi"):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="also list cases that agree")
    ap.add_argument("--repo", default=".", help="repository root (default: .)")
    args = ap.parse_args()

    if not shutil.which("go"):
        sys.stderr.write("go not found on PATH\n")
        return 2

    cases = build_cases()
    no_oracle = [c for c in cases if not c["oracle_ok"]]
    cases = [c for c in cases if c["oracle_ok"]]
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
        print(f"{len(refused_ok)} unsupported constructs refused at parse time, as designed")
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

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
