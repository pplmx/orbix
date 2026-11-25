#!/usr/bin/env bash
# Drive a race-enabled orbix server with the load mix a deployment sees: plain requests, SSE
# streams, and clients that hang up mid-generation. Invoked by `just stress`; see that recipe
# for why this exists as a script rather than a Go test.
#
# Usage: scripts/stress.sh <race-binary> <checkpoint>
#
# Exits non-zero if the race detector reported anything, if the server returned a 5xx, if
# orbix_requests_active did not return to zero, or if no request was ever cancelled or
# abandoned. The gauge is the leak check: a stale sequence holds its KV blocks, and the
# number is how that shows up without reading heap dumps. The last condition is the
# self-check -- a run that cancelled nothing exercised no disconnect, so the other three
# passed having tested little, and saying "ok" there is worse than failing.
set -uo pipefail

bin=${1:?usage: stress.sh <race-binary> <checkpoint>}
model=${2:?usage: stress.sh <race-binary> <checkpoint>}
addr=127.0.0.1:18599
work=$(mktemp -d)
trap 'kill "${srv:-}" 2>/dev/null; rm -rf "$work"' EXIT

"$bin" serve --model "$model" --addr "$addr" \
    --max-active 1 --blocks 64 --block-size 16 --prefix-cache --log-format json \
    >"$work/log" 2>&1 &
srv=$!

# The wait is generous because the binary is a race build and the checkpoint is read from disk:
# measured on Gemma 3 270M, the first start took 45.8s while the next two took 2.9s and 2.5s, so
# page cache rather than the race detector is what the first run pays for. The 20s a previous
# version allowed was enough on a fixture of a few KB and not on anything real, and the failure
# read "server did not become ready" on a server that was about to be ready.
#
# Waited for on the terminal state rather than a guessed number of seconds: the process either
# answers or exits, so a dead server fails immediately instead of after the full wait.
for _ in $(seq 1 1200); do
    curl -sf "http://$addr/healthz" >/dev/null 2>&1 && break
    kill -0 "$srv" 2>/dev/null || break
    sleep 0.1
done
if ! curl -sf "http://$addr/healthz" >/dev/null 2>&1; then
    echo "stress: server did not become ready; log follows" >&2
    tail -20 "$work/log" >&2
    exit 1
fi

api="http://$addr/v1/completions"
# The model name has to come from the server. It is derived from the checkpoint's filename, so
# a hardcoded name is a 404 that never reaches the engine -- which is what made an earlier
# version of this script report "no request was cancelled" while nothing was ever submitted.
# Split on the id key itself. The response is {"data":[{"created":0,"id":"tiny",...}]}, so a
# plain field split on quotes picks up "created" -- which is what an earlier version did, and
# why every request came back 404 with nothing ever reaching the engine.
model_name=$(curl -s --max-time 5 "http://$addr/v1/models" |
    sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)
if [ -z "$model_name" ]; then
    echo "stress: the server lists no model; cannot build requests" >&2
    exit 1
fi
echo "stress: model $model_name"

# payload builds a request body with the served model name substituted.
payload() { printf '%s' "${1/__MODEL__/$model_name}"; }
# The ceiling is 120s rather than the 30s these two used to carry, for the reason the holder's own
# ceiling gives below: a race build on a real checkpoint is slow, and a harness that gives up first
# reports an engine defect that is not there. Measured on the converted Qwen2.5-0.5B-Instruct, a
# 16-token request against the race binary took 15.4s, 15.5s, 17.4s and 15.7s on an idle engine --
# already half of 30s with no load -- and under the load this script itself creates it went past
# it. The failure was three 503s whose durations were exactly 30000ms, which is the client giving
# up rather than the server answering: the request was still being generated when curl closed it.
#
# The value is a bound on a broken run, not a tuning knob. A run that has actually stopped
# advancing fails on the gauge below and on the holder's timeout; a run that is merely slow must
# not be reported as either.
post() { curl -s -o /dev/null --max-time 120 -X POST "$api" -H 'Content-Type: application/json' -d "$(payload "$1")"; }
stream() { curl -s -o /dev/null -N --max-time 120 -X POST "$api" -H 'Content-Type: application/json' -d "$(payload "$1")"; }

# Requests run one after another rather than in parallel subshells. A background job plus
# `wait` in a pipeline is unreliable here -- the earlier version of this script hung at the
# first `wait` and had to be killed -- and the load that matters for the leak check is the
# cancel path, which the next section drives three times in a row. The concurrency the engine
# sees still comes from the server's own scheduler admitting several sequences per round.
echo "stress: plain requests"
post '{"model":"__MODEL__","prompt":"alpha","max_tokens":16}'
post '{"model":"__MODEL__","prompt":"beta","max_tokens":16}'

echo "stress: streams"
stream '{"model":"__MODEL__","prompt":"gamma","max_tokens":16,"stream":true}'
stream '{"model":"__MODEL__","prompt":"delta","max_tokens":16,"stream":true}'

# A cancel that lands while the engine is mid-generation, made deterministic rather than
# raced. A timeout cannot do this against a small fixture: 200 tokens complete in about 8ms,
# so any ceiling either fires before the request is admitted or not at all -- the first version
# used 50ms and cancelled nothing, the second used 2ms and the server never saw the request.
#
# Instead the server runs with --max-active 1, so one sequence is advanced per round and
# everything else queues. A slow request is started first and left running; the requests after
# it are necessarily still queued when their client hangs up, and the engine therefore has a
# submitted sequence to cancel rather than a finished one.
# The budget has to fit the context or the holder is refused before it can hold anything: this
# fixture's window is 256 positions, so a 250-token budget plus a prompt of any length exceeds
# it and the request returns in ~1ms having generated nothing. 120 leaves room for the prompt
# and still keeps the engine busy for several rounds.
# The holder's ceiling is its own, not the shared 30s. The binary is a race build, and race
# instrumentation is what it costs: measured on the 15M fixture, this exact 120-token request
# takes 47.8s alone and takes longer while three clients are queued behind it -- so a 30s
# ceiling cancelled the very request that is supposed to stay alive, and the failure surfaced
# as the holder returning 503 "abandoned before it finished". That reads like an engine defect
# and is an artefact of the harness giving up first.
#
# Generous rather than tuned: the run either finishes or the server stops advancing, and
# waiting too long only makes a broken run slower.
echo "stress: one long request holds the engine"
curl -s -o /dev/null --max-time 300 -X POST "$api" -H 'Content-Type: application/json' \
    -d "$(payload '{"model":"__MODEL__","prompt":"holder","max_tokens":120}')" &
holder=$!
# Give the holder a moment to be admitted and to start advancing. Without this the queued
# requests below can arrive before it, and with --max-active 1 one of them becomes the active
# sequence instead.
sleep 0.2

echo "stress: queued clients that hang up"
for _ in 1 2 3; do
    curl -s -o /dev/null -N --max-time 0.02 -X POST "$api" -H 'Content-Type: application/json' \
        -d "$(payload '{"model":"__MODEL__","prompt":"zeta","max_tokens":120,"stream":true}')" || true
done

wait "$holder"

# The engine advances a sequence per round, so the gauge needs a few rounds to reach zero
# after the last disconnect. Poll rather than sleeping a fixed time.
active=1
for _ in $(seq 1 100); do
    active=$(curl -s --max-time 5 "http://$addr/metrics" | awk '/^orbix_requests_active /{print $2}')
    [ "$active" = "0" ] && break
    sleep 0.1
done

status=0
races=$(grep -c 'DATA RACE' "$work/log" || true)
if [ "$races" != "0" ]; then
    echo "stress: $races DATA RACE reports" >&2
    grep -A 20 'DATA RACE' "$work/log" | head -60 >&2
    status=1
fi

if grep -q '"status":5' "$work/log"; then
    echo "stress: the server answered a 5xx" >&2
    grep '"status":5' "$work/log" | head -10 >&2
    status=1
fi

if [ "$active" != "0" ]; then
    echo "stress: orbix_requests_active settled at $active, want 0 -- a sequence leaked" >&2
    status=1
fi

# The disconnect path has to have actually run, or the checks above tested nothing. A ceiling
# longer than the generation time lets every request finish on its own, the gauge returns to
# zero without a disconnect, and the script reports ok -- which is what the first version did
# with a 50ms ceiling against a request that completes in 8ms.
#
# Both reasons count. A client that hangs up while its sequence is being advanced finishes as
# "cancel"; one that hangs up while still queued finishes as "abandoned". This script runs
# with --max-active 1 and a holder in front, so its disconnects are necessarily the queued
# kind -- watching only for "cancel" made several rounds of debugging chase a metric that
# could never move in this configuration.
cancelled=$(curl -s --max-time 5 "http://$addr/metrics" |
    awk -F'[{}"]' '/finished_by_reason_total\{reason="(cancel|abandoned)"\}/{gsub(/ /,"",$5); s+=$5} END {print s+0}')
if [ "${cancelled:-0}" = "0" ]; then
    echo "stress: no request was cancelled or abandoned, so the disconnect path went untested;" >&2
    echo "        lower the --max-time on the cancel requests below the generation time" >&2
    status=1
fi

if [ "$status" = "0" ]; then
    echo "stress: ok -- no races, no 5xx, active returned to 0, $cancelled cancelled"
fi
exit "$status"
