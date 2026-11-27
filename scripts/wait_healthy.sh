#!/usr/bin/env bash
# Start an Orbix image against a checkpoint and wait for Docker's own health verdict.
#
# Usage: wait_healthy.sh <image-ref> <checkpoint-path-on-host> [platform]
#
# Extracted because two workflows need it and they must not drift: the CI job proves the
# Dockerfile builds and runs, and the release job proves the image it just pushed to GHCR runs.
# A copy of this loop in each file would be two places to get the timing wrong, and the timing
# is not a free choice -- see the bound below.
#
# The exit status is the answer: 0 only when the container reported "healthy".

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <image-ref> <checkpoint> [platform]" >&2
    exit 2
fi

image=$1
checkpoint=$2
platform=${3:-linux/amd64}

if [ ! -f "$checkpoint" ]; then
    echo "wait_healthy: no checkpoint at $checkpoint" >&2
    exit 2
fi

model=$(basename "$checkpoint")
dir=$(cd "$(dirname "$checkpoint")" && pwd)

# The format string is passed as a variable rather than inline: docker inspect -f takes a Go
# template, and interleaving its braces with the shell's own quoting is a reading hazard.
fmt='{{.State.Health.Status}}'

cid=$(docker run -d --platform "$platform" \
    -v "$dir:/models:ro" \
    "$image" serve --model "/models/$model" --addr :8080)
echo "wait_healthy: container $cid from $image"

cleanup() {
    docker rm -f "$cid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# 90 attempts of 2s is 180s, which covers the image's own numbers rather than a guess: the
# Dockerfile sets --start-period=120s and --interval=30s, so a container may legitimately stay
# in "starting" for two minutes and the first verdict does not exist before thirty seconds.
# Waiting less than the start-period would call a working container unhealthy.
for i in $(seq 1 90); do
    state=$(docker inspect -f "$fmt" "$cid" 2>/dev/null || echo none)
    echo "wait_healthy: attempt $i: $state"
    case "$state" in
    healthy)
        docker logs "$cid" 2>&1 | tail -3
        exit 0
        ;;
    unhealthy)
        echo "wait_healthy: the container reported unhealthy" >&2
        docker logs "$cid" 2>&1 | tail -40 >&2
        exit 1
        ;;
    esac
    sleep 2
done

echo "wait_healthy: no healthy verdict within 180s" >&2
docker logs "$cid" 2>&1 | tail -40 >&2
exit 1
