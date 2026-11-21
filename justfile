# Orbix development entry points.
#
# CGO_ENABLED=0 is exported for every recipe rather than inherited from the environment,
# because a build that accidentally picks up CGo is the one failure mode this project cannot
# tolerate. Exporting it here means the recipes stay correct even if the caller exported
# CGO_ENABLED=1.
#
# `set shell` is bash with -u: an unset variable in a recipe is an error rather than an empty
# string. That is the behaviour the previous Makefile got from `set -e` plus explicit checks,
# and it is what keeps a typo from turning into a silent no-op.

set shell := ["bash", "-uc"]

BIN := "bin/orbix"
PKG := "./..."
VERSION := `git describe --tags --always --dirty 2>/dev/null || echo dev`
COMMIT := `git rev-parse --short HEAD 2>/dev/null || echo none`
DATE := `date -u +%Y-%m-%dT%H:%M:%SZ`
LDFLAGS := "-s -w -X main.version=" + VERSION + " -X main.commit=" + COMMIT + " -X main.date=" + DATE

export CGO_ENABLED := "0"

# Every platform the project claims to build for. The list is short on purpose: each entry is
# a platform somebody runs the binary on, and a platform with no user is a build that breaks
# without anyone noticing.
CROSS_TARGETS := "linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64"

# Per-target fuzz budget. A run that reports FAIL has written the input to the package's
# testdata/fuzz, and that file belongs in the commit that fixes it.
FUZZTIME := env("FUZZTIME", "30s")

# The commit range `just commits` checks. Defaults to a whole branch; a shallow CI checkout has
# no origin/main, so the workflow passes its own range explicitly.
RANGE := env("RANGE", "origin/main..HEAD")
MODEL := env("MODEL", "")

# The local gate. `just` with no arguments runs this.
default: check

# compile the orbix binary into bin/
build:
    @mkdir -p "$(dirname {{BIN}})"
    go build -trimpath -ldflags '{{LDFLAGS}}' -o {{BIN}} ./cmd/orbix
    @echo "built {{BIN}} ({{VERSION}})"

# run the unit test suite with CGo disabled
test:
    @# -count=1 disables the test result cache. A cached pass is a pass that did not run, and
    @# the gate exists to run things: a contributor who changes a test helper and sees green
    @# from a cached binary has learned nothing.
    go test -count=1 {{PKG}}

# run the unit test suite under the race detector
test-race:
    @# The race detector requires cgo, so this recipe overrides the CGo ban for itself only.
    @# That does not weaken the build: the ban exists so no shipping binary ever links C, and
    @# this recipe produces no binary -- it is a test instrument. Without the override the
    @# recipe could never have succeeded, because CGO_ENABLED=0 is exported to every recipe.
    CGO_ENABLED=1 go test -count=1 -race {{PKG}}

# check the suite for order dependencies and state leaks
isolate:
    @# Two properties `test` cannot show, because it runs each package once in source order.
    @#
    @# -shuffle=on runs each package's tests in a different order. A test that only passes
    @# because an earlier one left something behind -- a package-level table, a file, an
    @# environment variable -- fails here and names its seed, so the failure reproduces.
    @#
    @# -count=2 runs every test twice in one process. A test that is not idempotent, or that
    @# closes something the second run needs, fails on the second pass. The engine and the
    @# paged cache hold the state that matters, but the flag is cheap everywhere.
    @#
    @# Not part of `check`: a shuffled failure names a seed rather than a bug, and the gate
    @# is meant to answer yes or no. Run this after touching test helpers, package-level
    @# variables, or anything with a Close.
    go test -count=2 -shuffle=on {{PKG}}

# drive a race-enabled server with concurrent mixed load
stress:
    @# The race detector needs cgo, and the server it instruments is the thing under test, so
    @# this builds a race binary rather than reusing bin/orbix. Like `test-race`, it produces no
    @# shippable artifact.
    @#
    @# What this covers that `test-race` cannot: the unit tests only race the concurrency that
    @# appears in them. This exercises the paths a deployed server actually mixes -- plain
    @# requests, SSE streams, and clients that disconnect mid-stream -- against one engine, which
    @# is where KV block recycling and goroutine teardown collide.
    @#
    @# Needs a checkpoint: `just MODEL=/path/to.gguf stress`. There is no synthetic fallback,
    @# because the disconnect path this exists to exercise needs a request that takes long
    @# enough to interrupt, and the in-repo fixtures finish in single-digit milliseconds.
    @test -n "{{MODEL}}" || { echo "stress: needs MODEL=/path/to.gguf" >&2; exit 1; }
    @CGO_ENABLED=1 go build -race -o .cache/orbix-race ./cmd/orbix
    @scripts/stress.sh .cache/orbix-race "{{MODEL}}"

# cross-check a GGUF file against an independent parser, tensor by tensor
xcheck:
    @# The readers a repository can compare against itself are all its own: the parser, its
    @# fixtures, and the converter that wrote them. A wrong assumption in that set is invisible
    @# to every test built on it, and it is exactly the kind of assumption -- a dimension order,
    @# a block layout, where a tensor's bytes begin -- that produces plausible numbers rather
    @# than a crash. The `gguf` Python package is an implementation nobody here wrote, so it is
    @# the second opinion.
    @#
    @# Compares type, dimensions and SHA-256 of the stored bytes for every tensor, so a wrong
    @# offset fails even when the shape agrees. Measured on four files -- 33, 57, 201 and 290
    @# tensors -- all agreeing; shifting the reported offset by one byte fails every tensor that
    @# has a neighbour, which is the property this exists for.
    @#
    @# Needs uv for the Python side and ./bin/orbix for the Go side, for the same reason `docs`
    @# and `hooks` are separate recipes: a contributor with only the Go toolchain should not be
    @# blocked by a check that needs Python.
    @command -v uv >/dev/null 2>&1 || { \
        echo "uv not found: install it (https://docs.astral.sh/uv/) or skip this check"; \
        exit 1; \
    }
    @test -n "{{MODEL}}" || { echo "xcheck: needs MODEL=/path/to.gguf" >&2; exit 1; }
    @test -x {{BIN}} || { echo "xcheck: {{BIN}} not found; run \`just build\` first" >&2; exit 1; }
    uv run --with gguf python3 scripts/gguf_xcheck.py "{{MODEL}}" {{BIN}}

vet:
    go vet {{PKG}}

# format every Go file
fmt:
    gofmt -l -w .

# run golangci-lint
lint:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v golangci-lint >/dev/null 2>&1 || {
        echo "golangci-lint not found: install it (https://golangci-lint.run/docs/welcome/install/) or run CI's check job"
        exit 1
    }
    # .golangci.yml is a v2 configuration and golangci-lint v1 refuses to read one. Its own
    # message is about the config file rather than the toolchain, so a reader has no way to
    # learn that the fix is a different binary, and `just check` goes red on a clean tree --
    # the failure mode this gate exists to prevent. Measured with mise resolving v1.64.8
    # while CI pins v2.15.0: check failed with "you are using a configuration file for
    # golangci-lint v2 with golangci-lint v1".
    #
    # A shebang recipe rather than a plain one: the body is handed to bash verbatim, so `$`
    # needs no escaping. The first attempt wrote `$$` in a plain recipe and just turned it
    # into the PID, which made the version probe a file name.
    v="$(golangci-lint --version 2>/dev/null | sed -n 's/.*version v\{0,1\}\([0-9][0-9]*\).*/\1/p' | head -1)"
    if [ "${v:-}" != "2" ]; then
        echo "golangci-lint is not v2 (found ${v:-unknown}): .golangci.yml is a v2 config and v1 refuses to read it"
        echo "install golangci-lint v2 (CI pins v2.15.0): https://golangci-lint.run/docs/welcome/install/"
        exit 1
    fi
    golangci-lint run

# run kernel and engine benchmarks
bench:
    go test -run '^$' -bench . -benchmem {{PKG}}

# The targets live beside the code they check and `just test` runs their seed corpus on every
# run, which is what keeps a known crash from coming back. This recipe is for finding new ones:
# it runs each target for FUZZTIME, so a full pass costs a couple of minutes and is worth doing
# before a release rather than on every save.
#
# run every fuzz target for a short time
fuzz:
    #!/usr/bin/env bash
    set -euo pipefail
    for pkg in $(go list ./...); do
        dir=$(go list -f '{{'{{'}}.Dir{{'}}'}}' "$pkg")
        targets=$(grep -ho 'func Fuzz[A-Za-z0-9_]*' "$dir"/*_test.go 2>/dev/null \
            | sed 's/func //' | sort -u || true)
        for t in $targets; do
            echo "==> $pkg $t ({{FUZZTIME}})"
            # The `$` is written `\$`: a shebang recipe's body is handed to bash as written,
            # so `$$` there is bash's PID, not just's escape for a literal dollar. That made
            # the pattern `^FuzzParse1234`, which matches no target.
            out=$(go test "$pkg" -run '^$' -fuzz "^$t\$" -fuzztime {{FUZZTIME}} 2>&1) || {
                printf '%s\n' "$out"
                exit 1
            }
            # Go answers a -fuzz regexp matching nothing with this warning and exit 0, so
            # without the check the recipe exits "ok" having fuzzed nothing -- which is how
            # the `$$` bug above survived: every run reported success.
            if printf '%s' "$out" | grep -q 'no fuzz tests to fuzz'; then
                echo "$t matched no fuzz target in $pkg" >&2
                exit 1
            fi
            printf '%s\n' "$out" | tail -2
        done
    done
    echo "fuzz: ok"

# the full local gate - format, vet, test, lint
check: vet test lint
    #!/usr/bin/env bash
    set -euo pipefail
    unformatted=$(gofmt -l .)
    if [ -n "$unformatted" ]; then
        echo "gofmt needed on:"
        echo "$unformatted"
        # Name the recipe that fixes it. check reports rather than rewrites -- a gate that
        # edited the tree would make "did I change this?" unanswerable -- so the fix has to be
        # somewhere, and it is `just fmt`.
        #
        # Single quotes, not double: this body is handed to bash as written, so backticks in a
        # double-quoted string are command substitution. The first draft did that and printed
        # "run internal/engine/zz_fmt.go to fix" -- it ran the filename.
        echo 'run `just fmt` to fix'
        exit 1
    fi
    echo "check: ok"

# lint Markdown with rumdl
docs:
    @# CI runs this as its own job, and the repository's convention is that Markdown is code:
    @# the README is the entry page, everything under docs/ is the body, and each package's
    @# doc.go is its authoritative description, so a malformed table or a broken link is a
    @# defect like any other. It walks the tree, so a page under docs/ is linted whether or
    @# not the README links it; what it does not see is the package doc.go files, which are Go.
    @# It is a separate recipe rather than part of `check` because rumdl is a Python-era tool
    @# that a Go-only contributor may not have, and `check` should stay runnable with the Go
    @# toolchain and golangci-lint alone.
    @command -v rumdl >/dev/null 2>&1 || { \
        echo "rumdl not found: install it (https://rumdl.dev) or run CI's docs job"; \
        exit 1; \
    }
    rumdl check .

# run the pre-commit hooks over every tracked file
hooks:
    @# The hooks check what `just check` cannot see -- trailing whitespace and final newlines in
    @# non-Go files, YAML/TOML/JSON syntax, merge-conflict markers. Measured: a whitespace-only
    @# line appended to a workflow file left `check` green.
    @#
    @# A separate recipe because prek is a Rust binary that a Go-only contributor may not have,
    @# for the same reason `docs` is separate from `check`. CI runs it as its own job with
    @# --all-files, which is the same command.
    @command -v prek >/dev/null 2>&1 || { \
        echo "prek not found: install it (https://prek.j178.dev) or run CI's prek job"; \
        exit 1; \
    }
    prek run --all-files

# check commit messages against the Conventional Commits format
commits:
    @# The commit-msg hook does this for each commit as it is made, but a hook only runs where
    @# `prek install` ran. CI needs the same check over a range rather than one message, and
    @# `cz check --rev-range` is that command. prek's file-oriented run cannot substitute:
    @# commit-msg hooks need a message file and are skipped by `prek run --all-files`.
    @command -v cz >/dev/null 2>&1 || { \
        echo "cz not found: install commitizen (https://commitizen-tools.github.io/commitizen/)"; \
        exit 1; \
    }
    cz check --rev-range {{RANGE}}

# build every supported platform with CGo disabled
cross:
    #!/usr/bin/env bash
    set -euo pipefail
    for target in {{CROSS_TARGETS}}; do
        os=${target%/*}
        arch=${target#*/}
        ext=""
        if [ "$os" = "windows" ]; then ext=".exe"; fi
        out="bin/orbix-$os-$arch$ext"
        echo "building $out"
        GOOS=$os GOARCH=$arch go build -trimpath -ldflags '{{LDFLAGS}}' -o "$out" ./cmd/orbix
    done

# Prints the version cz would bump to and the changelog section it would write, then stops. The
# real bump is a person's decision: it writes a tag, and a tag is what `git describe` turns into
# the version a binary reports.
#
# show what the next release would be, without changing anything
release-preview:
    @command -v cz >/dev/null 2>&1 || { \
        echo "cz not found: install commitizen (https://commitizen-tools.github.io/commitizen/)"; \
        exit 1; \
    }
    cz bump --dry-run --yes

# `major_version_zero` is on in .cz.toml, so a breaking change before 1.0 bumps the minor rather
# than the major -- the project is pre-1.0 and says so in the README.
#
# bump the version, write CHANGELOG.md, and tag
release:
    @command -v cz >/dev/null 2>&1 || { \
        echo "cz not found: install commitizen (https://commitizen-tools.github.io/commitizen/)"; \
        exit 1; \
    }
    cz bump --yes

# The list mirrors .gitignore's artifact entries rather than naming only bin/: a `clean` that
# leaves 7 MiB of rumdl cache and stray test binaries is a clean that surprises the next person
# to look at the working tree.
#
# `coverage.*` is listed separately from `*.out` because it is a different name, not a different
# extension: `go test -coverprofile=coverage.out` is caught by the `*.out` pattern, but the same
# run redirected to coverage.html or coverage.txt -- which is what producing a browsable report
# looks like -- was left behind while .gitignore called it ignored.
#
# remove everything the build and the tools leave behind
clean:
    @# The compiler's own cache is shared across projects and lives outside the tree; removing
    @# it here would slow every other Go checkout on the machine.
    @rm -rf bin .rumdl_cache .cache
    @# `go build ./...` writes ./orbix at the module root; .gitignore lists it for the same
    @# reason and this list mirrors that one.
    @rm -f orbix
    @find . -name '*.test' -type f -delete
    @find . -name '*.out' -type f -delete
    @find . -name '*.prof' -type f -delete
    @find . -name 'coverage.*' -type f -delete
    @echo "clean: removed bin/, caches, and test/profiling artifacts"
