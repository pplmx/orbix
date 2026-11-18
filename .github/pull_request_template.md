# What this changes

<!-- One or two sentences. The commits carry the reasoning; this is the summary. -->

## Why

<!--
What was wrong, or what was missing. For a defect, the observed behaviour and what it should
have been -- a document that promised something else counts as evidence.
-->

## Checks

The repository's gates, in the order worth running them:

- [ ] `just check` — vet, test, lint, gofmt
- [ ] `just docs` — if any Markdown changed
- [ ] `just hooks` — if any non-Go file changed
- [ ] `just commits` — the commit messages follow Conventional Commits

Two things the gates do not check, and that are worth a line here when they apply:

- [ ] A new behaviour has a test, and the test was confirmed to fail before the fix
- [ ] A claim in the README or a `doc.go` that this change makes untrue is updated

## Notes for review

<!--
Anything a reviewer would otherwise have to reconstruct: a trade-off taken, an alternative
rejected, a part left undone on purpose.
-->
