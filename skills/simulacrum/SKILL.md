---
name: simulacrum
description: Stress-test an idea, plan, architecture frame, definition, criterion, draft, or done claim using Pact's packaged Jeremy Simulacrum. Use proactively before committing to categorical claims or when the user asks for Sim, Simulacrum, Jeremy-style pushback, or the strongest counterargument.
---

# Simulacrum

Use Pact's packaged Simulacrum rather than searching for a user-specific skill
or executable.

For a direct claim review:

```bash
pact review "$TARGET_REPO" --sim-only --claim "$CLAIM"
```

For multi-tool post-implementation review:

```bash
pact review "$TARGET_REPO" --claim "$DONE_CLAIM"
```

Read the persisted `simulacrum.md`. A successful process means Sim completed;
it does not mean Sim approved the claim. Adjudicate the critique, correct the
frame or work when warranted, and rerun the review.

Pact ships the Sim runtime and annotated corpus. Provider calls use the local
operator's `ANTHROPIC_API_KEY` or `PACT_REVIEW_ANTHROPIC_API_KEY` directly;
Pact does not proxy or subsidize them. The optional `PACT_SIMULACRUM_CMD`
variable is an explicit operator override only. Operators can override the
packaged Anthropic model IDs with `SIMULACRUM_CLASSIFIER_MODEL` and
`SIMULACRUM_SPECIALIST_MODEL`.
