---
name: pact-engineer
description: Use Pact as a plan-first contract and test engine while the active Claude or Codex agent implements, reconciles infeasible contracts, validates the result, and runs Advocate plus Simulacrum review. Use for substantial feature work, architecture changes, risky fixes, or any task where the user asks to plan with Pact, run Pact plan-only, follow the engineer workflow, or obtain adversarial review before declaring done.
---

# Pact Engineer

Use Pact to make the implementation bet explicit before coding. Keep Pact's
contracts and tests authoritative, but treat planning and implementation as a
feedback loop rather than a one-way handoff.

## Workflow

1. Source `~/.profile` when feasible.
2. Start or resume a Kindex session, then search the task and affected domain.
3. Read repository instructions and the relevant implementation, tests, and
   recent history before framing the change.
4. Normalize vague task language into a concrete problem statement,
   boundaries, constraints, and acceptance criteria. Use Constrain artifacts
   when available; otherwise create equivalent `prompt.md` and
   `constraints.yaml` inputs.
5. Run central architecture claims past Simulacrum before locking the frame.
6. Run Pact in plan-only mode and advance it through interview/decomposition
   until it pauses after contracts and tests:

```bash
pact init "$PACT_DIR" --budget 10
pact run "$PACT_DIR" --constrain-dir "$CONSTRAIN_DIR" --plan-only --once
```

Answer or approve interview questions as needed, then continue `pact run
--once` until the plan-only pause. Do not bypass unresolved ambiguity.

7. Read `design.md`, `.pact/decomposition/tree.json`, contracts, visible tests,
   and generated task lists. Run `pact assess` on the affected source tree.
8. Render a focused brief before implementing each component:

```bash
pact handoff "$PACT_DIR" "$COMPONENT_ID" --validate --max-tokens 2000
```

9. Implement directly with the active agent. Follow the repository's existing
   patterns and verify incrementally.
10. Reconcile contracts whenever implementation exposes infeasibility:
    - Stop instead of silently deviating.
    - Record the conflict and the failed assumption in Kindex.
    - Update task/constraints/contracts and regenerate affected tests or rerun
      Pact plan-only.
    - Resume only after the contract and implementation agree.
11. Run repository tests, type checks, lint, `pact validate`, and `pact audit`
    where applicable.
12. Run post-implementation review:

```bash
pact review "$TARGET_REPO" \
  --claim "This change is done because: <specific evidence>"
```

Treat Advocate critical/high findings and Simulacrum premise rejections as
blocking. Fix the work, not merely the wording, then rerun review.
13. Check [references/done-gate.md](references/done-gate.md), capture durable
    decisions and results in Kindex, and end the Kindex session.

## Execution Modes

- Default to Pact plan-only plus active-agent implementation.
- Use `pact run <project> --implement` only when the user explicitly wants Pact
  to own implementation and integration.
- Use `pact build <project> <component>` for targeted Pact-managed
  implementation after plan-only decomposition.
- After three genuine failed approaches or material scope expansion, stop,
  document the invalid assumption, and choose a simpler contract-satisfying
  approach or report the blocker.

## Review Tool Behavior

`pact review` persists `advocate.json`, `simulacrum.md`, command output, and a
machine-readable `review.json` under `.pact/reviews/`. Pact uses its packaged
Simulacrum runtime and annotated corpus by default; `PACT_SIMULACRUM_CMD` is an
explicit operator override. A successful command means requested tools
completed and Advocate returned no critical/high findings; the active agent
must still read and adjudicate Simulacrum's response.
