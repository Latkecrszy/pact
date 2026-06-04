# Done Gate

Verify each applicable item with evidence. Mark an item N/A only with a
specific reason.

## Contract Integrity

- Pact contracts, constraints, tests, and implementation agree.
- Implementation-discovered conflicts were reconciled in Pact artifacts.
- No required component or acceptance criterion is missing.

## Code Quality

- No stubs, placeholders, accidental short-circuits, or unresolved TODOs.
- External I/O and failure paths have explicit error handling.
- Inputs are validated at boundaries; authorization and secret handling were
  reviewed where relevant.
- Logging, events, metrics, and alerts exist where the system boundary warrants
  them.

## Verification

- Focused tests pass.
- Full relevant suite, type checks, and lint pass.
- `pact validate` passes.
- `pact audit` has no unresolved spec-compliance findings where applicable.

## Adversarial Review

- Advocate has no unresolved critical or high findings.
- The core architecture and done assertion survived Simulacrum review.
- Review artifacts are persisted and findings were fixed in the work.

## Delivery

- User-facing or operator-facing documentation is updated where needed.
- Durable decisions, discoveries, and notable verification results are in
  Kindex.
- The final report names residual risk and anything genuinely deferred.
