# CLAUDE.md — instructions for Claude Code sessions in this repo

You are running the main experiments for **spade-zero-gradient** on Roland's
machine (Max subscription; model calls go through headless `claude -p` — the
`ClaudeCLIBackend` — never through a metered API key).

**Read PREREG.md first. It is frozen.** Fitness, gates, endpoints, and hard
rules H-RULE-1..4 are not yours to change; propose a dated AMENDMENT and STOP
for Roland if you believe a change is needed.

## Stages (one per session is fine; STOP means wait for Roland)

- **S1 — local verification.** `pip install -e . && pip install pytest`,
  `make test`, `make smoke` must be green. Then `make gates` (R1–R3 probe:
  10 S0 environments, Sonnet designer / Haiku solver, G=3). Report the probe
  JSON, wall-clock, and call counts. **STOP: Roland sets D-04 scale from
  these numbers.**
- **S2 — pilot chain.** One C2 chain, 3 rounds, the D-04 G: `szg round
  --condition C2 --round-index {0,1,2} --mutate --out runs/pilot_c2`.
  Sanity-check: fitness values plausible, memory digest growing, mutation
  texts diagnostic. **STOP: show Roland the first evolved strategies.**
- **S3 — main grid.** C0, C1, C2 at D-04 scale, same round grid and seeds
  per condition. C2 replicate chains use different `--rng-seed`. Commit
  `runs/` after every round (crash-safe; JSONL appends are resumable).
- **S4 — analysis + writeup.** Endpoints exactly as registered (final-third
  means, bootstrap CIs over environments, held-out best-vs-S0). Descriptives:
  learnable fraction, diversity, difficulty trends. Writeup states which
  registered outcome obtained.

## Hard rules (repo-specific, additive to PREREG)

- Never edit `szg/gates.py` thresholds, `szg/fitness.py`, or the env contract
  after S1 without a PREREG amendment.
- Never invent budget or scale commitments; Roland sets D-04 at the S1 STOP.
- Log everything: every round via the CLI (it writes envs.jsonl,
  episodes.jsonl with full transcripts, rounds.jsonl). Record
  `claude --version` in the run directory once per session.
- Generated env code executes under `szg/env_api.py`'s restricted namespace
  with timeouts; do not "helpfully" loosen the sandbox or the import allowlist.
- If a gate fails, that is a FINDING, not an obstacle: log it, diagnose in
  one paragraph, propose the smallest compliant fix, STOP for Roland.
  (Pair every failure diagnosis with a constructive build path.)
- All four registered outcomes are reportable. Do not steer runs toward
  DESIGN_IMPROVES; the null answers the question too.

## Context

Deliverable for the week (agreed 2026-08-24): repo + logged data + writeup.
Phase A only (frozen solver). Phase B (solver notebook co-adaptation) is
exploratory and starts only after Phase A data exists and Roland says go.
