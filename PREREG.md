# PREREG — spade-zero-gradient v0.1

Registered 2026-08-25 (Cowork session, Roland + Claude). Frozen at S1 unless a
dated AMENDMENT entry is added below. All outcomes are reportable; a null or
negative result answers the research question and will be written up as such.

## 1. Research question

SPADE (Liu, Yu et al., arXiv:2608.19197) trains its Environment Designer with
RL — weight updates driven by hint-based regret. Its Future Directions state:

> "Future work could explore Environment Designers that improve through
> in-context learning rather than gradient updates."

**Question: can the Environment Designer improve — produce measurably more
teachable environments for a fixed solver — with zero weight updates, using
only (a) accumulated in-context environment memory and (b) evolution of its
written design strategy?**

## 2. Setting (Phase A — primary)

- **Designer:** Claude Sonnet, frozen weights (D-05, 2026-08-24). Everything it
  can "learn" arrives in-context: strategy text, memory digest, contract.
- **Solver:** Claude Haiku, frozen weights, identical prompt scaffold in all
  conditions and rounds (D-05). Freezing the solver removes SPADE's
  co-evolution confound: any fitness trend is attributable to the designer side.
- **Domain (D-01, accepted 2026-08-25):** SPADE's games setting in miniature,
  three cognitive skills in rotation: logical_deduction, pattern_recognition,
  optimization. Environments are single-class Python programs under the frozen
  contract in `szg/env_api.py` (Gym-style reset/step, seeded determinism,
  ≤ 10 steps, rewards in [0,1], allowlisted imports).
- **Grounding:** fixed topic pack `seeds/topics.jsonl` (60 topics, 20/skill),
  sampled with seeded RNG — a fixed-size stand-in for SPADE's pretraining-corpus
  grounding, identical across conditions.

## 3. Conditions

| | strategy | environment memory |
|---|---|---|
| **C0** static designer (SPADE's no-training/no-memory control) | S0 frozen | off |
| **C1** in-context learning only | S0 frozen | accumulating |
| **C2** full gradient-free designer | evolved (ShinkaEvolve-style over strategy texts) | accumulating |

S0 = `strategies/S0_baseline.md`, distilled from the SPADE paper's designer
guidance; C2's archive is seeded with exactly S0. Same number of designer
calls, envs per round, and solver episodes per condition.

## 4. Fitness (D-02, accepted 2026-08-25)

Floored hint-based regret is the ONLY selective fitness:

    r_D(e) = max(0, mean_return(solver | hint) − mean_return(solver))
    strategy_fitness(round) = mean r_D over VALID environments

Paired arms: same G seeds with and without hint. Validity is gated, not
blended into reward (gates ≠ fitness):

- **V1** executability battery (load, seeded random-walk, per-step timeout,
  determinism under repeated seed).
- **V2** hint legality: ≤ 600 chars, no `ACTION:` string, no fenced code.
- **V3** eligibility band: no-hint win rate ∈ [0.05, 0.95]. This excludes the
  password-environment degenerate optimum (unsolvable without hint) and
  trivial environments, without adding any reward term.
- A round with < 2 valid environments scores fitness 0 (INVALID_ROUND).

Descriptive metrics (logged and reported, never selected on): learnable
fraction (no-hint win ∈ [0.2, 0.8], SPADE's band), valid fraction, mean win
rates, diversity proxy (distinct token trigram ratio), env length and
partial-reward counts.

## 5. Endpoints

- **Primary:** per-round strategy fitness trajectories for C0/C1/C2 (same
  round grid). Test statistic: difference in mean fitness over the final third
  of rounds, C2−C0 and C1−C0, with bootstrap 95% CIs over environments
  (10,000 resamples). Slope over rounds reported as supporting evidence.
- **Secondary:** best-vs-S0 held-out comparison — the highest-mean-fitness C2
  strategy and S0 each evaluated on a fresh topic/seed grid never used in
  evolution.

## 6. Registered outcomes (all reportable)

- **DESIGN_IMPROVES** — C2 final-third fitness exceeds C0's with CI excluding 0,
  and the held-out comparison confirms the best evolved strategy beats S0.
- **MEMORY_ALONE_SUFFICES** — C1 beats C0 (CI excludes 0) and C2−C1 CI includes 0.
- **NO_TRACTION** — neither C1−C0 nor C2−C0 excludes 0.
- **SATURATION** — S0 already produces mean fitness ≥ 0.6 with learnable
  fraction ≥ 0.5 at probe time (R3 gate), leaving no headroom; run redesigned
  (difficulty dials in contract) before any main run, with a dated amendment.

## 7. Gates before compute (lessons of 2026-02→08 encoded)

- **R1 interface** (probe, real models): ≥ 50% of 10 S0 environments pass
  V1+V2+V3. Guards against interface starvation — the 2026-08-24 failure mode.
- **R2 spread:** max−min valid-env regret ≥ 0.15 across the probe. No spread →
  nothing to select on.
- **R3 headroom:** S0 must NOT already saturate (see SATURATION above).
- **R4 invalid-with-feedback:** machinery smoke — broken envs are rejected with
  reasons that reach the designer/mutator feedback path.
- **R5 determinism:** machinery smoke — identical seed ⇒ identical episode.

R4/R5 run on the mock backend (`make smoke`, no model calls). R1–R3 run once
on real models (`make gates`) and are a STOP: Roland reviews the probe report
before any main run.

## 8. Scale defaults (D-04 — proposed, to be SET BY ROLAND at the S1 STOP)

Proposed: G=6 paired episodes/arm, 6 envs/round, 8 rounds/condition, 2
replicate chains for C2. Per-unit cost is measured by the probe (wall-clock
and call counts are logged); Roland sets the final grid from those numbers.
No spend or scale commitments are made in this document beyond the probe.

## 9. Hard rules

- **H-RULE-1 Information parity:** solver prompts identical across arms except
  the hint block; designer and mutator prompts contain the full contract and
  the full measured feedback. No condition sees vocabulary another lacks.
- **H-RULE-2:** no changes to fitness, gates, or endpoints after S1 except by
  dated AMENDMENT here.
- **H-RULE-3:** every episode is logged (JSONL, full transcript); every
  designer/mutator call's model alias is recorded; `claude --version` and
  model IDs recorded per run.
- **H-RULE-4:** all four outcomes are publishable; the writeup commits to
  reporting whichever obtains.

## 10. Phase B (exploratory, not part of v0.1 claims)

Co-adaptation with an in-context-learning solver (solver notebook accumulating
across rounds). Pre-registered as exploratory only; built after Phase A data
exists. (D-06, 2026-08-24.)

## Decision log

- D-01 domain = 3-skill games — accepted 2026-08-25.
- D-02 fitness = floored regret + validity gates V1–V3 — accepted 2026-08-25.
- D-03 first move = scaffold + in-session pilot before handoff — accepted 2026-08-25.
- D-04 main-run scale — OPEN; Roland sets at S1 STOP from probe costs.
- D-05 casting: designer Sonnet / solver Haiku — accepted 2026-08-24.
- D-06 Phase A primary (frozen solver), Phase B exploratory — accepted 2026-08-24.

## Amendments

(none)
