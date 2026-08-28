# External mid-grid review — verification, corrections, new findings

Recorded 2026-08-28 by a separate Claude Code session at Roland's request
("check how the work is advancing, create suggestions for solving problems").
Read-only review of commit `0604b6d` (snapshot 2026-08-27 ~18:54 UTC,
mid-round-4). Nothing in the repo was modified except adding this file.
Method: independent recomputation of every headline number from the JSONL
logs, plus a line-level audit of `szg/` against PREREG.md. Numbers below were
recomputed at least once from raw data; the load-bearing ones (selection
replay, topic pairing, boundary env, digest mislabels) were verified twice
independently.

## A. What checks out (verified clean)

- **README status block: zero discrepancies.** Trajectory table, means/sds
  (C0 0.1069/0.0857, C1 0.0898/0.0352, C2 0.0377/0.0478), 12/18 rounds,
  79/108 envs, outcome census (22/25/21 evaluated; yields 88/93/78%), raw
  regret split 24 helped / 28 harmed / 16 none, and both probe tables all
  reproduce exactly from the logs.
- **Data integrity after two crash recoveries: clean.** No duplicate env_ids
  in any envs.jsonl; no duplicate (env_id, arm, seed) episodes; exactly 6
  episodes per evaluated env (132/150/126); memory.jsonl matches envs.jsonl
  in both directions; the pruned round-2 and round-4 partials left no orphans.
- **H-RULE-1 solver-side parity holds in all 408 grid episodes**: one
  identical base scaffold across conditions, hint block present/absent
  exactly per arm. `claude --version` (2.1.246) recorded per run dir; the
  binding probe (probe_s0_a2) ran on the same version as the grid.
- **A1/A2 discipline**: the original probe is correctly reported as PASS
  under neither band; only the fresh re-probe gates the grid, as registered.

## B. Corrections to already-logged findings

### B1. The C2 timeout confound is real, but the mechanism in
`C2_designer_timeout_confound.md` is wrong — and its proposed digest fix
would not work

The doc attributes the timeouts to C2 "carrying more context". Measured:

- The digest is **bounded by construction** (≤ 4 high + 2 easy + 2 hard +
  6 recent lines + headers): observed 55 → 3,974 → 4,818 → 5,553 → 5,438 →
  5,355 chars at rounds 0..5 — a ~5.4 KB plateau that *shrank* between the
  two timeout rounds.
- Full reconstructed designer prompts: C0 ≈ 3.9 KB, C1 ≈ 8.9–9.5 KB,
  C2 ≈ 9.6–9.8 KB. C2 exceeds C1 by ~266 chars — exactly the S3-vs-S0
  strategy-text delta, ~3–4%. C1 at essentially the same prompt size has
  **zero** timeouts in 27 envs.
- All 4 timeouts are full 5-attempt exhaustions (inter-record gaps
  1275–1276 s = 5×240 s + backoff): the generation deterministically exceeds
  240 s every attempt — not transient load, not input length.
- They land **only** on rounds running evolved strategy S3, and only on the
  logical_deduction / pattern_recognition slots (e00/e01 in BOTH r3 and r4);
  every optimization-slot call under S3 succeeded. S3's text demands heavy
  pre-output verification ("compute residual uncertainty at the moment of
  final commit…"), which plausibly drives >240 s generations.

**Implication:** the confound is *evolved-strategy-induced generation time*.
Digest truncation targets a term that is already bounded and near-identical
across C1/C2. The remedy that matches the mechanism is a larger `timeout_s`
(infrastructure, as the doc argues) — for a re-run/replicate, not mid-grid.
The direction of the confound (against C2, inside the final third) stands.

Related gap: `ClaudeCLIBackend` increments stats only after `subprocess.run`
returns, so **timed-out attempts appear in no counter** — C2 r3 logs 34
backend_calls while ~20 additional 240 s timeout attempts are invisible
(H-RULE-3 is weakened at the stats level; the env error strings are the only
trace). Also `rounds.jsonl.backend_calls` is written before the mutate call
and so undercounts by 1 vs the stdout JSON.

### B2. "Fitness is nearly single-environment" (P4) — quantified

Top valid env carries 39–100% of round regret (median ≈ 64%); exactly
single-env in 1 of 13 scored cells, ≥64% in 5, but 3–4 envs contribute in
c1 r1/r3 and c2 r1. Deeper power fact: only 21 of 47 valid envs (45%) carry
*any* positive regret; pooled per-env sd 0.123 ⇒ SE ≈ 0.036 for a 12-env
final-third mean — same order as every observed between-condition difference
(0.017–0.069). P1 confirmed independently at env level.

## C. New findings (not in any logged finding)

### C1. CRITICAL — strategy selection is deterministic under the seed
policy, and round 5 is already pinned to S0

`rng = Random(2000 + round_index)` and `select_parent`'s epsilon draw is the
round's first `rng.random()`. Replaying reproduces every realized pick
exactly: draws 0.4485 / 0.5864 / 0.7275 / **0.1128 (ε-slot → S3)** /
0.5245 (exploit → S3). For round 5: `Random(2005).random() = 0.4504 ≥ 0.25`
forces the exploit branch (untried strategies unreachable there), and the
second draw (0.0764) picks S0 unless S3's mean exceeds ~0.407, i.e. unless
round-4 fitness > 0.76 — impossible (grid max is 0.2083).

So the **registered final third of C2 is already fixed**: one
timeout-crippled S3 round (r4) + one S0 round (r5, behaviorally ≡ C1), and
S1/S2/S4/S5 will never be evaluated in this chain — 4 of 6 evolution
outputs dead on arrival. Sharper than the logged P2 (which treated selection
as stochastic): the ε "exploration" happened exactly once and the whole
schedule was fixed when seeds 2000–2005 were chosen. Note the interaction:
the *only* reason the final third contains an evolved round at all is the
contaminated r3 score lifting S3 into the exploit branch.

### C2. MAJOR — the registered topic pairing is silently broken for C2

PREREG §2 registers grounding "identical across conditions"; the run dirs
record "seed policy … IDENTICAL across conditions (paired design)". Verified:
C0 and C1 draw byte-identical topics in **25/25** comparable (round, slot)
cells; **C2 matches in 0/25** — because `select_parent(rng)` consumes 1–2
draws from the shared RNG before the topic loop. Paired evaluation seeds
(`seed0 = rng_seed*1000 + i*G`, arithmetic) are unaffected. Effect: the
C2−C0 and C2−C1 primary contrasts carry an unregistered topic-noise source
the C1−C0 contrast lacks, on an already underpowered comparison. Not fixable
mid-grid (re-pairing would change C2's remaining draws). For any re-run:
dedicated RNG streams per purpose (topics vs selection) restore the
registered design — an A1-style bug-fix amendment.

### C3. MAJOR — the memory digest gives both treatment arms objectively
false, self-contradictory guidance post-A2

`memory.py` still keys its "Too hard / hint-gated (… EXCLUDED from
fitness)" bucket on `win_nohint < 0.05` — the pre-A2 criterion A2 explicitly
retired. Post-A2, high partial-credit envs are valid and fitness-counted at
`win_nohint = 0`. In the actual logs, **8 C1 + 5 C2 valid, fitness-counted
records** are labeled "EXCLUDED from fitness" in the digest — including each
condition's highest-regret frontier envs (C1_r000_e05 regret 0.337;
C2_r001_e00 regret 0.278). Reconstructed digests show C2_r001_e00 listed
simultaneously under "build on these" and "EXCLUDED from fitness". The
in-context learning signal — the mechanism under test — is contradicting
itself on exactly the highest-fitness designs, plausibly steering the
designer *away* from its best region: a bias **against** the treatment arms.
`memory.py` is not frozen, but fixing it mid-grid would make rounds
non-comparable (same argument as the timeout patch). Log now, fix for any
re-run, caveat in the writeup. (Also: easy/hard buckets take the *first* 2
matching records ever, not recent/extreme ones; V1/V2 rejects never enter
memory at all — so C1 has no feedback path for the chronic near-600-char
hints, while C2's mutator does see V2 failures via `render_feedback`: an
undocumented feedback asymmetry between the two memory arms.)

### C4. MAJOR — float-precision artifact at the V3 boundary flipped a
C2 round: an exactly-in-band env was excluded

`C2_r002_e00` no-hint returns are `[1.0, 0.8500000000000001, 1.0]` (float
error from partial-reward summation) → `mean_nohint = 0.9500000000000001`.
`apply_band_gate` tests the **unrounded** value against the inclusive band,
so it failed V3 while the log prints the rounded value in a
self-contradictory line: *"no-hint mean return 0.95 outside [0.05, 0.95]"*.
Exact arithmetic gives exactly 0.95 = BAND_HI = in-band. With the env
admitted, C2 r2 = 3/6 valid, fitness 0.0167 (not 0.0000), and S0's archive
mean 0.0387 (not 0.0332). Replay confirms the realized selection sequence is
unchanged either way, but the logged trajectory, C2's condition mean, and
the archive weights are all altered — direction against C2. Fix for re-runs
(rounding or exact-fraction comparison at gate boundaries) needs an A1-style
amendment; for S4, report the endpoint with and without the env.

### C5. MAJOR (ops, live risk for rounds 4–5) — confirmed double-count
hazard in the prune-and-retry path

`cmd_round` writes the rounds.jsonl row and calls `record_fitness` **before**
the mutate LLM call. A BackendError at mutate (the outage mode that has
already struck three times) exits rc=1; `run_condition.sh` then prunes and
retries — but `prune_round.py` strips only envs/episodes/memory and its
docstring's claim that rounds.jsonl "is absent for a failed round by
construction" is falsified by the repo's own history:
`_aborted/c2_round0_crashed/` died at mutate with the rounds row AND S0's
fitness already persisted. An automatic retry from that state would have
silently double-counted the round in `select_parent` and in rounds.jsonl.
Smallest fix (operational, recovery-path only): prune matching rounds.jsonl
rows and pop the corresponding strategy-fitness entry, or make the driver
skip a round whose rounds.jsonl row already exists.

### C6. MAJOR (ops) — the committed "crash-safe" state cannot actually
resume: `.round_N_done` markers are gitignored

`run_condition.sh`'s only skip criterion is the marker file; `.gitignore`
excludes `runs/**/.round_*_done`; this clone has rounds 0–3 logged done and
zero markers. A restore-from-git (the recovery mode CLAUDE.md's
commit-after-every-round policy exists for) would rerun round 0 against
fully-populated logs with rc=0 — prune never fires — appending duplicate
envs, episodes, memory rows, rounds rows, fitnesses, and a spurious child,
all silently. The container-restart recovery survived only because the local
disk did. Fix: stop gitignoring the markers, or derive doneness from
rounds.jsonl (after C5's guard).

### C7. MAJOR — S3's archive lead is a 2-env, single-skill measurement now
steering both remaining rounds and the secondary endpoint

S3's 0.0513 comes from two optimization-skill envs only; its
logical_deduction / pattern_recognition arms have **never been measured**
(all died at design-timeout or V2). Weights: exp(0.0513/0.15)=1.408 vs S0's
exp(0.0332/0.15)=1.248 — a 0.018 edge from 2 envs of one skill vs S0's
18-env, 3-round mixed-skill mean, with per-round sd 0.035–0.086. And
`archive.best()` — which PREREG §5's held-out secondary endpoint would
naively use — currently returns S3. Full provenance exists (rounds.jsonl
n_valid + envs.jsonl per-env strategy_id/stage/regret), so S4 can recompute
env-weighted strategy means with **zero code change**; the held-out
"best C2 strategy" should be chosen from that recomputation, not from
`archive.best()`.

### C8. Minor findings

- `StrategyArchive._persist` rewrites strategies.jsonl in place with
  `open("w")` — non-atomic; a container kill mid-write can truncate the
  archive. 3-line fix (temp file + `os.replace`).
- `evolve.py`'s docstring promises an ε slot for the "least-tried lineage";
  the code is uniform choice over all untried. The 1.25-expected-evolved-
  rounds math assumed uniform, so it stands; the documented novelty pressure
  is not what runs.
- `C1_r002_e05`: all 6 episodes crashed with `NameError` (V1's random-walk
  battery missed the code path); it is recorded as evaluated/"too hard"
  (mean 0.00 → V3) and entered C1's digest as a difficulty exemplar. 6 of
  408 episodes; reclassify in S4 descriptives.
- C0 (frozen, memoryless — should be exchangeable draws) shows a monotone
  day-2 drift: mean_win_nohint 0.611, 0.611, 0.444, 0.222; C1 shows the same
  drop. n=4, may be chance; report round dates alongside the trajectory.
  Same-round cross-condition contrasts are protected (concurrent runs).
- Round-0 natural experiment: the aborted C2 round 0 (identical seed,
  identical topics, same S0) yielded 2/6 valid vs the re-run's 5/6 —
  designer sampling noise alone spans the grid's entire between-condition
  yield range. Useful for the S4 power discussion.
- Heads-up at snapshot time: C1 round 4's first 3 envs are all V3-invalid at
  mean_nohint 1.00 — 0/3 valid, INVALID_ROUND risk inside the final third
  (needs 2 valid of the remaining 3).
- probe_s0_a2 passed R1 (6/10, threshold 5) and R2 (0.20, threshold 0.15),
  but its R2 spread rests entirely on one env — the concentration problem
  was visible in the gate data.
- Descriptive dilution: learnable_fraction / mean_win_nohint divide by
  n_envs including design failures, conflating "not learnable" with "never
  measured" (r3: 1/6, 0.111). Descriptives only, never selected on.

## D. Suggestions

### D1. Safe now, mid-grid (operational, non-frozen, no measurement change)

1. **Guard the retry path before rounds 4–5 finish** (C5): extend
   `prune_round.py` to also drop matching rounds.jsonl rows and pop the
   corresponding strategy-fitness entry (or have the driver skip a round
   whose rounds row exists). Recovery tooling only; measurement untouched.
2. **Make committed state resumable** (C6): stop gitignoring
   `.round_N_done`, or derive doneness from rounds.jsonl (after 1).
3. **Atomic archive writes** (C8): temp + `os.replace` in `_persist`.
4. Optionally add additive `n_design_failures` / `n_pre_eval_rejects` keys
   to the rounds.jsonl row (cli.py, purely additive; also derivable post-hoc
   from envs.jsonl, so skipping is defensible).
5. Do **not** touch `memory.py`, `gates.py`, ε, `timeout_s`, or the RNG
   layout mid-grid — every one would trade a documented constant bias for a
   round-index-correlated one. (Same reasoning the timeout finding already
   applied.)

### D2. For Roland at the S4 STOP (cheapest first)

1. **Analysis-only remedies, zero code change**: report realized strategy
   per round; sensitivity analyses — primary endpoint with/without the
   contaminated r3, with/without the V3 boundary env; env-weighted strategy
   means; realized env counts per condition alongside fitness.
2. **Secondary endpoint**: pick the held-out "best C2 strategy" from
   env-weighted recomputed means, not `archive.best()` (C7).
3. **The pre-declared D-04 C2 replicate chain** (distinct `--rng-seed`, if
   ≥ 36 h runway) is the cheapest registered path to more C2 signal — no
   amendment needed. Pair it with a raised `timeout_s` (infrastructure,
   per the corrected mechanism in B1) or the replicate will be crippled the
   same way. It would still inherit C2/C3/C4 unless those bug-fix
   amendments are approved first — worth deciding as a package.
4. **Full C2 re-run** with raised `timeout_s` + A1-style bug-fix amendments
   (RNG stream separation, digest classifier updated to the A2 criterion,
   gate-boundary rounding) — the clean-arm option the timeout finding
   already proposed, now with the mechanism corrected.
5. **Selection redesign for any re-run** (dated amendment — the mechanism is
   registered): higher ε, or evaluate each new child once before exploit
   weighting. Under the current scheme 4/6 evolution outputs were
   structurally unevaluable (C1).

### D3. Writeup duties (S4)

P1–P4 with corrected mechanisms (B1, B2); C1 is the only condition whose
hints help on net (raw regret +0.0301 all-envs / +0.0587 valid-only, vs
C0 −0.0598 / +0.0255 and C2 −0.0196 / −0.0393) — a MEMORY_ALONE-flavored
descriptive the floored metric compresses; C0 nonstationarity; the round-0
yield natural experiment; backend_calls undercount of timeouts; the
C1_r002_e05 reclassification; probe R2 concentration.

## Provenance of this review

Produced by parallel analysis agents (run-data + evolution-code audits
completed; infra and PREREG audits partially, cut short by a usage limit)
whose load-bearing claims — selection replay, topic pairing, boundary env,
digest mislabels, code-order facts — were then re-verified by hand against
the raw logs and code before recording. PREREG.md was read in full;
no frozen surface was touched and no run was started.
