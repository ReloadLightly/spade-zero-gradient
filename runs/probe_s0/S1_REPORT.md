# S1 report — R1–R3 probe, S0 baseline, Sonnet designer / Haiku solver

Run 2026-08-25, `claude` 2.1.245, Python 3.11.15, G=3, n=10, seed grid
`seed0 = i*G`. Backend: headless `claude -p` in a Claude Code remote
container (NOT Roland's machine — see §5).

`make test` 14/14 green and `make smoke` R4/R5 PASS were confirmed before
the probe.

## 1. Verdict

    R1 interface  FAIL   3/10 valid  (needs >= 5)
    R2 spread     FAIL   0.083       (needs >= 0.15)
    R3 headroom   PASS   S0_fitness 0.0376, learnable_fraction 0.30

Two of three gates red. Per CLAUDE.md this is a finding, not an obstacle;
nothing was tuned to move it.

R3 passes on a technicality worth naming: it passes because S0 is *weak*
(fitness 0.0376, nowhere near the 0.6 saturation trip), not because it
leaves healthy headroom. A near-zero baseline and a saturated one both
clear this gate; only the first is good news, and this is the first.

## 2. Cost (for D-04)

    wall clock          78.8 min total, 7.9 min/env
    backend calls       201 (>=), 20.1 per env
      solver            191 (= sum of episode steps over 54 episodes)
      designer          10  (1 per env; ClaudeCLIBackend retries unlogged)
    episodes            54  (9 evaluated envs x 2 arms x G=3)

Scaling note: S3 is three conditions across a round grid with replicate
chains. At 7.9 min/env this container gives roughly 7.6 envs/hour. A
100-env condition is ~13 h; the three-condition grid with replicates is
measured in days, not hours.

## 3. Per-env results

    id   stage      win_nh  mean_nh  raw      floored  V3(win)  V3(mean)
    e00  evaluated  0.6667  0.9167    0.0833  0.0833   pass     pass
    e01  evaluated  0.0000  0.4000    0.2000  0.2000   FAIL     pass
    e02  evaluated  0.3333  0.9444   -0.2006  0.0000   pass     pass
    e03  evaluated  1.0000  1.0000   -0.7778  0.0000   FAIL     FAIL
    e04  evaluated  0.0000  0.0000    0.5000  0.5000   FAIL     FAIL
    e05  evaluated  0.6667  0.9343    0.0295  0.0295   pass     pass
    e06  evaluated  0.0000  0.4333   -0.1333  0.0000   FAIL     pass
    e07  evaluated  0.0000  0.5000   -0.2000  0.0000   FAIL     pass
    e08  gates          -       -         -       -    REJECT   REJECT
    e09  evaluated  0.0000  0.1667    0.3333  0.3333   FAIL     pass

Failure modes: V3 floor 6, V3 ceiling 1, pre-eval V2 1, valid 3.
Diversity proxy 0.8186. mean_win_nohint 0.2667.

## 4. Findings

**F1 — V3 excludes the envs that carry the fitness signal.** The gate reads
`win_nohint` (return >= 0.99); fitness reads mean return. Six of ten envs
fail at the floor, and they hold the probe's three largest regrets (0.50,
0.33, 0.20). Every admitted env is under 0.09. R2 cannot clear on the
admitted population — not for want of envs, but because the gate removes
the high-regret ones by construction. Detail in DIAGNOSTIC_hint_arm.md §1.

**F2 — hints harm the solver about as often as they help, and flooring
hides it.** Of 9 evaluated envs, 5 helped and 4 harmed (-0.7778, -0.2006,
-0.2000, -0.1333), all recorded as 0.0000.

Direction is NOT established, and this note has flipped twice as n grew
(median negative at n=8, positive at n=9); excluding e03 both mean and
median are positive. That instability is itself the result: at n=10 the
sign of the effect is not resolvable. What is robust is the *count* —
roughly half of S0's hints make a frozen solver worse.

Consequence for Phase A: selection sees only the positive part, so a
designer that learns to write better hints and one that learns to fail
less catastrophically are indistinguishable under the registered fitness.

**F3 — hint-induced confabulation (e03).** Cold [1.0,1.0,1.0] in 3.67
steps; hinted [0.167,0.167,0.333] in exactly 2.00 steps every seed. After
one CLUE returning "Clue 1/4" the solver asserted it had all four clues,
fabricated 2–4 and solved on them. Transcript in DIAGNOSTIC_hint_arm.md §2.

**F4 — V2 compliance miss (e08).** 650-char hint against the 600 limit,
which designer.py:31 states explicitly. Rejected pre-evaluation, costing a
designer call and no solver calls. This is the most tractable failure in
the batch: obeying a constraint it was already given is exactly what an
accumulated in-context memory (C1) could fix without weight updates.

**F5 — CRASH: cmd_probe never wrote its report.** `fitness.py:56` calls
`is_learnable(r.get("win_nohint", -1))`. For a pre-eval-rejected env the
key EXISTS with value None, so the -1 sentinel never applies and
`gates.py:74` raises `TypeError: '<=' not supported between instances of
'float' and 'NoneType'`.

This fires whenever any env fails V1/V2 — it would kill any real S2/S3
round, not just this probe. No data was lost (all 10 records are in
envs.jsonl); only the aggregation died, after every model call was paid
for. `probe_report.reconstructed.json` is recomputed offline from the logs
using the -1 semantics the code intended; the tool's own report does not
exist for this run.

fitness.py is frozen, so nothing was patched. Minimal compliant fix, for
sign-off:

    -    is_learnable(r.get("win_nohint", -1))
    +    is_learnable(r.get("win_nohint") if r.get("win_nohint") is not None else -1)

This changes no registered semantics: an unevaluated env is not learnable,
which is what the -1 default already encoded.

## 5. Environment caveat

This ran in a Claude Code remote container, not the machine CLAUDE.md
describes. `claude -p` worked, but whether it drew on the Max subscription
the way the local route does could not be verified from inside. PREREG
assumes that route. Acceptable for a probe; confirm before S3.

## 6. STOP — decisions for Roland

1. **D-04 scale.** Numbers in §2. My input: this container is too slow for
   the S3 grid at any serious scale (~7.6 envs/hour).

2. **F5 crash fix.** Needs sign-off since fitness.py is frozen. Without it
   any round containing a V1/V2 rejection loses its report. Recommend
   fixing before S2 — the probe shows V2 rejections do occur.

3. **V3 amendment** (proposed in DIAGNOSTIC_hint_arm.md §3): band on
   `mean_nohint` instead of `win_nohint`.

       current   valid 3/10  spread 0.0833   R1 FAIL  R2 FAIL
       proposed  valid 7/10  spread 0.3333   R1 PASS  R2 PASS

   **Read that with maximum suspicion.** It was chosen after seeing the
   data, on n=10, and it flips the entire S1 outcome from two red gates to
   two green. That is precisely the forking path PREREG exists to prevent,
   and the fact that it rescues the run is an argument for distrusting it,
   not for adopting it. It is presented because F1 is a real measurement
   defect that would be worth fixing even if it changed no gate — not
   because S1 needs to pass. If adopted, re-probe under the amended gate
   and report both results.

4. **Reportability.** F2/F3 stand on their own regardless of the above:
   "in-context hints can degrade a frozen solver, and floored regret hides
   it" is a result about hint design, independent of whether the designer
   improves.
