# S1 diagnostic — hint arm behaviour and the V3 / fitness measure mismatch

Written during the R1–R3 probe (first 6 of 10 envs). No frozen file was
touched. Everything below is read-only analysis of `envs.jsonl` and
`episodes.jsonl`. Two findings, then one proposed amendment, then STOP.

## 0. Hint injection is NOT broken (hypothesis refuted)

`win_hint == 0.0` in 3 of the first 5 envs looked like a delivery defect.
It is not. The hint is appended to the solver system prompt exactly as
intended:

    A privileged HINT is available for this episode:
    HINT: <text>

Both arms are otherwise identical and share seeds. e00 (`win_hint` 1.00) and
e05 (0.667) rule out a systematic degradation. Recording this so the
hypothesis is not re-run later.

## 1. Finding A — V3 excludes exactly the envs where the hint works

Fitness and the gate read different quantities:

    fitness  floored_regret = max(0, mean_hint - mean_nohint)   [mean return]
    gate V3  win_nohint in [0.05, 0.95]         [win rate, i.e. return >= 0.99]

An env can show large partial-credit regret while never being *fully* solved
in either arm. V3 sees only full solves, so it drops it.

    id   win_nh  mean_nh  regret  V3
    e00  0.6667  0.9167   0.0833  pass
    e01  0.0000  0.4000   0.2000  FAIL floor
    e02  0.3333  0.9444   0.0000  pass
    e03  1.0000  1.0000   0.0000  FAIL ceiling
    e04  0.0000  0.0000   0.5000  FAIL floor
    e05  0.6667  0.9343   0.0295  pass

The two largest regrets in the probe (0.50, 0.20) are both excluded. Every
admitted env carries regret < 0.09. This is not a coincidence of small n:
it is structural. Envs hard enough for the hint to matter are the ones whose
cold win rate falls under the 0.05 floor.

Consequence for R2: spread over the admitted population is 0.0833 against a
0.15 gate, and no additional low-regret valid env can raise it.

e01 is the clearest case. Cold returns [0.4, 0.4, 0.4]; with the hint
[0.7, 0.4, 0.7]. The solver plainly makes progress without the hint — it is
not a password environment — but it never reaches 0.99, so `win_nohint` is
0.00 and V3 drops it at the floor.

## 2. Finding B — hint-induced confabulation (separate issue)

e03: cold [1.0, 1.0, 1.0] in 3.67 steps; with the hint
[0.167, 0.167, 0.333] in exactly 2.00 steps every seed. The hint arm is
terminating early and losing.

From the seed-9 hint transcript, after ONE `CLUE` action returning
"Clue 1/4", the solver wrote:

    "Clue 4 confirms: Bird -> Coffee. Now I have all four clues and can
     solve with certainty" -> ACTION: SOLVE bird coffee rabbit milk fish juice

It had received one clue. Clues 2–4 were fabricated, and it answered on
them. The no-hint arm, given the same env and seeds, requested clues one at
a time and scored 1.0.

The hint (V2-legal: 517 chars, no ACTION line, no fence) opens
"Request every clue before you SOLVE — there are exactly four". Both arms
can learn the count from the env's own "Clue 1/4" response, so the count is
not leaked information; the plausible mechanism is that stating the total
up front invited the model to treat the set as known rather than to gather
it. That is a hypothesis from one env at G=3, not an established cause.

Why it matters regardless of cause: a hint can make the solver worse, and
`max(0, ...)` silently floors that to 0.0. e03 and e02 both have a genuinely
negative raw regret and are recorded as 0.0000, so the fitness signal cannot
distinguish "hint was useless" from "hint was actively harmful". Under D-02
that distinction is invisible to selection.

## 3. Proposed amendment (NOT applied — needs sign-off)

Smallest compliant change: have V3 band the same quantity fitness uses.

    V3  mean_nohint in [0.05, 0.95]      (was: win_nohint)

Recomputed over the 6 probe envs:

    current   V3 on win_nohint : valid=3  regrets=[0.0833, 0.0, 0.0295]       spread=0.0833  R2 FAIL
    proposed  V3 on mean_nohint: valid=4  regrets=[0.0833, 0.2, 0.0, 0.0295]  spread=0.2000  R2 PASS

It preserves V3's stated purpose. The password-environment optimum D-02
blocks is an env yielding nothing without the hint: e04 (mean_nohint 0.000)
is still excluded, as is e03 (1.000, solved cold). Only e01 is admitted —
the env where the solver demonstrably makes cold progress.

**Caveat, stated plainly.** This rule was chosen after looking at probe
data, on n=6, and it changes a gate outcome from FAIL to PASS. That is
exactly the forking path PREREG exists to prevent. It should not be adopted
because it rescues R2. If it is adopted, the honest course is to re-run the
probe under the amended gate and report both the original and amended
results.

Finding B suggests a second candidate — log raw (unfloored) regret alongside
the floored value, so hint-harm is visible in the data — but that touches
fitness.py and is not proposed here.

## 4. STOP

For Roland, at the S1 STOP, alongside D-04:

1. Accept, reject, or defer the V3 amendment above.
2. If accepted: re-probe under the amended gate, or carry the original
   result forward as the registered one?
3. Finding B is reportable on its own — "in-context hints can degrade a
   frozen solver" is a result about hint design, independent of whether
   the designer improves.
