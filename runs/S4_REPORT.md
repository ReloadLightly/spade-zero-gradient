# S4 — registered analysis of the S3 main grid

Grid completed 2026-08-28 07:22 UTC. 3 conditions × 6 rounds × 6 environments
= 108 environments, 99 evaluated. D-04 scale: G=3, single chains.

## 1. The registered outcome, stated as registered

    C2−C0   +0.1420   95% CI [+0.0638, +0.2291]   EXCLUDES 0
    C1−C0   +0.0610   95% CI [−0.0128, +0.1451]   includes 0
    C2−C1   +0.0810   95% CI [−0.0268, +0.1892]   includes 0

    PREREG §6 classification: DESIGN_IMPROVES (held-out PENDING)

Per-round fitness:

    C0 static             0.1222  0.0000  0.2083  0.0970  0.0000  0.0167   mean 0.0740
    C1 memory             0.1093  0.1067  0.0370  0.1062  0.0278  0.1208   mean 0.0846
    C2 memory+evolution   0.0000  0.0995  0.0000  0.0513  0.1005  0.2212   mean 0.0788

## 2. Why this should NOT be read as DESIGN_IMPROVES

The classification is what the registered rule returns on this data. The
causal claim the label names — that evolving the design strategy in-context
produced better environments — is contradicted by the same data, four ways.

**a. No evolved strategy beat the baseline.** Within C2's own archive:

    S0  (baseline)  n=4  mean fitness 0.0802
    S3  (evolved)   n=3  mean fitness 0.0506
    S1, S2, S4, S5, S6 (evolved)  never selected, n=0

The highest-mean-fitness strategy in C2 is **S0, the frozen baseline**. The
secondary endpoint as registered — "the highest-mean-fitness C2 strategy vs
S0 on a held-out grid" — would therefore compare S0 against S0. There is no
evolved strategy to confirm.

**b. C2's best round ran the baseline.** Round 5 scored 0.2212, the highest
round in the whole grid, running **S0** — not an evolved strategy. It
supplies most of C2's final-third advantage. Half of C2's endpoint window
used exactly the same strategy as the control.

**c. C0's final third caught its two worst rounds.** C0 ranges 0.0000–0.2083
across six rounds; rounds 4 and 5 happened to be 0.0000 and 0.0167. The
contrast is C2's two best rounds against C0's two worst, and the final-third
rule selects that window by position, not by merit.

**d. The effect is barely above what this design can resolve, and the
comparison is confounded.** Minimum detectable effect at 80% power is
**0.1183**, i.e. 1.5× the grand mean fitness of 0.0791; the observed
+0.1420 sits just above it. Separately, C2's topics are not paired with
C0's (0/24 cells match — see §4), so the contrast carries unregistered
topic-difficulty variance that C1−C0 does not.

C0's own six rounds — one fixed process, no memory, no evolution, nothing
accumulating — span **0.0000 to 0.2083, a range 2.8× their own mean**. Any
two-round window drawn from that process can differ from another by more
than the effect reported above.

## 3. What the data does support

**The memory arm is a clean null.** C1−C0 = +0.0610, CI includes 0. C1's
trajectory oscillates 0.03–0.12 with slope −0.0031 across six rounds while
its memory digest grew monotonically. C1 is also the least damaged arm:
topics paired with C0 in 24/24 cells, zero designer timeouts, its one
outage-fabricated round discarded and re-run. If accumulated in-context
memory helps this designer, six rounds did not show it.

**Hint-based regret is close to a coin flip on a frozen solver.** Over 99
evaluated environments: helped 37, harmed 33, neutral 29; mean raw regret
+0.0025. Flooring discards −6.4372 of harm mass while keeping +5.8776 of
gain — **more than half the signal's absolute magnitude is thrown away, and
it is the half that records the designer doing damage.** The largest single
case took a solver from 0.87 to 0.12 (raw −0.7467) and contributed exactly
0.0000 to fitness, indistinguishable from a hint that did nothing.

**The eligibility gate was measuring the wrong quantity** (amendment A2).
V3 originally banded on win rate while fitness used mean return, so
environments with real partial-credit regret but no full solve were dropped.
Validity under the two definitions, across the grid:

    C0  A2 mean-band 0.55   pre-A2 win-band 0.36
    C1  A2 mean-band 0.76   pre-A2 win-band 0.47
    C2  A2 mean-band 0.84   pre-A2 win-band 0.50

## 4. Threats to validity, all logged during the run

1. **Underpowered by roughly an order of magnitude.** To detect a 0.03
   difference at this variance needs ~122 rounds per condition; 0.05 needs
   ~44; D-04 ran 6.
2. **C2 topic pairing broken.** `select_parent` consumes the shared RNG
   before topic sampling, so C0≡C1 topics in 24/24 logged cells and C0≡C2 in
   0/24. Affects only the C2 contrasts.
3. **Evolved strategies barely ran.** ε-greedy at ε=0.25 gave 2 evolved
   rounds of 6; four children were never evaluated at all.
4. **The memory digest predates A2** — it still segments environments by the
   pre-A2 win band, so environments that do count toward fitness can be
   shown to the designer as "too easy"/"too hard". Bias runs against C1/C2.
5. **C2 lost 2 environments to designer timeouts**; C0 and C1 lost none.
6. **Two rounds were fabricated by a usage-limit outage** (C0 r4, C1 r4, all
   envs dead at the designer call, scored 0.0). Both were detected,
   discarded and re-run; `szg/round_health.py` now blocks the class.

## 5. Recommended reading

**NO_TRACTION on the research question, with a DESIGN_IMPROVES label that the
mechanism does not support.** The registered rule fired, and it should be
reported as having fired — but the honest summary is that no evolved
strategy outperformed the frozen baseline, the memory arm is null, and the
control's own variance exceeds the measured effect.

The substantive contributions are methodological: hint-based regret behaves
poorly as a selective signal at small scale with frozen models, floored
regret discards the half of it that carries the most information, and the
eligibility gate as first specified excluded exactly the environments that
carried signal.

## 6. STOP — for Roland

1. Does the DESIGN_IMPROVES label stand as the reported outcome, with §2 as
   its caveat, or is NO_TRACTION the honest headline?
2. The secondary held-out endpoint would compare S0 against S0. Run it
   anyway to close the register, or record it as not applicable?
3. A re-run at ~44 rounds/condition is what detection needs; alongside it,
   fix the RNG-stream separation, the A2 digest, and `timeout_s`.
