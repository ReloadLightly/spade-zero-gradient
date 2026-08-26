# S2 report — A2 re-probe (fresh topics, --rng-seed 1000, amended gate)

Run 2026-08-26, `claude` 2.1.245, G=3, n=10, S0 baseline, Sonnet designer /
Haiku solver. This is the run that gates the main grid; per PREREG §7 the
2026-08-25 probe counts as a PASS under neither band.

## 1. Verdict — all three gates PASS

    R1 interface  PASS   6/10 valid   (needs >= 5)
    R2 spread     PASS   0.200        (needs >= 0.15)
    R3 headroom   PASS   S0_fitness 0.0333, learnable_fraction 0.60

    wall clock    148.1 min   14.8 min/env
    backend calls 222         40.0 s/call    60 episodes

## 2. Per-env results

    id   mean_nh  win_nh  raw      floored  A2band  oldband
    e00  0.7167   0.6667  -0.2833  0.0000   valid   valid
    e01  1.0000   1.0000   0.0000  0.0000   -       -
    e02  0.7431   0.0000  -0.3453  0.0000   valid   -        A2 admits
    e03  0.7500   0.6667  -0.5000  0.0000   valid   valid
    e04  0.3333   0.3333   0.0000  0.0000   valid   valid
    e05  0.9667   0.6667  -0.0111  0.0000   -       valid    A2 excludes
    e06  0.4463   0.0000   0.2000  0.2000   valid   -        A2 admits
    e07  0.9167   0.6667  -0.2500  0.0000   valid   valid
    e08  0.9822   0.6667   0.0044  0.0044   -       valid    A2 excludes
    e09  1.0000   1.0000   0.0000  0.0000   -       -

## 3. What A2 actually did — narrower than claimed mid-run

Both bands admit exactly **6/10**. A2 did not widen eligibility; it **swapped
membership**, trading two near-saturated ceiling envs (e05 0.9667, e08 0.9822)
for two floor envs the win-rate gate could not see (e02, e06).

    A2  band: valid 6/10  spread 0.2000   R1 PASS  R2 PASS
    old band: valid 6/10  spread 0.0044   R1 PASS  R2 FAIL

So **A2 carries R2 alone, not both gates.** At 7 envs this note said the old
band would have failed both; with all 10 in that is wrong — R1 passes either
way. The entire difference is e06, the one env in the probe with meaningful
positive regret (+0.2000), which the win-rate floor excluded for having zero
full solves while the solver scored 0.45 mean return cold.

The two ceiling exclusions look correct rather than incidental: both envs the
solver nearly maxes unaided, and both carry negligible regret (-0.0111,
+0.0044). Excluding them is what V3's ceiling is for; the win-rate band was
letting them through.

## 4. A1 was NOT exercised by this run

Zero envs were rejected pre-evaluation, so the crash path A1 fixes was never
hit. The report was written, but that only shows the run had no V1/V2
rejection — not that the fix works in situ. A1's evidence remains the two
unit tests, one of which was verified to fail against the pre-A1 expression.
The live path stays untested until a round contains a V1/V2 rejection; the
2026-08-25 probe had one in ten, so S3 will likely produce some.

## 5. The finding that matters more than the gates

    raw regret n=10: mean -0.1185, median -0.0056
    helped 2, harmed 5, neutral 3

Hint-harm dominates on fresh topics: five envs where the hint made a frozen
Haiku solver worse (-0.5000, -0.3453, -0.2833, -0.2500, -0.0111) against two
positives (+0.2000, +0.0044). The 2026-08-25 probe split 5 helped / 4 harmed;
this one is 2 / 5. Across both probes: 7 helped, 9 harmed, 3 neutral in 19
evaluated envs.

Consequence for S3, stated before the grid runs rather than after:
`S0_fitness` is **0.0333**, and it is the mean of [0, 0, 0, 0, 0.2, 0] — a
single env in six carrying the entire selective signal while flooring clips
five negatives to zero. If that rate holds, every condition's round fitness
will sit near zero and be driven by whichever one or two envs happen to land
positive. The C2−C0 and C1−C0 contrasts would then rest on a handful of envs
per condition, which is thin ground for the final-third comparison in §5.
This does not block the grid — R1–R3 passed as registered — but it is the
likeliest reason the primary endpoint returns NO_TRACTION, and if that
obtains it should be reported as a measurement-floor result, not as evidence
that in-context design cannot work.

## 6. Throughput for D-04

14.8 min/env sustained (40.0 s/call), against 7.9 min/env on 2026-08-25.
Earlier estimates in this session ran high because probe_e00 was a double
outlier (46 steps AND a slow patch, 61.6 min alone); the 10-env figure is the
reliable one. D-04's 108-env grid is therefore ~26 h sequential, inside the
36 h runway the optional replicate is framed around. Real three-process
throughput will be measured on round 0 before the full grid commits.
