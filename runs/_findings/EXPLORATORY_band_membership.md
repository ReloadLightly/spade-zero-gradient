# Exploratory — the designer may be learning DIFFICULTY CALIBRATION, not regret

**Status: exploratory, post-hoc, NOT a registered endpoint.** This measure was
chosen after seeing the data. It is exactly the forking path PREREG guards
against, and it is written down here as a hypothesis to be pre-registered and
tested on fresh data — not as a result.

## The observation

The registered fitness (floored hint-based regret) shows nothing. But the
fraction of environments landing inside the V3 validity band — the designer's
ability to hit a usable difficulty level at all — is ordered and large:

    condition                envs in band      rate
    C0 static                18/33             0.545
    C1 memory                26/34             0.765
    C2 memory + evolution    27/32             0.844

    C1 − C0   +0.219   z=1.89   p=0.059    (topics paired 24/24)
    C2 − C0   +0.298   z=2.61   p=0.009    (topics NOT paired — confounded)
    C2 − C1   +0.079   z=0.81   p=0.420

On the only properly paired contrast, C1 vs C0 on identical topics, a McNemar
test over 31 matched pairs gives:

    both in band 12 | neither 3 | only C1 12 | only C0 4
    McNemar χ² = 3.06, p = 0.080

A 3:1 discordance in favour of the memory condition, just short of
significance at n=31.

## Why this is mechanistically plausible

The memory digest literally reports, per past environment, whether it was too
easy or too hard. That is direct, low-variance feedback on *difficulty
calibration*. It is not feedback on how to make a hint more useful — which is
what floored regret actually selects on, and which is a far subtler target.

So the designer may be learning the thing its feedback channel can teach,
while the registered endpoint measures something else.

## Why it is a better measurement than floored regret

1. **A proportion, not a floored mean.** Bounded, no floor effect, no discarded
   half of the distribution.
2. **Measured per environment.** n ≈ 33 per condition here, versus n = 2 rounds
   for the registered endpoint. The main grid spent 108 environment
   measurements to obtain n = 2 per arm.
3. **It is SPADE's own criterion.** A "learnable" environment is defined by
   difficulty, not by regret magnitude.
4. **It survives the noise that sank the primary endpoint.** Round-level
   fitness sd was 0.084 against a mean of 0.079; this contrast is a rate
   difference of 0.22 with a standard error near 0.12.

## Caveats that must not be dropped

- Post-hoc measure selection. Needs pre-registration and fresh data.
- C2's topics are unpaired with C0's (the `select_parent` RNG bug), so the
  significant C2−C0 result is confounded and the clean C1−C0 one is not
  significant.
- The C1/C2 memory digest classified environments by the pre-A2 win band, so
  the treatment arms received a partly self-contradictory signal. If anything
  this biases against the effect.
- G=3 makes each environment's `mean_nohint` noisy, which blurs band
  membership near the edges.
