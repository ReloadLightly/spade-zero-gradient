STRATEGY S0 (baseline — distilled from SPADE, arXiv:2608.19197, games setting):

Design one self-contained, seeded, multi-turn game per request that tests the
target skill through interaction, not recall.

1. Target the solver's frontier: the game should be winnable by a careful
   solver in 4–8 steps, but not on the first attempt without exploration.
   Avoid both extremes — puzzles a solver answers correctly in one step, and
   puzzles that cannot be solved without privileged information.
2. Make it state-gated: hide the key quantity in internal state and force the
   solver to discover it through probing actions with informative feedback
   (comparisons, partial reveals, constraint eliminations). The first
   observation must state the goal, the exact action format, and the step
   limit; every subsequent observation must carry usable evidence.
3. Reward verifiable progress: 1.0 total for success; where natural, split
   into 2–4 partial rewards for objectively checkable milestones computed
   from internal state. Never reward the agent for merely claiming progress.
4. Ground the game in the given TOPIC for surface variety, but keep the
   underlying mechanic aligned with the TARGET SKILL (deduction: eliminate
   hypotheses; pattern recognition: infer a hidden rule from instances;
   optimization: choose under a budget with feedback on cost/value).
5. Keep the action grammar tiny and stated explicitly (one or two verb forms
   with arguments), and handle malformed actions gracefully with a corrective
   observation, no reward, and the step still counted.
6. The HINT must be a genuine strategic shortcut — the insight an expert would
   use (an invariant, an ordering, a halving argument) — that measurably
   raises the win rate, while never stating the answer, the target values, or
   any literal action string.
7. Vary concept, surface story, and difficulty dial away from anything listed
   as recent or too easy in MEMORY; if MEMORY shows too-hard designs, lower
   the discovery burden (fewer hidden variables, more informative feedback),
   not the step count.
