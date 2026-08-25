# spade-zero-gradient

Can SPADE's Environment Designer improve **without weight updates**?

[SPADE](https://arxiv.org/abs/2608.19197) (Liu, Yu et al. 2026, UW/Jaques
group) trains a single LLM to both write executable Gym-style training
environments and solve them, rewarding the designer with *hint-based regret* —
the solver's return with a privileged hint minus without — optimized by RL.
Its Future Directions leave one question open:

> "Future work could explore Environment Designers that improve through
> in-context learning rather than gradient updates."

This repo answers exactly that, at small scale, with frozen models only:

- **Designer** (Sonnet, frozen) improves only through (a) an accumulated
  in-context **environment memory** and (b) **ShinkaEvolve-style evolution of
  its written design strategy**, selected on floored hint-based regret.
- **Solver** (Haiku, frozen) removes SPADE's co-evolution confound: any
  fitness trend is attributable to the designer side.
- Conditions: **C0** static designer · **C1** memory only · **C2**
  memory + strategy evolution. Fitness and gates are pre-registered in
  [PREREG.md](PREREG.md); all outcomes (including null) are reportable.

## Layout

    szg/            harness: env contract & sandbox, runner, gates, memory,
                    designer, evolution, fitness, CLI
    strategies/     S0 baseline designer strategy (control + evolution seed)
    seeds/          fixed topic grounding pack (60 topics, 3 skills)
    tests/          acceptance tests pinning the package API (no model calls)
    runs/           JSONL logs (envs, episodes with transcripts, rounds)
    PREREG.md       frozen design: fitness, gates, endpoints, decision log
    CLAUDE.md       stage plan for Claude Code main runs (subscription-backed
                    headless `claude -p`, no API key)

## Quickstart

    pip install -e . && pip install pytest
    make test     # acceptance tests, no model calls
    make smoke    # machinery gates R4/R5, no model calls
    make gates    # R1-R3 real-model probe (claude CLI) — STOP gate
    python -m szg.cli round --condition C2 --round-index 0 --n-envs 6 \
        --mutate --out runs/c2_chain0

## Status

- 2026-08-25: v0.1 scaffold; 14/14 acceptance tests green; machinery smoke
  green; in-session pilot episode logged under `runs/pilot/`.
