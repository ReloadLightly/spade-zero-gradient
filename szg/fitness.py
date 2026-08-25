"""Fitness aggregation and descriptive metrics.

Selective fitness (the ONLY thing selection sees, D-02):
    strategy_fitness = mean floored hint-based regret over its VALID envs
    (valid = passes V1/V2/V3 gates; a round with < MIN_VALID valid envs
    scores 0.0 and is flagged INVALID_ROUND).

Descriptive metrics (logged, reported, never selected on):
    learnable fraction (no-hint win in [0.2, 0.8], as in SPADE Fig. analysis),
    valid fraction, mean win rates, diversity proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gates import is_learnable

MIN_VALID = 2


@dataclass
class RoundScore:
    fitness: float
    n_envs: int
    n_valid: int
    invalid_round: bool
    learnable_fraction: float
    mean_win_nohint: float
    diversity: float
    per_env: list = field(default_factory=list)


def diversity_proxy(codes: list[str]) -> float:
    """Distinct-token-trigram ratio across env sources (cheap Vendi stand-in)."""
    all_tris: list[tuple] = []
    for code in codes:
        toks = code.split()
        all_tris += [tuple(toks[i:i + 3]) for i in range(max(0, len(toks) - 2))]
    if not all_tris:
        return 0.0
    return len(set(all_tris)) / len(all_tris)


def score_round(env_records: list[dict]) -> RoundScore:
    """env_records: dicts with floored_regret, win_nohint, valid_for_fitness, code."""
    valid = [r for r in env_records if r.get("valid_for_fitness")]
    invalid_round = len(valid) < MIN_VALID
    fitness = 0.0 if invalid_round else sum(r["floored_regret"] for r in valid) / len(valid)
    n = len(env_records)
    return RoundScore(
        fitness=round(fitness, 4),
        n_envs=n,
        n_valid=len(valid),
        invalid_round=invalid_round,
        learnable_fraction=(sum(1 for r in env_records if is_learnable(r.get("win_nohint", -1))) / n) if n else 0.0,
        mean_win_nohint=(sum(r.get("win_nohint", 0.0) for r in env_records) / n) if n else 0.0,
        diversity=round(diversity_proxy([r.get("code", "") for r in env_records]), 4),
        per_env=[{k: r.get(k) for k in
                  ("env_id", "skill", "concept", "win_nohint", "win_hint",
                   "floored_regret", "valid_for_fitness")} for r in env_records],
    )
