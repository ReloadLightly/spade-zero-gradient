"""Command-line entry points. Everything logs JSONL under runs/.

Conditions (PREREG):
  C0  static designer: S0 strategy, memory OFF        (SPADE's no-training/no-memory control)
  C1  memory only:     S0 strategy, memory ON         (pure in-context learning)
  C2  memory+evolution: memory ON, strategy evolved   (the full gradient-free designer)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from . import env_api
from .backends import make_backend
from .designer import generate
from .evolve import Strategy, StrategyArchive, mutate, render_feedback
from .fitness import score_round
from .gates import apply_band_gate, pre_eval_gates
from .memory import EnvMemory
from .runner import evaluate_env

SKILLS = ["logical_deduction", "pattern_recognition", "optimization"]  # D-01


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_topics(path: str = "seeds/topics.jsonl") -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sample_topic(topics: list[dict], skill: str, rng: random.Random) -> str:
    pool = [t["topic"] for t in topics if t["skill"] == skill] or [t["topic"] for t in topics]
    return rng.choice(pool)


def run_env_cycle(backend, designer_model: str, solver_model: str,
                  strategy_text: str, skill: str, topic: str,
                  memory_digest: str | None, G: int, seed0: int,
                  out_dir: Path, env_id: str) -> dict:
    """One full cycle: design -> gates -> paired evaluation -> record dict."""
    record: dict = {"env_id": env_id, "skill": skill, "topic": topic,
                    "t": _now(), "G": G, "seed0": seed0,
                    "designer_model": designer_model, "solver_model": solver_model}
    try:
        design = generate(backend, designer_model, strategy_text, skill, topic,
                          memory_digest)
    except Exception as e:  # noqa: BLE001
        record.update(stage="design", error=repr(e), valid_for_fitness=False,
                      gate_issues=[f"design failure: {e!r}"])
        return record
    record.update(concept=design.concept, code=design.code, hint=design.hint)
    (out_dir / "envs").mkdir(parents=True, exist_ok=True)
    (out_dir / "envs" / f"{env_id}.py").write_text(design.code)

    gates = pre_eval_gates(design.code, design.hint)
    if not (gates.v1_battery and gates.v2_hint):
        record.update(stage="gates", valid_for_fitness=False,
                      gate_issues=gates.issues, win_nohint=None)
        return record

    ev = evaluate_env(design.code, design.hint, backend, solver_model,
                      G=G, seed0=seed0)
    gates = apply_band_gate(gates, ev.mean_nohint)   # A2 2026-08-26
    record.update(
        stage="evaluated",
        win_nohint=round(ev.win_nohint, 4), win_hint=round(ev.win_hint, 4),
        mean_nohint=round(ev.mean_nohint, 4), mean_hint=round(ev.mean_hint, 4),
        floored_regret=round(ev.floored_regret, 4),
        # A2 descriptive (2026-08-26): raw gap, negative when the hint HARMS
        # the solver. Never selected on; flooring remains the selective rule.
        raw_regret=round(ev.mean_hint - ev.mean_nohint, 4),
        valid_for_fitness=gates.valid_for_fitness,
        gate_issues=gates.issues,
        episode_meta=[{"arm": arm, "seed": seed, "return": r.return_,
                       "steps": r.steps, "ended": r.ended}
                      for arm, seed, r in ev.episodes],
    )
    for arm, seed, r in ev.episodes:
        _jsonl(out_dir / "episodes.jsonl",
               {"env_id": env_id, "arm": arm, "seed": seed,
                "return": r.return_, "steps": r.steps, "ended": r.ended,
                "transcript": r.transcript})
    return record


def cmd_round(args) -> int:
    out_dir = Path(args.out)
    rng = random.Random(args.rng_seed)
    backend = make_backend(args.backend)
    topics = load_topics(args.topics)
    memory = EnvMemory(out_dir / "memory.jsonl") if args.condition in ("C1", "C2") else None
    archive = StrategyArchive(out_dir / "strategies.jsonl") if args.condition == "C2" else None

    if archive is not None and not archive.strategies:
        archive.add(Strategy("S0", None, Path(args.strategy).read_text(), []))
    if archive is not None:
        parent = archive.select_parent(rng)
        strategy_text, strategy_id = parent.text, parent.id
    else:
        strategy_text, strategy_id = Path(args.strategy).read_text(), "S0"

    digest = memory.digest() if memory is not None else None
    records = []
    for i in range(args.n_envs):
        skill = SKILLS[i % len(SKILLS)]
        env_id = f"{args.condition}_r{args.round_index:03d}_e{i:02d}"
        rec = run_env_cycle(backend, args.designer_model, args.solver_model,
                            strategy_text, skill, sample_topic(topics, skill, rng),
                            digest, args.G, seed0=args.rng_seed * 1000 + i * args.G,
                            out_dir=out_dir, env_id=env_id)
        rec.update(condition=args.condition, round_index=args.round_index,
                   strategy_id=strategy_id)
        _jsonl(out_dir / "envs.jsonl", rec)
        records.append(rec)
        if memory is not None and rec.get("stage") == "evaluated":
            memory.add({k: rec.get(k) for k in
                        ("env_id", "skill", "concept", "win_nohint", "win_hint",
                         "floored_regret", "valid_for_fitness", "gate_issues")})

    score = score_round(records)
    _jsonl(out_dir / "rounds.jsonl",
           {"t": _now(), "condition": args.condition, "round_index": args.round_index,
            "strategy_id": strategy_id, "fitness": score.fitness,
            "n_valid": score.n_valid, "n_envs": score.n_envs,
            "invalid_round": score.invalid_round,
            "learnable_fraction": score.learnable_fraction,
            "mean_win_nohint": score.mean_win_nohint, "diversity": score.diversity,
            "backend_calls": backend.stats.calls})

    if archive is not None:
        archive.record_fitness(strategy_id, score.fitness)
        if args.mutate:
            child_id = f"S{len(archive.strategies)}"
            child = mutate(backend, args.designer_model,
                           archive.strategies[strategy_id],
                           render_feedback(score, records), child_id)
            archive.add(child)
            print(f"mutated {strategy_id} -> {child_id}")

    print(json.dumps({"condition": args.condition, "round": args.round_index,
                      "strategy": strategy_id, "fitness": score.fitness,
                      "valid": f"{score.n_valid}/{score.n_envs}",
                      "learnable_fraction": score.learnable_fraction,
                      "backend_calls": backend.stats.calls}, indent=2))
    return 0


def cmd_gen(args) -> int:
    backend = make_backend(args.backend)
    topics = load_topics(args.topics)
    rng = random.Random(args.rng_seed)
    topic = args.topic or sample_topic(topics, args.skill, rng)
    design = generate(backend, args.designer_model,
                      Path(args.strategy).read_text(), args.skill, topic, None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.py").write_text(design.code)
    (out / "meta.json").write_text(json.dumps(
        {"concept": design.concept, "hint": design.hint, "skill": args.skill,
         "topic": topic, "t": _now()}, indent=2))
    gates = pre_eval_gates(design.code, design.hint)
    print(json.dumps({"concept": design.concept, "topic": topic,
                      "v1_battery": gates.v1_battery, "v2_hint": gates.v2_hint,
                      "issues": gates.issues, "out": str(out)}, indent=2))
    return 0 if (gates.v1_battery and gates.v2_hint) else 1


def cmd_eval_env(args) -> int:
    backend = make_backend(args.backend)
    code = Path(args.env).read_text()
    meta = json.loads(Path(args.meta).read_text()) if args.meta else {}
    hint = args.hint or meta.get("hint", "")
    ev = evaluate_env(code, hint, backend, args.solver_model, G=args.G,
                      seed0=args.rng_seed)
    gates = apply_band_gate(pre_eval_gates(code, hint), ev.mean_nohint)  # A2
    result = {"win_nohint": ev.win_nohint, "win_hint": ev.win_hint,
              "mean_nohint": ev.mean_nohint, "mean_hint": ev.mean_hint,
              "floored_regret": ev.floored_regret,
              "raw_regret": round(ev.mean_hint - ev.mean_nohint, 4),   # A2
              "valid_for_fitness": gates.valid_for_fitness,
              "issues": gates.issues,
              "episodes": [{"arm": a, "seed": s, "return": r.return_,
                            "steps": r.steps, "ended": r.ended}
                           for a, s, r in ev.episodes]}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        full = dict(result)
        full["transcripts"] = [{"arm": a, "seed": s, "transcript": r.transcript}
                               for a, s, r in ev.episodes]
        out.write_text(json.dumps(full, ensure_ascii=False, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def cmd_probe(args) -> int:
    """R1–R3 real-model probe: n S0 envs, gate + spread + headroom report."""
    backend = make_backend(args.backend)
    topics = load_topics(args.topics)
    rng = random.Random(args.rng_seed)
    out_dir = Path(args.out)
    records = []
    for i in range(args.n):
        skill = SKILLS[i % len(SKILLS)]
        rec = run_env_cycle(backend, args.designer_model, args.solver_model,
                            Path(args.strategy).read_text(), skill,
                            sample_topic(topics, skill, rng), None, args.G,
                            seed0=i * args.G, out_dir=out_dir, env_id=f"probe_e{i:02d}")
        _jsonl(out_dir / "envs.jsonl", rec)
        records.append(rec)
        print(f"[{i + 1}/{args.n}] {rec.get('concept', 'DESIGN-FAIL')[:60]} "
              f"valid={rec.get('valid_for_fitness')} "
              f"regret={rec.get('floored_regret')}")
    score = score_round(records)
    evaluated = [r for r in records if r.get("stage") == "evaluated"]
    regrets = [r["floored_regret"] for r in evaluated if r.get("valid_for_fitness")]
    spread = (max(regrets) - min(regrets)) if len(regrets) >= 2 else 0.0
    report = {
        "R1_interface": {"pass": score.n_valid >= max(2, args.n // 2),
                         "valid": f"{score.n_valid}/{args.n}"},
        "R2_spread": {"pass": spread >= 0.15, "spread": round(spread, 3)},
        "R3_headroom": {"pass": not (score.fitness >= 0.6 and score.learnable_fraction >= 0.5),
                        "note": "baseline must not already saturate",
                        "S0_fitness": score.fitness,
                        "learnable_fraction": score.learnable_fraction},
        "backend_calls": backend.stats.calls,
    }
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all(report[k]["pass"] for k in ("R1_interface", "R2_spread", "R3_headroom")) else 1


def cmd_smoke(args) -> int:
    """R4/R5 machinery smoke on the mock backend + fixture envs. No model calls."""
    from tests.fixtures import GOOD_ENV, BROKEN_ENV, solver_policy
    from .backends import MockBackend

    ok = True
    rep = env_api.run_battery(GOOD_ENV)
    print(f"R5 determinism+battery (good fixture): {'PASS' if rep.passed else 'FAIL ' + str(rep.issues)}")
    ok &= rep.passed
    rep_bad = env_api.run_battery(BROKEN_ENV)
    print(f"R4 invalid-with-feedback (broken fixture rejected): "
          f"{'PASS' if not rep_bad.passed and rep_bad.issues else 'FAIL'}")
    ok &= (not rep_bad.passed and bool(rep_bad.issues))
    backend = MockBackend(policy=solver_policy)
    ev = evaluate_env(GOOD_ENV, "Track which digits are confirmed and eliminate.",
                      backend, "mock-solver", G=2, seed0=0)
    print(f"paired eval on fixture: nohint={ev.mean_nohint:.2f} hint={ev.mean_hint:.2f} "
          f"regret={ev.floored_regret:.2f} calls={backend.stats.calls}")
    ok &= backend.stats.calls > 0
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="szg")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, designer=True):
        sp.add_argument("--backend", default="claude-cli")
        sp.add_argument("--solver-model", default="haiku")
        if designer:
            sp.add_argument("--designer-model", default="sonnet")
            sp.add_argument("--strategy", default="strategies/S0_baseline.md")
            sp.add_argument("--topics", default="seeds/topics.jsonl")
        sp.add_argument("--G", type=int, default=3)
        sp.add_argument("--rng-seed", type=int, default=0)

    sp = sub.add_parser("round", help="one designer round under a condition")
    common(sp)
    sp.add_argument("--condition", choices=["C0", "C1", "C2"], required=True)
    sp.add_argument("--round-index", type=int, required=True)
    sp.add_argument("--n-envs", type=int, default=6)
    sp.add_argument("--mutate", action="store_true",
                    help="C2: mutate the evaluated strategy after scoring")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_round)

    sp = sub.add_parser("gen", help="one designer generation + static gates")
    common(sp)
    sp.add_argument("--skill", choices=SKILLS, required=True)
    sp.add_argument("--topic")
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_gen)

    sp = sub.add_parser("eval-env", help="paired hint/no-hint evaluation of one env file")
    common(sp, designer=False)
    sp.add_argument("--env", required=True)
    sp.add_argument("--meta")
    sp.add_argument("--hint")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_eval_env)

    sp = sub.add_parser("probe", help="R1-R3 real-model probe on the S0 strategy")
    common(sp)
    sp.add_argument("--n", type=int, default=10)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_probe)

    sp = sub.add_parser("smoke", help="R4/R5 machinery smoke, mock backend only")
    sp.set_defaults(fn=cmd_smoke)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
