"""Acceptance tests pinning the szg package API (no model calls)."""

import json

import pytest

from szg import env_api
from szg.backends import MockBackend
from szg.designer import parse_designer_output, DesignerParseError
from szg.evolve import Strategy, StrategyArchive, mutate
from szg.fitness import score_round, diversity_proxy
from szg.gates import apply_band_gate, check_hint, pre_eval_gates
from szg.memory import EnvMemory
from szg.runner import evaluate_env, play_episode

from tests.fixtures import BROKEN_ENV, GOOD_ENV, NONDET_ENV, solver_policy


# ---------- env contract ----------

def test_validate_source_rejects_forbidden_import():
    issues = env_api.validate_source(BROKEN_ENV)
    assert any("forbidden import" in i for i in issues)


def test_battery_good_env_passes():
    rep = env_api.run_battery(GOOD_ENV)
    assert rep.passed, rep.issues


def test_battery_catches_nondeterminism():
    rep = env_api.run_battery(NONDET_ENV)
    assert not rep.passed
    assert any("nondeterministic" in i for i in rep.issues)


# ---------- designer output parsing ----------

SAMPLE_DESIGNER_OUTPUT = """CONCEPT: Cargo manifest deduction — cross-checking lies under a step budget
```python
import random

class CargoEnv:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.x = self.rng.randint(0, 3)
        self.steps_left = 4
        return "Which crate (0-3) holds the cargo? Ask 'probe N' or 'answer N'.", {}

    def step(self, action):
        self.steps_left -= 1
        a = str(action).strip().split()
        trunc = self.steps_left <= 0
        if len(a) == 2 and a[0] == "answer" and a[1].isdigit():
            if int(a[1]) == self.x:
                return "Correct.", 1.0, True, False, {}
            return "Wrong.", 0.0, True, False, {}
        if len(a) == 2 and a[0] == "probe" and a[1].isdigit():
            return ("warm" if abs(int(a[1]) - self.x) <= 1 else "cold"), 0.0, False, trunc, {}
        return "Format: 'probe N' or 'answer N'.", 0.0, False, trunc, {}
```
HINT: Probe the two middle crates first; the warm/cold pattern of those two answers
uniquely identifies the crate, so you never need a third probe before answering."""


def test_parse_designer_output_roundtrip():
    out = parse_designer_output(SAMPLE_DESIGNER_OUTPUT)
    assert out.concept.startswith("Cargo manifest")
    assert "class CargoEnv" in out.code
    assert out.hint.startswith("Probe the two middle")
    assert env_api.run_battery(out.code).passed


def test_parse_designer_output_requires_code_block():
    with pytest.raises(DesignerParseError):
        parse_designer_output("CONCEPT: x\nHINT: y")


# ---------- gates ----------

def test_hint_gate_rejects_action_injection():
    assert any("ACTION" in i for i in check_hint("Just do this: ACTION: answer 3"))
    assert check_hint("Reason about parity first, then eliminate halves.") == []


def test_band_gate_excludes_password_env():
    """A2 (2026-08-26): V3 bands on mean_nohint, not win_nohint."""
    rep = pre_eval_gates(GOOD_ENV, "A fine strategic hint about halving intervals.")
    rep = apply_band_gate(rep, mean_nohint=0.0)   # nothing without the hint
    assert not rep.valid_for_fitness
    rep2 = pre_eval_gates(GOOD_ENV, "A fine strategic hint about halving intervals.")
    rep2 = apply_band_gate(rep2, mean_nohint=0.5)
    assert rep2.valid_for_fitness


def test_band_gate_admits_partial_credit_env():
    """The case A2 exists for: real partial-credit progress, no full solve.

    Under the pre-A2 win-rate band this env was excluded (win_nohint 0.0)
    even though the solver plainly made cold progress (mean 0.4). Probe
    env e01 on 2026-08-25 was exactly this.
    """
    rep = pre_eval_gates(GOOD_ENV, "A fine strategic hint about halving intervals.")
    rep = apply_band_gate(rep, mean_nohint=0.4)
    assert rep.valid_for_fitness


def test_band_gate_still_excludes_trivial_env():
    rep = pre_eval_gates(GOOD_ENV, "A fine strategic hint about halving intervals.")
    rep = apply_band_gate(rep, mean_nohint=1.0)   # solved cold
    assert not rep.valid_for_fitness


# ---------- runner + regret ----------

def test_paired_eval_positive_regret_on_fixture():
    backend = MockBackend(policy=solver_policy)
    ev = evaluate_env(GOOD_ENV, "Halve the interval using the too-low/too-high feedback.",
                      backend, "mock", G=3, seed0=3)
    assert ev.mean_hint >= ev.mean_nohint
    assert ev.floored_regret == pytest.approx(ev.mean_hint - ev.mean_nohint)
    assert backend.stats.calls > 0


def test_play_episode_format_failure_path():
    backend = MockBackend(outputs=["no action here", "still none", "nope"])
    env_cls = env_api.load_env_class(GOOD_ENV)
    res = play_episode(env_cls, backend, "mock", seed=0)
    assert res.ended == "format_failure"
    assert res.return_ == 0.0


# ---------- fitness ----------

def _rec(regret, win, valid=True, code="a b c d e f g"):
    return {"floored_regret": regret, "win_nohint": win,
            "valid_for_fitness": valid, "code": code}


def test_score_round_math_and_invalid_round():
    score = score_round([_rec(0.4, 0.5), _rec(0.2, 0.3), _rec(0.9, 0.0, valid=False)])
    assert score.fitness == pytest.approx(0.3)
    assert score.n_valid == 2 and not score.invalid_round
    bad = score_round([_rec(0.9, 0.0, valid=False), _rec(0.8, 1.0, valid=False)])
    assert bad.invalid_round and bad.fitness == 0.0


def test_score_round_tolerates_pre_eval_rejected_env():
    """A1 regression (2026-08-26): a V1/V2-rejected env must not kill the round.

    cmd_probe writes ``win_nohint=None`` for an env rejected before
    evaluation. Before A1 the -1 sentinel in score_round never applied to a
    present-but-None key, so is_learnable() raised TypeError and the whole
    round lost its report after every model call had been paid for. This
    happened for real on probe env e08 (650-char hint, V2 rejection).
    """
    rejected = {"env_id": "e08", "stage": "gates", "valid_for_fitness": False,
                "gate_issues": ["V2: hint exceeds 600 chars"], "win_nohint": None,
                "code": "a b c d e f g"}
    score = score_round([_rec(0.4, 0.5), _rec(0.2, 0.3), rejected])
    assert score.n_envs == 3
    assert score.n_valid == 2 and not score.invalid_round
    assert score.fitness == pytest.approx(0.3)
    # the unevaluated env counts as not-learnable, never as learnable
    assert score.learnable_fraction == pytest.approx(2 / 3)
    json.dumps(score.per_env)


def test_score_round_tolerates_absent_win_nohint():
    """The missing-key path A1's sentinel was originally written for."""
    absent = {"env_id": "e99", "valid_for_fitness": False, "code": "a b c"}
    score = score_round([_rec(0.4, 0.5), _rec(0.2, 0.3), absent])
    assert score.n_valid == 2
    assert score.learnable_fraction == pytest.approx(2 / 3)


def test_diversity_proxy_bounds():
    assert 0.0 <= diversity_proxy(["a b c d", "a b c d"]) <= 1.0


# ---------- memory ----------

def test_memory_digest_sections(tmp_path):
    mem = EnvMemory(tmp_path / "m.jsonl")
    mem.add({"env_id": "e1", "skill": "optimization", "concept": "route planning",
             "win_nohint": 0.5, "win_hint": 0.9, "floored_regret": 0.4,
             "valid_for_fitness": True})
    mem.add({"env_id": "e2", "skill": "logical_deduction", "concept": "trivial riddle",
             "win_nohint": 1.0, "win_hint": 1.0, "floored_regret": 0.0,
             "valid_for_fitness": True})
    d = mem.digest()
    assert "route planning" in d and "Too easy" in d and "MEMORY" in d


# ---------- evolution ----------

def test_strategy_archive_persist_and_mutate(tmp_path):
    arch = StrategyArchive(tmp_path / "s.jsonl")
    arch.add(Strategy("S0", None, "Design multi-turn deduction games.", []))
    arch.record_fitness("S0", 0.25)
    backend = MockBackend(outputs=["STRATEGY S1:\nDesign harder state-gated games "
                                   "with partial-reward ladders."])
    child = mutate(backend, "mock", arch.strategies["S0"], "fitness 0.25, too easy", "S1")
    arch.add(child)
    arch2 = StrategyArchive(tmp_path / "s.jsonl")   # reload from disk
    assert set(arch2.strategies) == {"S0", "S1"}
    assert arch2.strategies["S1"].parent == "S0"
    assert arch2.strategies["S1"].text.startswith("Design harder")
    assert arch2.best().id == "S0"                   # only S0 has measured fitness


def test_render_feedback_tolerates_pre_eval_rejected_env():
    """Regression: the A1 defect also lived in evolve.render_feedback.

    It killed C2 round 0 of the main grid on 2026-08-26 at the mutate step,
    AFTER the round had been scored and written -- so the round data survived
    but the strategy never evolved. A rejected env carries win_nohint /
    win_hint / floored_regret as None, and ``rec.get(key, 0):.2f`` formats
    None rather than the default.
    """
    from szg.evolve import render_feedback
    rejected = {"env_id": "e03", "skill": "optimization", "concept": "broken env",
                "stage": "gates", "valid_for_fitness": False,
                "win_nohint": None, "win_hint": None, "floored_regret": None,
                "gate_issues": ["V2: hint exceeds 600 chars"]}
    score = score_round([_rec(0.4, 0.5), _rec(0.2, 0.3), rejected])
    text = render_feedback(score, [_rec(0.4, 0.5), rejected])
    assert "no-hint 0.00" in text
    assert "V2: hint exceeds 600 chars" in text


def test_round_log_is_json_serializable(tmp_path):
    score = score_round([_rec(0.4, 0.5)])
    json.dumps(score.per_env)
