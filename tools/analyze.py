"""S4 analysis — registered endpoints plus the descriptives A2 requires.

Read-only over runs/. Implements PREREG §5 exactly:

  Primary   difference in mean fitness over the FINAL THIRD of rounds,
            C2−C0 and C1−C0, with bootstrap 95% CIs over ENVIRONMENTS
            (10,000 resamples). Slope over rounds as supporting evidence.
  Secondary held-out best-evolved-vs-S0 (requires a separate held-out run;
            reported as PENDING until runs/heldout/ exists).

and adds the sensitivity analysis without which a null is uninterpretable:
the minimum effect this design could have detected.

    python3 tools/analyze.py            # human-readable report
    python3 tools/analyze.py --json     # machine-readable
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CONDITIONS = ("c0", "c1", "c2")
COND_NAME = {"c0": "C0 static", "c1": "C1 memory", "c2": "C2 memory+evolution"}
BAND_LO, BAND_HI = 0.05, 0.95
LEARN_LO, LEARN_HI = 0.20, 0.80
N_BOOT = 10_000
BOOT_SEED = 20260827


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def final_third(rounds: list[dict]) -> list[int]:
    """Round indices making up the final third (PREREG §5)."""
    if not rounds:
        return []
    idx = sorted(r["round_index"] for r in rounds)
    k = max(1, len(idx) // 3)
    return idx[-k:]


def valid_envs(envs: list[dict], round_idx: list[int] | None = None) -> list[dict]:
    out = [e for e in envs if e.get("valid_for_fitness")]
    if round_idx is not None:
        out = [e for e in out if e.get("round_index") in round_idx]
    return out


def boot_diff(a: list[float], b: list[float], n=N_BOOT, seed=BOOT_SEED):
    """Bootstrap over environments: resample each arm with replacement."""
    if not a or not b:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(st.mean(ra) - st.mean(rb))
    diffs.sort()
    lo = diffs[int(0.025 * n)]
    hi = diffs[int(0.975 * n)]
    return {"point": st.mean(a) - st.mean(b), "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": (lo > 0 or hi < 0), "se": st.stdev(diffs)}


def slope(rounds: list[dict]) -> float | None:
    pts = [(r["round_index"], r["fitness"]) for r in rounds]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def analyse() -> dict:
    rounds = {c: _jsonl(RUNS / c / "rounds.jsonl") for c in CONDITIONS}
    envs = {c: _jsonl(RUNS / c / "envs.jsonl") for c in CONDITIONS}
    out: dict = {"conditions": {}, "primary": {}, "descriptives": {}, "sensitivity": {}}

    for c in CONDITIONS:
        f = [r["fitness"] for r in rounds[c]]
        ft = final_third(rounds[c])
        out["conditions"][c] = {
            "name": COND_NAME[c],
            "n_rounds": len(f),
            "fitness_by_round": f,
            "mean": st.mean(f) if f else None,
            "sd": st.stdev(f) if len(f) > 1 else None,
            "slope": slope(rounds[c]),
            "final_third_rounds": ft,
            "strategies_used": [r.get("strategy_id") for r in rounds[c]],
            "n_envs_logged": len(envs[c]),
            "n_envs_evaluated": sum(1 for e in envs[c] if e.get("stage") == "evaluated"),
        }

    # ---- GUARD: the endpoint is only meaningful on a balanced grid ----
    # final_third() takes k = max(1, n_rounds//3), so a condition that is one
    # round short gets a ONE-round window while the others get two. On
    # 2026-08-28, with C2 at 5 rounds and C0/C1 at 6, that compared C2's best
    # round against C0's two worst and produced C2-C0 = +0.0871, CI excluding
    # zero, classified DESIGN_IMPROVES. Entirely an artifact of the unequal
    # window. Refuse to report the primary endpoint until the grid is balanced.
    n_by_cond = {c: len(rounds[c]) for c in CONDITIONS}
    balanced = len(set(n_by_cond.values())) == 1 and min(n_by_cond.values()) > 0
    out["grid_balanced"] = balanced
    out["rounds_by_condition"] = n_by_cond
    if not balanced:
        out["primary"] = {}
        out["registered_outcome"] = (
            "NOT COMPUTED — grid unbalanced %s; the final-third window would "
            "differ in width between conditions" % n_by_cond)
        out["sensitivity"] = {"note": "withheld until the grid is balanced"}

    # ---- PRIMARY: final-third env-level regrets, bootstrap over environments ----
    ft_regret = {}
    for c in CONDITIONS:
        ft = final_third(rounds[c])
        ft_regret[c] = [e["floored_regret"] for e in valid_envs(envs[c], ft)]
        out["conditions"][c]["final_third_valid_envs"] = len(ft_regret[c])
    if balanced:
        for label, (a, b) in (("C2-C0", ("c2", "c0")), ("C1-C0", ("c1", "c0")),
                              ("C2-C1", ("c2", "c1"))):
            out["primary"][label] = boot_diff(ft_regret[a], ft_regret[b])

    # ---- SENSITIVITY: what could this design have detected? ----
    if not balanced:
        return out
    all_round_f = [r["fitness"] for c in CONDITIONS for r in rounds[c]]
    c0f = [r["fitness"] for r in rounds["c0"]]
    se = out["primary"].get("C2-C0", {}).get("se") if out["primary"].get("C2-C0") else None
    out["sensitivity"] = {
        "control_round_sd": st.stdev(c0f) if len(c0f) > 1 else None,
        "control_round_range": (max(c0f) - min(c0f)) if c0f else None,
        "pooled_round_sd": st.stdev(all_round_f) if len(all_round_f) > 1 else None,
        "bootstrap_se_C2_C0": se,
        "ci_half_width_C2_C0": (1.96 * se) if se else None,
        "min_detectable_effect_80pct": (2.8 * se) if se else None,
        "observed_grand_mean_fitness": st.mean(all_round_f) if all_round_f else None,
    }

    # ---- DESCRIPTIVES ----
    ev = [e for c in CONDITIONS for e in envs[c] if e.get("stage") == "evaluated"]
    raw = [e["raw_regret"] for e in ev if e.get("raw_regret") is not None]
    out["descriptives"]["hint_effect"] = {
        "n": len(raw),
        "helped": sum(1 for x in raw if x > 0),
        "harmed": sum(1 for x in raw if x < 0),
        "neutral": sum(1 for x in raw if x == 0),
        "mean_raw_regret": st.mean(raw) if raw else None,
        "median_raw_regret": st.median(raw) if raw else None,
        "min": min(raw) if raw else None,
        "max": max(raw) if raw else None,
        "harm_mass_hidden_by_floor": sum(x for x in raw if x < 0),
    }
    # learnable fraction under BOTH band definitions (A2 requires both)
    lf = {}
    for c in CONDITIONS:
        e = [x for x in envs[c] if x.get("stage") == "evaluated"]
        if not e:
            continue
        lf[c] = {
            "learnable_win_band": sum(1 for x in e if LEARN_LO <= x["win_nohint"] <= LEARN_HI) / len(e),
            "valid_under_A2_mean_band": sum(1 for x in e if BAND_LO <= x["mean_nohint"] <= BAND_HI) / len(e),
            "valid_under_pre_A2_win_band": sum(1 for x in e if BAND_LO <= x["win_nohint"] <= BAND_HI) / len(e),
        }
    out["descriptives"]["bands"] = lf
    out["descriptives"]["diversity_by_round"] = {
        c: [r.get("diversity") for r in rounds[c]] for c in CONDITIONS}
    out["descriptives"]["difficulty_trend_mean_nohint"] = {
        c: [round(st.mean([x["mean_nohint"] for x in envs[c]
                           if x.get("stage") == "evaluated" and x.get("round_index") == r["round_index"]] or [0]), 4)
            for r in rounds[c]] for c in CONDITIONS}
    # env loss census
    census = {}
    for c in CONDITIONS:
        cc = {"evaluated": 0, "designer_timeout": 0, "designer_backend_error": 0,
              "V2_hint_too_long": 0, "other_gate": 0}
        for e in envs[c]:
            iss = (e.get("gate_issues") or [""])[0]
            if e.get("stage") == "evaluated":
                cc["evaluated"] += 1
            elif e.get("stage") == "design":
                # two distinct failures, previously conflated: a true timeout
                # (generation exceeded timeout_s, seen only under C2/S3) vs a
                # backend refusal rc=1 (the usage-limit outage, hit all three)
                cc["designer_timeout" if "timed out" in iss
                   else "designer_backend_error"] += 1
            elif "hint exceeds" in iss:
                cc["V2_hint_too_long"] += 1
            else:
                cc["other_gate"] += 1
        census[c] = cc
    out["descriptives"]["env_census"] = census

    # ---- C2 evolution realised ----
    arch = _jsonl(RUNS / "c2" / "strategies.jsonl")
    used = [r.get("strategy_id") for r in rounds["c2"]]
    out["descriptives"]["c2_evolution"] = {
        "archive": [{"id": a["id"], "parent": a.get("parent"),
                     "fitnesses": a.get("fitnesses", [])} for a in arch],
        "strategies_used_per_round": used,
        "rounds_running_an_evolved_strategy": sum(1 for s in used if s != "S0"),
        "rounds_total": len(used),
    }

    # ---- registered outcome (PREREG §6) ----
    p = out["primary"]
    c2c0, c1c0, c2c1 = p.get("C2-C0"), p.get("C1-C0"), p.get("C2-C1")
    heldout = (RUNS / "heldout").exists()
    if c2c0 and c1c0:
        if c2c0["excludes_zero"] and c2c0["point"] > 0:
            outcome = "DESIGN_IMPROVES (pending held-out confirmation)" if not heldout else "DESIGN_IMPROVES"
        elif c1c0["excludes_zero"] and c1c0["point"] > 0 and c2c1 and not c2c1["excludes_zero"]:
            outcome = "MEMORY_ALONE_SUFFICES"
        elif not c2c0["excludes_zero"] and not c1c0["excludes_zero"]:
            outcome = "NO_TRACTION"
        else:
            outcome = "UNCLASSIFIED — see CIs"
    else:
        outcome = "INCOMPLETE — grid still running"
    out["registered_outcome"] = outcome
    out["heldout_run_present"] = heldout
    return out


def report(a: dict) -> str:
    L = []
    L.append("=" * 72)
    L.append("S4 ANALYSIS — spade-zero-gradient (PREREG §5 endpoints)")
    L.append("=" * 72)
    L.append("")
    L.append("CONDITIONS")
    for c in CONDITIONS:
        d = a["conditions"][c]
        L.append(f"  {d['name']:<22} rounds={d['n_rounds']} "
                 f"mean={d['mean']:.4f} sd={(d['sd'] or 0):.4f} "
                 f"slope={(d['slope'] if d['slope'] is not None else float('nan')):+.4f}"
                 if d["mean"] is not None else f"  {d['name']}: no rounds")
        L.append(f"      per-round: {[round(x,4) for x in d['fitness_by_round']]}")
        L.append(f"      strategies: {d['strategies_used']}")
        L.append(f"      final third: rounds {d['final_third_rounds']} "
                 f"({d['final_third_valid_envs']} valid envs)")
    L.append("")
    if not a.get("grid_balanced", True):
        L.append("PRIMARY ENDPOINT — WITHHELD")
        L.append(f"  grid is unbalanced: rounds per condition {a['rounds_by_condition']}")
        L.append("  the final-third window would be narrower for the short condition,")
        L.append("  which biases the comparison. Endpoint reported only on a balanced grid.")
        L.append("")
        L.append(f"  {a['registered_outcome']}")
        L.append("=" * 72)
        return "\n".join(L)
    L.append("PRIMARY ENDPOINT — final-third mean fitness difference")
    L.append("  bootstrap 95% CI over environments, 10,000 resamples")
    for k, v in a["primary"].items():
        if not v:
            L.append(f"  {k}: insufficient data")
            continue
        star = "EXCLUDES 0" if v["excludes_zero"] else "includes 0"
        L.append(f"  {k:<7} {v['point']:+.4f}  95% CI [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]  {star}")
    L.append("")
    L.append(f"  REGISTERED OUTCOME: {a['registered_outcome']}")
    L.append("")
    s = a["sensitivity"]
    L.append("SENSITIVITY — what this design could have detected")
    if s.get("bootstrap_se_C2_C0"):
        L.append(f"  control (C0) round-level sd     {s['control_round_sd']:.4f}"
                 f"   range {s['control_round_range']:.4f}")
        L.append(f"  bootstrap SE of C2-C0           {s['bootstrap_se_C2_C0']:.4f}")
        L.append(f"  95% CI half-width               {s['ci_half_width_C2_C0']:.4f}")
        L.append(f"  min detectable effect (80% pwr) {s['min_detectable_effect_80pct']:.4f}")
        L.append(f"  observed grand mean fitness     {s['observed_grand_mean_fitness']:.4f}")
        mde = s["min_detectable_effect_80pct"]
        gm = s["observed_grand_mean_fitness"]
        if mde and gm:
            L.append(f"  -> the smallest detectable effect is {mde/gm:.1f}x the grand mean fitness")
    L.append("")
    h = a["descriptives"]["hint_effect"]
    L.append("DESCRIPTIVE — does the hint help? (A2 raw vs floored regret)")
    L.append(f"  n={h['n']}  helped={h['helped']}  harmed={h['harmed']}  neutral={h['neutral']}")
    L.append(f"  mean raw {h['mean_raw_regret']:+.4f}  median {h['median_raw_regret']:+.4f}"
             f"  range [{h['min']:+.4f}, {h['max']:+.4f}]")
    L.append(f"  total harm mass floored away: {h['harm_mass_hidden_by_floor']:+.4f}")
    L.append("")
    e = a["descriptives"]["c2_evolution"]
    L.append("C2 EVOLUTION — realised, not intended")
    L.append(f"  rounds running an evolved strategy: "
             f"{e['rounds_running_an_evolved_strategy']}/{e['rounds_total']}")
    for s_ in e["archive"]:
        L.append(f"    {s_['id']:<3} parent={str(s_['parent']):<4} fitnesses={s_['fitnesses']}")
    L.append("")
    L.append("ENV CENSUS (loss is directional — see runs/_findings/)")
    for c in CONDITIONS:
        cc = a["descriptives"]["env_census"][c]
        n = sum(cc.values())
        L.append(f"  {COND_NAME[c]:<22} evaluated {cc['evaluated']}/{n} "
                 f"| designer-timeout {cc['designer_timeout']} "
                 f"| backend-rc1 {cc['designer_backend_error']} "
                 f"| V2 {cc['V2_hint_too_long']} | other {cc['other_gate']}")
    L.append("")
    L.append("BANDS — validity under both V3 definitions (A2)")
    for c, d in a["descriptives"]["bands"].items():
        L.append(f"  {COND_NAME[c]:<22} A2 mean-band {d['valid_under_A2_mean_band']:.2f} "
                 f"| pre-A2 win-band {d['valid_under_pre_A2_win_band']:.2f} "
                 f"| learnable {d['learnable_win_band']:.2f}")
    L.append("")
    if not a["heldout_run_present"]:
        L.append("SECONDARY ENDPOINT: PENDING — held-out best-vs-S0 run not yet performed")
        L.append("  (requires a fresh topic/seed grid never used in evolution)")
    L.append("=" * 72)
    return "\n".join(L)


if __name__ == "__main__":
    a = analyse()
    if "--json" in sys.argv:
        print(json.dumps(a, indent=2))
    else:
        print(report(a))
