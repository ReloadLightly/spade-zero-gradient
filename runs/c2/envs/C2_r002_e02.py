import random
import re


class GridDispatchEnv:
    SOURCES = ("solar", "wind", "gas")

    def __init__(self):
        self.rng = None
        self.base = {}
        self.slope = {}
        self.cap = {}
        self.demand = 0
        self.cost_optimal = 0.0
        self.cost_worst = 0.0
        self.steps = 0
        self.max_steps = 10
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.base = {s: round(self.rng.uniform(2.0, 8.0), 1) for s in self.SOURCES}
        self.slope = {s: round(self.rng.uniform(0.05, 0.30), 2) for s in self.SOURCES}
        self.cap = {s: self.rng.randint(20, 50) for s in self.SOURCES}
        total_cap = sum(self.cap.values())
        low = int(0.55 * total_cap)
        high = int(0.85 * total_cap)
        self.demand = self.rng.randint(low, high)

        self.cost_optimal, opt_alloc = self._economic_dispatch(self.demand)
        prop_alloc = {s: self.demand * self.cap[s] / total_cap for s in self.SOURCES}
        self.cost_worst = sum(self._cost(s, prop_alloc[s]) for s in self.SOURCES)
        if self.cost_worst <= self.cost_optimal:
            self.cost_worst = self.cost_optimal + 1.0

        self.steps = 0
        self.done = False

        obs = (
            "GRID DISPATCH: meet a demand of exactly "
            f"{self.demand} MW using three sources: solar, wind, gas.\n"
            "Each source has a hidden capacity and a hidden marginal cost "
            "(cost of its next MW) that rises steadily with its own output; "
            "both are fixed for the episode.\n"
            "Actions:\n"
            "  QUERY <source> <amount> - reveals the marginal cost ($/MW) "
            "at that output level (or the capacity ceiling if you overshoot it)\n"
            "  COMMIT solar=<x> wind=<y> gas=<z> - final allocation in MW; "
            "ends the episode\n"
            f"You have {self.max_steps} actions total, including COMMIT. "
            "Minimize total generation cost while meeting demand exactly."
        )
        return obs, {"demand": self.demand}

    def _cost(self, source, x):
        x = max(0.0, x)
        return self.base[source] * x + 0.5 * self.slope[source] * x * x

    def _marginal(self, source, x):
        return self.base[source] + self.slope[source] * x

    def _alloc_at_lambda(self, lam):
        alloc = {}
        for s in self.SOURCES:
            x = (lam - self.base[s]) / self.slope[s]
            x = min(max(x, 0.0), self.cap[s])
            alloc[s] = x
        return alloc

    def _economic_dispatch(self, demand):
        lo = min(self.base.values())
        hi = max(self.base[s] + self.slope[s] * self.cap[s] for s in self.SOURCES)
        for _ in range(80):
            mid = (lo + hi) / 2.0
            total = sum(self._alloc_at_lambda(mid).values())
            if total < demand:
                lo = mid
            else:
                hi = mid
        alloc = self._alloc_at_lambda(hi)
        cost = sum(self._cost(s, alloc[s]) for s in self.SOURCES)
        return cost, alloc

    def _maybe_truncate(self, obs, reward):
        if self.steps >= self.max_steps:
            self.done = True
            obs = obs + " Step limit reached; episode truncated without a commit."
            return obs, reward, False, True, {}
        return obs, reward, False, False, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()

        m = re.match(
            r"^QUERY\s+(solar|wind|gas)\s+(-?\d+(?:\.\d+)?)\s*$", action, re.IGNORECASE
        )
        if m:
            source = m.group(1).lower()
            amount = float(m.group(2))
            if amount < 0:
                obs = "Invalid: output amount cannot be negative. No query performed."
                return self._maybe_truncate(obs, 0.0)
            cap = self.cap[source]
            if amount > cap:
                mc = self._marginal(source, cap)
                obs = (
                    f"{source} cannot exceed {cap} MW capacity. "
                    f"Marginal cost at max output ({cap} MW): {mc:.2f} $/MW"
                )
            else:
                mc = self._marginal(source, amount)
                obs = f"Marginal cost of {source} at {amount:g} MW: {mc:.2f} $/MW"
            return self._maybe_truncate(obs, 0.0)

        m = re.match(
            r"^COMMIT\s+solar=(-?\d+(?:\.\d+)?)\s+wind=(-?\d+(?:\.\d+)?)\s+"
            r"gas=(-?\d+(?:\.\d+)?)\s*$",
            action,
            re.IGNORECASE,
        )
        if m:
            x = {
                "solar": float(m.group(1)),
                "wind": float(m.group(2)),
                "gas": float(m.group(3)),
            }
            self.done = True
            over_cap = [
                s for s in self.SOURCES if x[s] > self.cap[s] + 1e-6 or x[s] < -1e-6
            ]
            demand_ok = abs(sum(x.values()) - self.demand) <= 0.6
            feasible = demand_ok and not over_cap

            if not feasible:
                reasons = []
                if not demand_ok:
                    reasons.append(
                        f"total supplied {sum(x.values()):.1f} MW != demand "
                        f"{self.demand} MW"
                    )
                if over_cap:
                    reasons.append(f"exceeded capacity on: {', '.join(over_cap)}")
                obs = "COMMIT rejected as infeasible: " + "; ".join(reasons)
                return obs, 0.0, True, False, {"feasible": False}

            cost = sum(self._cost(s, x[s]) for s in self.SOURCES)
            efficiency = (self.cost_worst - cost) / (self.cost_worst - self.cost_optimal)
            efficiency = min(1.0, max(0.0, efficiency))
            reward = 0.3 + 0.7 * efficiency
            obs = (
                f"COMMIT accepted. Total cost: {cost:.2f}. "
                f"Optimal achievable cost: {self.cost_optimal:.2f}. "
                f"Efficiency score: {efficiency:.2f}."
            )
            return (
                obs,
                round(reward, 4),
                True,
                False,
                {"feasible": True, "cost": cost, "optimal_cost": self.cost_optimal},
            )

        obs = (
            "Invalid action. Use 'QUERY <source> <amount>' or "
            "'COMMIT solar=<x> wind=<y> gas=<z>'."
        )
        return self._maybe_truncate(obs, 0.0)
