import random
import re


class PortfolioOptEnv:
    def __init__(self):
        self.n = 6
        self.max_steps = 10
        self.rng = None
        self.costs = []
        self.values = []
        self.budget = 0
        self.optimal_value = 0
        self.best_fraction = 0.0
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.costs = [self.rng.randint(2, 9) for _ in range(self.n)]
        self.values = [self.rng.randint(3, 15) for _ in range(self.n)]
        total_cost = sum(self.costs)
        min_cost = min(self.costs)
        lo = min_cost * 2
        hi = total_cost - min_cost
        target = int(total_cost * 0.55)
        self.budget = max(lo, min(hi, target))
        self._compute_optimal()
        self.best_fraction = 0.0
        self.steps = 0
        self.done = False
        obs = self._render_intro()
        info = {"budget": self.budget, "costs": list(self.costs), "n": self.n}
        return obs, info

    def _compute_optimal(self):
        best_value = 0
        for mask in range(1, 1 << self.n):
            cost = 0
            val = 0
            for i in range(self.n):
                if mask & (1 << i):
                    cost += self.costs[i]
                    val += self.values[i]
                    if cost > self.budget:
                        break
            if cost <= self.budget and val > best_value:
                best_value = val
        self.optimal_value = best_value

    def _render_intro(self):
        lines = []
        lines.append(
            "PORTFOLIO OPTIMIZATION: Fund a subset of {} projects (numbered 1-{}) "
            "without exceeding the budget, maximizing total (hidden) value.".format(
                self.n, self.n
            )
        )
        lines.append("Budget: {}".format(self.budget))
        cost_str = ", ".join(
            "Project {}: cost {}".format(i + 1, c) for i, c in enumerate(self.costs)
        )
        lines.append("Costs (values are hidden): " + cost_str)
        lines.append(
            "ACTION FORMAT (exactly one per turn):\n"
            "  'PROBE i j'  - compare value-per-cost efficiency of project i vs j. "
            "No reward, reveals which is more efficient (or tie).\n"
            "  'TRY i,j,k'  - propose a portfolio (comma-separated project numbers) to fund. "
            "You are rewarded for any improvement over your best portfolio's fraction of the "
            "true optimal achievable value. Reaching the true optimum ends the episode."
        )
        lines.append("You have {} total actions (probes and tries combined).".format(self.max_steps))
        return "\n".join(lines)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        raw = (action or "").strip()
        text = raw.upper()

        probe_match = re.match(r'^PROBE\s+(\d+)[\s,]+(\d+)\s*$', text)
        try_match = re.match(r'^TRY\s+([\d,\s]+)$', text)

        if probe_match:
            obs, reward, terminated, truncated = self._do_probe(
                int(probe_match.group(1)), int(probe_match.group(2))
            )
        elif try_match:
            obs, reward, terminated, truncated = self._do_try(try_match.group(1))
        else:
            obs = (
                "Malformed action '{}'. Use 'PROBE i j' or 'TRY i,j,k' with project "
                "numbers between 1 and {}.".format(raw, self.n)
            )
            reward = 0.0
            terminated = False
            truncated = self.steps >= self.max_steps

        if terminated or truncated:
            self.done = True

        remaining = self.max_steps - self.steps
        if not (terminated or truncated):
            obs = obs + "\nSteps remaining: {}.".format(remaining)

        info = {"steps_used": self.steps, "best_fraction": self.best_fraction}
        return obs, reward, terminated, truncated, info

    def _do_probe(self, i, j):
        if not (1 <= i <= self.n) or not (1 <= j <= self.n) or i == j:
            return (
                "Invalid PROBE indices; choose two distinct project numbers between "
                "1 and {}.".format(self.n),
                0.0,
                False,
                self.steps >= self.max_steps,
            )
        lhs = self.values[i - 1] * self.costs[j - 1]
        rhs = self.values[j - 1] * self.costs[i - 1]
        if lhs > rhs:
            msg = "Project {} has HIGHER value-per-cost efficiency than Project {}.".format(i, j)
        elif lhs < rhs:
            msg = "Project {} has LOWER value-per-cost efficiency than Project {}.".format(i, j)
        else:
            msg = "Project {} and Project {} have EQUAL value-per-cost efficiency.".format(i, j)
        return msg, 0.0, False, self.steps >= self.max_steps

    def _do_try(self, index_str):
        parts = [p for p in re.split(r'[,\s]+', index_str.strip()) if p != ""]
        try:
            indices = [int(p) for p in parts]
        except ValueError:
            return (
                "Could not parse TRY indices; use comma-separated project numbers.",
                0.0,
                False,
                self.steps >= self.max_steps,
            )
        if not indices or len(set(indices)) != len(indices) or any(
            not (1 <= idx <= self.n) for idx in indices
        ):
            return (
                "Invalid TRY list; give distinct project numbers between 1 and {}, "
                "no duplicates.".format(self.n),
                0.0,
                False,
                self.steps >= self.max_steps,
            )

        cost_sum = sum(self.costs[idx - 1] for idx in indices)
        value_sum = sum(self.values[idx - 1] for idx in indices)

        if cost_sum > self.budget:
            return (
                "Portfolio {} costs {} which exceeds the budget of {}. Infeasible.".format(
                    sorted(indices), cost_sum, self.budget
                ),
                0.0,
                False,
                self.steps >= self.max_steps,
            )

        fraction = value_sum / self.optimal_value
        improvement = max(0.0, fraction - self.best_fraction)
        is_new_best = fraction > self.best_fraction
        self.best_fraction = max(self.best_fraction, fraction)
        success = value_sum == self.optimal_value

        obs = (
            "Portfolio {} costs {} (budget {}), total value {}. This is {:.0%} of the "
            "best value achievable within budget. {}".format(
                sorted(indices),
                cost_sum,
                self.budget,
                value_sum,
                fraction,
                "New best!" if is_new_best else "No improvement over your prior best.",
            )
        )
        if success:
            obs += " OPTIMAL PORTFOLIO FOUND. Episode complete."

        terminated = success
        truncated = (not terminated) and self.steps >= self.max_steps
        return obs, improvement, terminated, truncated
