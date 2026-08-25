import random
import re


class ScoutedKnapsackEnv:
    def __init__(self):
        self.rng = None
        self.n = 7
        self.costs = []
        self.values = []
        self.scouted = set()
        self.budget = 0
        self.optimal_value = 0
        self.steps = 0
        self.max_steps = 10
        self.done = False
        self._scout_re = re.compile(r'^\s*SCOUT\s+(\d+)\s*$', re.IGNORECASE)
        self._select_re = re.compile(r'^\s*SELECT\s+([0-9,\s]+)\s*$', re.IGNORECASE)

    def _knapsack_optimal(self):
        dp = [0] * (self.budget + 1)
        for cost, val in zip(self.costs, self.values):
            if cost > self.budget:
                continue
            for b in range(self.budget, cost - 1, -1):
                cand = dp[b - cost] + val
                if cand > dp[b]:
                    dp[b] = cand
        return dp[self.budget]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.n = 7
        self.costs = [self.rng.randint(15, 45) for _ in range(self.n)]
        self.values = [
            max(5, int(c * self.rng.uniform(0.5, 2.2))) for c in self.costs
        ]
        total_cost = sum(self.costs)
        self.budget = int(total_cost * self.rng.uniform(0.35, 0.5))
        self.budget = max(self.budget, min(self.costs))
        self.optimal_value = self._knapsack_optimal()
        self.scouted = set()
        self.steps = 0
        self.done = False

        lines = [
            "PROJECT PORTFOLIO SELECTION",
            f"You have a funding budget of {self.budget} to spend on projects.",
            "Each project has a KNOWN cost but a HIDDEN true value you must "
            "discover before it counts toward your decision.",
            "",
            "Projects (id: cost, value):",
        ]
        for i in range(self.n):
            lines.append(f"  {i + 1}: cost={self.costs[i]}, value=?")
        lines.append("")
        lines.append(
            "Actions: 'SCOUT <id>' reveals a project's true value (costs one "
            "turn). 'SELECT <id,id,...>' locks in a final funded subset "
            "(ends the episode)."
        )
        lines.append(
            f"You have {self.max_steps} total turns for scouting and the "
            "final selection combined. Selections exceeding the budget fail."
        )
        return "\n".join(lines), {}

    def _project_list_str(self):
        parts = []
        for i in range(self.n):
            if i in self.scouted:
                parts.append(f"{i + 1}: cost={self.costs[i]}, value={self.values[i]}")
            else:
                parts.append(f"{i + 1}: cost={self.costs[i]}, value=?")
        return "\n".join(parts)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()

        m_scout = self._scout_re.match(action)
        m_select = self._select_re.match(action)

        if m_scout:
            pid = int(m_scout.group(1))
            if pid < 1 or pid > self.n:
                obs = (
                    f"Invalid project id {pid}. Valid ids are 1..{self.n}.\n"
                    + self._project_list_str()
                )
                return self._maybe_truncate(obs, 0.0)
            idx = pid - 1
            self.scouted.add(idx)
            obs = (
                f"Scouted project {pid}: cost={self.costs[idx]}, "
                f"value={self.values[idx]}.\n" + self._project_list_str()
            )
            return self._maybe_truncate(obs, 0.0)

        if m_select:
            raw_ids = [s for s in re.split(r'[,\s]+', m_select.group(1)) if s]
            try:
                ids = [int(s) for s in raw_ids]
            except ValueError:
                obs = "Malformed SELECT list. Use comma-separated integer ids."
                return self._maybe_truncate(obs, 0.0)

            if len(ids) == 0 or len(set(ids)) != len(ids):
                obs = "SELECT must list distinct project ids at least once."
                return self._maybe_truncate(obs, 0.0)

            if any(pid < 1 or pid > self.n for pid in ids):
                obs = f"SELECT contains an id outside 1..{self.n}."
                return self._maybe_truncate(obs, 0.0)

            idxs = [pid - 1 for pid in ids]
            total_cost = sum(self.costs[i] for i in idxs)
            self.done = True

            if total_cost > self.budget:
                obs = (
                    f"Selection cost {total_cost} exceeds budget {self.budget}. "
                    "Funding rejected."
                )
                return obs, 0.0, True, False, {
                    "total_cost": total_cost,
                    "budget": self.budget,
                    "success": False,
                }

            achieved_value = sum(self.values[i] for i in idxs)
            ratio = 0.0 if self.optimal_value == 0 else (
                min(1.0, achieved_value / self.optimal_value)
            )
            reward = 0.2 + 0.8 * ratio
            obs = (
                f"Selection accepted: cost={total_cost}/{self.budget}, "
                f"achieved value={achieved_value}, optimal possible="
                f"{self.optimal_value}. Reward={reward:.3f}."
            )
            return obs, reward, True, False, {
                "total_cost": total_cost,
                "achieved_value": achieved_value,
                "optimal_value": self.optimal_value,
                "success": True,
            }

        obs = (
            "Malformed action. Use 'SCOUT <id>' or "
            "'SELECT <id,id,...>'.\n" + self._project_list_str()
        )
        return self._maybe_truncate(obs, 0.0)

    def _maybe_truncate(self, obs, reward):
        if self.steps >= self.max_steps:
            self.done = True
            obs = obs + f"\nStep limit ({self.max_steps}) reached with no selection."
            return obs, reward, False, True, {"success": False}
        return obs, reward, False, False, {}
