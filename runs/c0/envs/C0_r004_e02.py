import random
import re


class SignalGreenSplitEnv:
    """Allocate a scarce green-time budget across 4 coordinated intersections
    to maximize weighted flow, given public priority weights but hidden demand."""

    N = 4
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.values = []
        self.demands = []
        self.budget = 0
        self.optimal_value = 0
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.values = [self.rng.randint(1, 5) for _ in range(self.N)]
        self.demands = [self.rng.randint(6, 24) for _ in range(self.N)]
        total_demand = sum(self.demands)
        shortfall = self.rng.randint(max(8, total_demand // 4), max(9, (total_demand * 45) // 100))
        self.budget = max(10, total_demand - shortfall)
        self.optimal_value = self._greedy_optimal()
        self.steps = 0
        self.done = False

        lines = [
            "AVENUE SIGNAL COORDINATION: 4 intersections (1-4) share one signal cycle.",
            f"You control a green-time budget of {self.budget} seconds to split across them this cycle.",
            "Each intersection has a known priority weight (traffic value per green-second served):",
        ]
        for i in range(self.N):
            lines.append(f"  Intersection {i + 1}: priority weight = {self.values[i]}")
        lines.append(
            "Each intersection also has a CURRENT QUEUE DEMAND (green-seconds needed to fully clear it) "
            "that is hidden until measured. Green given beyond an intersection's demand is wasted; green "
            "given below its demand loses value equal to the shortfall times that intersection's weight."
        )
        lines.append(
            "ACTIONS: 'PROBE <n>' (n=1..4) measures intersection n's exact current demand, costing a step. "
            "'ALLOCATE <g1> <g2> <g3> <g4>' commits four nonnegative integer green-seconds, one per "
            "intersection in order, summing to at most the budget, and ends the episode immediately."
        )
        lines.append(f"You have at most {self.MAX_STEPS} actions total. Maximize total weighted flow served.")
        return "\n".join(lines), {"budget": self.budget, "values": list(self.values)}

    def _greedy_optimal(self):
        order = sorted(range(self.N), key=lambda i: (-self.values[i], i))
        remaining = self.budget
        total = 0
        for i in order:
            g = min(self.demands[i], remaining)
            total += self.values[i] * g
            remaining -= g
        return total

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        probe_match = re.fullmatch(r"(?i)PROBE\s+([1-4])", action)
        allocate_match = re.fullmatch(r"(?i)ALLOCATE\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", action)

        if probe_match:
            idx = int(probe_match.group(1)) - 1
            obs = (
                f"Intersection {idx + 1} current demand: {self.demands[idx]} green-seconds "
                f"(priority weight {self.values[idx]})."
            )
            terminated, truncated = self._check_step_limit()
            return obs, 0.0, terminated, truncated, {}

        if allocate_match:
            g = [int(allocate_match.group(k)) for k in range(1, 5)]
            if any(x < 0 for x in g):
                obs = "Invalid allocation: green-seconds cannot be negative. Try again."
                terminated, truncated = self._check_step_limit()
                return obs, 0.0, terminated, truncated, {}
            total_g = sum(g)
            if total_g > self.budget:
                obs = (
                    f"Invalid allocation: total {total_g}s exceeds the {self.budget}s budget by "
                    f"{total_g - self.budget}s. Reduce and retry."
                )
                terminated, truncated = self._check_step_limit()
                return obs, 0.0, terminated, truncated, {}

            achieved = sum(self.values[i] * min(g[i], self.demands[i]) for i in range(self.N))
            reward = min(1.0, achieved / self.optimal_value) if self.optimal_value > 0 else 1.0
            self.done = True
            breakdown = ", ".join(
                f"I{i + 1}: served {min(g[i], self.demands[i])}/{self.demands[i]}s (w={self.values[i]})"
                for i in range(self.N)
            )
            obs = (
                f"Allocation committed. Weighted flow achieved: {achieved}/{self.optimal_value} "
                f"of optimal ({reward:.2f} of 1.0). Breakdown -> {breakdown}."
            )
            return obs, reward, True, False, {"achieved": achieved, "optimal": self.optimal_value}

        obs = (
            "Malformed action. Use 'PROBE <n>' with n in 1-4, or 'ALLOCATE <g1> <g2> <g3> <g4>' "
            "with four nonnegative integers."
        )
        terminated, truncated = self._check_step_limit()
        return obs, 0.0, terminated, truncated, {}

    def _check_step_limit(self):
        if self.steps >= self.MAX_STEPS:
            self.done = True
            return False, True
        return False, False
