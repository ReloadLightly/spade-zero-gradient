import random
import re


class GardenHoseEnv:
    def __init__(self):
        self.N = 14
        self.max_steps = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.M = self.rng.randint(4, 6)
        self.thirsty = sorted(self.rng.sample(range(self.N), self.M))
        self.R = self.rng.randint(1, 3)
        self.watered = set()
        self.steps = 0
        self.placements = 0
        self.terminated = False
        self.optimal = self._min_placements(self.thirsty, self.R)
        self.per_plot = 0.6 / self.M

        obs = (
            f"Garden row: plots numbered 0 to {self.N - 1}. "
            f"Thirsty plots needing water: {self.thirsty}. "
            "Your hose has a fixed water reach R around wherever it is planted "
            "(unknown to you): placing it at position p waters every thirsty "
            "plot with |plot - p| <= R and marks it watered. "
            "Goal: get ALL thirsty plots watered using as FEW placements as "
            "possible, within a total budget of 10 actions (every attempted "
            "action, valid or not, counts against the budget). "
            "Action format: 'place <position>' with an integer position from "
            f"0 to {self.N - 1}, e.g. 'place 7'. "
            "After each placement you will be told the wetted range, which "
            "lets you work out R yourself."
        )
        info = {"thirsty_positions": list(self.thirsty), "step_limit": self.max_steps}
        return obs, info

    def step(self, action):
        self.steps += 1
        remaining = self.max_steps - self.steps

        m = re.match(r"^\s*place\s+(-?\d+)\s*$", action, re.IGNORECASE)
        if not m:
            obs = (
                "Malformed action. Use exactly: place <position>, e.g. 'place 3'. "
                f"Steps remaining: {remaining}."
            )
            truncated = self.steps >= self.max_steps
            return obs, 0.0, False, truncated, {}

        p = int(m.group(1))
        if p < 0 or p >= self.N:
            obs = (
                f"Position {p} is out of range (0-{self.N - 1}). No hose moved. "
                f"Steps remaining: {remaining}."
            )
            truncated = self.steps >= self.max_steps
            return obs, 0.0, False, truncated, {}

        self.placements += 1
        lo, hi = p - self.R, p + self.R
        newly = sorted(t for t in self.thirsty if lo <= t <= hi and t not in self.watered)
        self.watered.update(newly)
        dry = sorted(set(self.thirsty) - self.watered)

        reward = self.per_plot * len(newly)
        terminated = False

        if not dry:
            efficiency = 0.4 * min(1.0, self.optimal / self.placements)
            reward += efficiency
            terminated = True
            obs = (
                f"Placed hose at {p}; wetted range [{lo}, {hi}]. "
                f"Newly watered: {newly if newly else 'none'}. "
                "All thirsty plots are now watered! "
                f"Used {self.placements} placement(s)."
            )
        else:
            obs = (
                f"Placed hose at {p}; wetted range [{lo}, {hi}]. "
                f"Newly watered: {newly if newly else 'none'}. "
                f"Still dry: {dry}. Steps remaining: {remaining}."
            )

        truncated = (not terminated) and (self.steps >= self.max_steps)
        info = {"watered": sorted(self.watered), "dry": dry}
        return obs, reward, terminated, truncated, info

    def _min_placements(self, points, R):
        # Greedy leftmost-point interval cover is provably optimal for
        # covering points with fixed-length windows (exchange argument).
        if not points:
            return 0
        pts = sorted(points)
        count = 0
        i = 0
        n = len(pts)
        while i < n:
            count += 1
            limit = pts[i] + 2 * R
            while i < n and pts[i] <= limit:
                i += 1
        return count
