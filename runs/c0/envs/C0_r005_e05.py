import random
import re


class GardenHoseEnv:
    """Water all known plants using the fewest hose repositions; the hose's
    reach is hidden and must be inferred from watering feedback."""

    MAX_STEPS = 10
    ACTION_RE = re.compile(r'^\s*PLACE\s+(-?\d+)\s*$', re.IGNORECASE)

    def __init__(self):
        self.rng = None
        self.L = 0
        self.K = 0
        self.R = 0
        self.plants = []
        self.watered = set()
        self.steps = 0
        self.moves_used = 0
        self.optimal_moves = 0
        self.last_place = None
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.L = 24
        self.K = self.rng.randint(5, 6)
        self.R = self.rng.randint(2, 4)
        self.plants = sorted(self.rng.sample(range(self.L), self.K))
        self.watered = set()
        self.steps = 0
        self.moves_used = 0
        self.last_place = None
        self.done = False
        self.optimal_moves = self._greedy_min_moves(self.R)

        obs = (
            f"GARDEN WATERING. The garden is a line of plots numbered 0 to {self.L - 1}. "
            f"Plants stand at plots: {self.plants}. Wherever you set the hose, it instantly "
            f"waters every plant within its reach on both sides — but the reach itself is "
            f"unknown to you and must be worked out from what gets wet.\n"
            f"GOAL: get every plant watered using as few hose placements as possible.\n"
            f"ACTION FORMAT: send exactly 'PLACE <plot number>', e.g. PLACE 9.\n"
            f"You have {self.MAX_STEPS} placements total. Go."
        )
        info = {"plants": list(self.plants), "max_steps": self.MAX_STEPS}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        m = self.ACTION_RE.match(action) if isinstance(action, str) else None
        valid_range = m is not None and 0 <= int(m.group(1)) <= self.L - 1

        if not valid_range:
            self.steps += 1
            truncated = self.steps >= self.MAX_STEPS
            if truncated:
                self.done = True
            obs = (
                f"Could not understand that action. Use exactly 'PLACE <plot number>' "
                f"with a plot between 0 and {self.L - 1}. "
                f"({self.MAX_STEPS - self.steps} placements left.)"
            )
            return obs, 0.0, False, truncated, {"valid": False}

        x = int(m.group(1))
        self.steps += 1
        self.moves_used += 1
        self.last_place = x

        newly_watered = sorted(
            p for p in self.plants
            if abs(p - x) <= self.R and p not in self.watered
        )
        self.watered.update(newly_watered)

        reward = 0.6 * (len(newly_watered) / self.K)

        remaining = sorted(p for p in self.plants if p not in self.watered)
        terminated = len(remaining) == 0

        if terminated:
            if self.moves_used == self.optimal_moves:
                reward += 0.4
            self.done = True

        truncated = (not terminated) and self.steps >= self.MAX_STEPS
        if truncated:
            self.done = True

        watered_str = "none" if not newly_watered else str(newly_watered)
        if remaining:
            nearest = min(remaining, key=lambda p: abs(p - x))
            nearest_info = f"Nearest still-dry plant: plot {nearest} (distance {abs(nearest - x)} from your placement)."
        else:
            nearest_info = "Every plant is watered."

        obs = (
            f"Placed hose at plot {x}. Newly watered this turn: {watered_str}. "
            f"Watered so far: {sorted(self.watered)}. Still dry: {remaining}. "
            f"{nearest_info} Placements used: {self.moves_used}/{self.MAX_STEPS}."
        )
        info = {
            "valid": True,
            "moves_used": self.moves_used,
            "watered_count": len(self.watered),
        }
        return obs, reward, terminated, truncated, info

    def _greedy_min_moves(self, reach):
        pts = self.plants
        i = 0
        n = len(pts)
        moves = 0
        while i < n:
            p = pts[i]
            limit = min(p + reach, self.L - 1)
            moves += 1
            j = i
            while j < n and pts[j] <= limit:
                j += 1
            i = j
        return moves
