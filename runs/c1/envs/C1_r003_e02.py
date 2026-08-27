import random
import itertools


class LadderRungBazaarEnv:
    def __init__(self):
        self.rng = None
        self.heights = []
        self.prices = []
        self.tier_lo = []
        self.tier_hi = []
        self.budget = 0
        self.inspected = set()
        self.inspections_used = 0
        self.max_inspections = 2
        self.step_count = 0
        self.max_steps = 10
        self.done = False
        self.optimal_height = 0

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        n = 4
        self.heights = self.rng.sample(range(3, 10), n)
        self.tier_lo = []
        self.tier_hi = []
        self.prices = []
        for _ in range(n):
            lo = self.rng.randint(2, 7)
            hi = lo + 3
            price = self.rng.randint(lo, hi)
            self.tier_lo.append(lo)
            self.tier_hi.append(hi)
            self.prices.append(price)
        total = sum(self.prices)
        frac = self.rng.uniform(0.45, 0.65)
        budget = int(round(total * frac))
        budget = max(min(self.prices), min(budget, total - 1))
        self.budget = budget

        self.optimal_height = 0
        for r in range(n + 1):
            for combo in itertools.combinations(range(n), r):
                cost = sum(self.prices[i] for i in combo)
                if cost <= self.budget:
                    h = sum(self.heights[i] for i in combo)
                    if h > self.optimal_height:
                        self.optimal_height = h

        self.inspected = set()
        self.inspections_used = 0
        self.step_count = 0
        self.done = False

        lines = [
            "LADDER RUNG BAZAAR",
            f"You have a budget of ${self.budget}. Buy a subset of the 4 rungs below to build "
            "the tallest ladder you can WITHOUT exceeding budget.",
            "Each rung's height is known; its exact price is hidden behind a tier range.",
        ]
        for i in range(n):
            lines.append(
                f"  Rung {i+1}: height={self.heights[i]}, price tier ${self.tier_lo[i]}-${self.tier_hi[i]}"
            )
        lines.append(
            f"You may INSPECT at most {self.max_inspections} rungs to learn their exact price."
        )
        lines.append("Actions:")
        lines.append("  INSPECT <n>   — reveal the exact price of rung n (uses one of your inspections)")
        lines.append("  SUBMIT <list> — end the episode and buy the listed rungs, e.g. 'SUBMIT 1,3' or 'SUBMIT NONE'")
        lines.append(f"You have {self.max_steps} actions total (inspections + the final submit).")
        obs = "\n".join(lines)
        info = {"budget": self.budget, "heights": list(self.heights)}
        return obs, info

    def _rung_status(self):
        parts = []
        for i in range(4):
            if i in self.inspected:
                parts.append(f"Rung {i+1}: height={self.heights[i]}, price=${self.prices[i]} (known)")
            else:
                parts.append(
                    f"Rung {i+1}: height={self.heights[i]}, price tier ${self.tier_lo[i]}-${self.tier_hi[i]} (unknown)"
                )
        return "\n".join(parts)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip().upper()

        if text.startswith("INSPECT"):
            rest = text[len("INSPECT"):].strip()
            digits = "".join(ch for ch in rest if ch.isdigit())
            if not digits:
                obs = "Malformed action. Use 'INSPECT <n>' with n from 1 to 4.\n" + self._rung_status()
                return obs, 0.0, False, self.step_count >= self.max_steps, {}
            n = int(digits)
            if n < 1 or n > 4:
                obs = f"Rung {n} does not exist. Choose a rung from 1 to 4.\n" + self._rung_status()
                return obs, 0.0, False, self.step_count >= self.max_steps, {}
            idx = n - 1
            if self.inspections_used >= self.max_inspections and idx not in self.inspected:
                obs = (
                    f"No inspections remaining ({self.max_inspections} used). "
                    "You must decide using known info or SUBMIT.\n" + self._rung_status()
                )
                return obs, 0.0, False, self.step_count >= self.max_steps, {}
            if idx not in self.inspected:
                self.inspected.add(idx)
                self.inspections_used += 1
            remaining = self.max_inspections - self.inspections_used
            obs = (
                f"Rung {n} price revealed: ${self.prices[idx]}. Inspections remaining: {remaining}.\n"
                + self._rung_status()
            )
            truncated = self.step_count >= self.max_steps
            if truncated:
                self.done = True
                obs += "\nStep limit reached with no purchase made."
                return obs, 0.0, False, True, {}
            return obs, 0.0, False, False, {}

        if text.startswith("SUBMIT"):
            rest = text[len("SUBMIT"):].strip()
            if rest == "NONE" or rest == "":
                chosen = set()
            else:
                nums = re.findall(r"\d+", rest)
                if not nums:
                    obs = "Malformed SUBMIT. List rung numbers separated by commas/spaces, or 'SUBMIT NONE'.\n" + self._rung_status()
                    return obs, 0.0, False, self.step_count >= self.max_steps, {}
                chosen = set()
                invalid = False
                for tok in nums:
                    v = int(tok)
                    if v < 1 or v > 4:
                        invalid = True
                        continue
                    chosen.add(v - 1)
                if invalid:
                    obs = "One or more rung numbers are out of range (1-4). Try again.\n" + self._rung_status()
                    return obs, 0.0, False, self.step_count >= self.max_steps, {}

            cost = sum(self.prices[i] for i in chosen)
            height = sum(self.heights[i] for i in chosen)
            self.done = True

            if cost > self.budget:
                obs = (
                    f"Purchase FAILED: chosen rungs cost ${cost}, over your ${self.budget} budget. "
                    "The ladder could not be bought."
                )
                return obs, 0.0, True, False, {"feasible": False, "cost": cost, "height": height}

            reward = 0.2
            ratio = (height / self.optimal_height) if self.optimal_height > 0 else 1.0
            tier_note = "feasible only"
            if ratio >= 0.999:
                reward += 0.8
                tier_note = "optimal"
            elif ratio >= 0.8:
                reward += 0.5
                tier_note = "near-optimal"

            obs = (
                f"Purchase SUCCESSFUL: cost=${cost} (budget ${self.budget}), ladder height={height}. "
                f"Best possible height under this budget was {self.optimal_height} ({tier_note}). "
                f"Total reward: {reward:.2f}."
            )
            return obs, reward, True, False, {
                "feasible": True,
                "cost": cost,
                "height": height,
                "optimal_height": self.optimal_height,
            }

        obs = (
            "Unrecognized action. Use 'INSPECT <n>' or 'SUBMIT <rung numbers or NONE>'.\n"
            + self._rung_status()
        )
        truncated = self.step_count >= self.max_steps
        if truncated:
            self.done = True
            obs += "\nStep limit reached with no purchase made."
        return obs, 0.0, False, truncated, {}


import re
