import random


class FarmPlotAllocationEnv:
    """Allocate a fixed farm budget across three crops with hidden concave
    yield curves, discovered via test plantings, then commit one final split."""

    CROPS = ("wheat", "corn", "soy")
    BUDGET = 12
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.a = {c: self.rng.randint(6, 10) for c in self.CROPS}
        self.b = {c: self.rng.choice([1, 2, 3, 4]) / 10.0 for c in self.CROPS}
        self.steps = 0
        self.tested_crops = set()
        self.done = False
        self.optimal_yield = self._compute_optimal()

        obs = (
            "You manage a farm with a planting budget of "
            f"{self.BUDGET} plot-units to split across three crops: "
            f"{', '.join(self.CROPS)}. Each crop's total yield rises with "
            "more plot-units devoted to it, but the *rate* of increase "
            "changes as you add more units to that same crop.\n\n"
            "Two action forms:\n"
            "  TEST <crop> <amount>   -- see the yield if you devoted "
            "<amount> plot-units (0-12) to <crop> alone (does not spend "
            "your real budget, just a turn).\n"
            "  PLANT <wheat> <corn> <soy>  -- commit final integer "
            "plot-units to each crop (must sum to at most "
            f"{self.BUDGET}); this ends the episode and scores your yield.\n\n"
            f"You have {self.MAX_STEPS} turns total, including any TEST "
            "calls. Use TEST to learn each crop's curve, then PLANT once "
            "you have a strong allocation."
        )
        info = {"budget": self.BUDGET, "crops": self.CROPS}
        return obs, info

    def _yield_for(self, crop, amount):
        a, b = self.a[crop], self.b[crop]
        return a * amount - b * amount * amount

    def _compute_optimal(self):
        best = -1.0
        for w in range(self.BUDGET + 1):
            for c in range(self.BUDGET + 1 - w):
                s = self.BUDGET - w - c
                total = (
                    self._yield_for("wheat", w)
                    + self._yield_for("corn", c)
                    + self._yield_for("soy", s)
                )
                if total > best:
                    best = total
        return best

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        reward = 0.0
        terminated = False
        obs = None
        parts = action.strip().split()

        if len(parts) == 3 and parts[0].upper() == "TEST":
            crop = parts[1].lower()
            if crop not in self.CROPS:
                obs = (
                    f"Unrecognized crop '{parts[1]}'. Valid crops: "
                    f"{', '.join(self.CROPS)}."
                )
            else:
                try:
                    amt = int(parts[2])
                except ValueError:
                    amt = None
                if amt is None or amt < 0 or amt > self.BUDGET:
                    obs = (
                        f"Invalid amount '{parts[2]}'. Must be an integer "
                        f"between 0 and {self.BUDGET}."
                    )
                else:
                    y = self._yield_for(crop, amt)
                    self.tested_crops.add(crop)
                    obs = (
                        f"TEST {crop} {amt} -> yield {y:.2f} "
                        "(this was a trial planting; your real budget is "
                        "untouched)."
                    )
        elif len(parts) == 4 and parts[0].upper() == "PLANT":
            try:
                w, c, s = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                w = c = s = None

            if w is None:
                obs = "Invalid PLANT amounts; use three integers, e.g. PLANT 4 5 3."
            elif w < 0 or c < 0 or s < 0 or (w + c + s) > self.BUDGET:
                obs = (
                    f"Invalid allocation: wheat={w}, corn={c}, soy={s} "
                    f"sums to {w + c + s}, which exceeds your budget of "
                    f"{self.BUDGET} (or contains a negative value)."
                )
            else:
                total_yield = (
                    self._yield_for("wheat", w)
                    + self._yield_for("corn", c)
                    + self._yield_for("soy", s)
                )
                ratio = 0.0
                if self.optimal_yield > 0:
                    ratio = min(1.0, total_yield / self.optimal_yield)
                exploration_bonus = 0.2 if len(self.tested_crops) >= 2 else 0.0
                quality = 0.8 * ratio
                reward = exploration_bonus + quality
                terminated = True
                self.done = True
                obs = (
                    f"PLANT {w} {c} {s} -> total yield {total_yield:.2f} "
                    f"(best possible was {self.optimal_yield:.2f}, "
                    f"{ratio * 100:.1f}% of optimal). Harvest complete."
                )
        else:
            obs = (
                "Malformed action. Use 'TEST <crop> <amount>' or "
                "'PLANT <wheat> <corn> <soy>'."
            )

        self.steps += 1
        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            obs += " No PLANT was committed before the step limit; season over with 0 yield scored."

        info = {"steps": self.steps, "optimal_yield": self.optimal_yield}
        return obs, reward, terminated, truncated, info
