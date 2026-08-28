import random


class SignalAvenueEnv:
    NUM_INTERSECTIONS = 3
    CYCLE_LENGTH = 80
    LOST_TIME = 20
    MIN_GREEN = 5
    MAX_STEPS = 10
    LABELS = ["A", "B", "C"]

    def __init__(self):
        self.rng = None
        self.s = []
        self.d = []
        self.g_total = self.CYCLE_LENGTH - self.LOST_TIME
        self.optimal_served = 0
        self.best_fraction = 0.0
        self.thresholds = [(0.7, 0.3), (0.9, 0.3), (1.0, 0.4)]
        self.given = [False, False, False]
        self.steps = 0
        self.last_allocation = None

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        n = self.NUM_INTERSECTIONS
        self.s = [self.rng.randint(1, 3) for _ in range(n)]
        self.d = [self.rng.randint(15, 45) for _ in range(n)]
        self.g_total = self.CYCLE_LENGTH - self.LOST_TIME
        self.optimal_served = self._compute_optimal()
        self.given = [False, False, False]
        self.best_fraction = 0.0
        self.steps = 0
        self.last_allocation = None

        s_desc = ", ".join(
            f"{self.LABELS[i]}(sat.flow={self.s[i]} veh/green-sec)"
            for i in range(n)
        )
        obs = (
            "AVENUE SIGNAL RETIMING. A cycle repeats every "
            f"{self.CYCLE_LENGTH}s along a {n}-intersection avenue; "
            f"{self.LOST_TIME}s per cycle is lost to amber/all-red clearance, "
            f"leaving {self.g_total}s of green to split across the {n} approaches. "
            f"Each approach needs at least {self.MIN_GREEN}s.\n"
            f"Known saturation flows: {s_desc}.\n"
            "Arrivals per cycle (demand) at each approach are UNKNOWN — discover "
            "them by trying allocations and reading the feedback.\n"
            "Goal: split the green time to maximize total vehicles served per "
            "cycle (vehicles served at an approach = min(demand, "
            "sat.flow * green_seconds)).\n"
            f"ACTION FORMAT: 'ALLOCATE gA,gB,gC' with {n} non-negative integers "
            f"(each >= {self.MIN_GREEN}) summing exactly to {self.g_total}, e.g. "
            f"'ALLOCATE {self.g_total - 2 * self.MIN_GREEN},{self.MIN_GREEN},"
            f"{self.MIN_GREEN}'.\n"
            f"You have {self.MAX_STEPS} steps total."
        )
        info = {}
        return obs, info

    def _compute_optimal(self):
        n = self.NUM_INTERSECTIONS
        best = 0
        lo, hi = self.MIN_GREEN, self.g_total - (n - 1) * self.MIN_GREEN
        for ga in range(lo, hi + 1):
            rem_b = self.g_total - ga
            lo_b = self.MIN_GREEN
            hi_b = rem_b - self.MIN_GREEN
            if hi_b < lo_b:
                continue
            for gb in range(lo_b, hi_b + 1):
                gc = self.g_total - ga - gb
                if gc < self.MIN_GREEN:
                    continue
                served = (
                    min(self.d[0], self.s[0] * ga)
                    + min(self.d[1], self.s[1] * gb)
                    + min(self.d[2], self.s[2] * gc)
                )
                if served > best:
                    best = served
        return best

    def _parse(self, action):
        text = action.strip()
        upper = text.upper()
        if upper.startswith("ALLOCATE"):
            text = text[len("ALLOCATE"):].strip()
        parts = [p.strip() for p in text.replace(" ", "").split(",")]
        if len(parts) != self.NUM_INTERSECTIONS:
            return None
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            return None
        return vals

    def step(self, action):
        self.steps += 1
        n = self.NUM_INTERSECTIONS
        vals = self._parse(action)

        if vals is None:
            obs = (
                f"MALFORMED ACTION. Use 'ALLOCATE gA,gB,gC' with {n} integers "
                f"summing to {self.g_total}, each >= {self.MIN_GREEN}."
            )
            terminated = False
            truncated = self.steps >= self.MAX_STEPS
            return obs, 0.0, terminated, truncated, {}

        if sum(vals) != self.g_total or any(v < self.MIN_GREEN for v in vals):
            obs = (
                f"INVALID ALLOCATION: values were {vals}, sum={sum(vals)} "
                f"(need {self.g_total}), min required {self.MIN_GREEN}s each."
            )
            terminated = False
            truncated = self.steps >= self.MAX_STEPS
            return obs, 0.0, terminated, truncated, {}

        self.last_allocation = vals
        served = []
        lines = []
        total_served = 0
        for i in range(n):
            sv = min(self.d[i], self.s[i] * vals[i])
            served.append(sv)
            total_served += sv
            if sv >= self.d[i]:
                status = "CLEARED"
            else:
                status = f"QUEUED (~{self.d[i] - sv} left waiting)"
            lines.append(f"  {self.LABELS[i]}: green={vals[i]}s served={sv} [{status}]")

        fraction = total_served / self.optimal_served if self.optimal_served else 1.0
        fraction = min(fraction, 1.0)

        reward = 0.0
        for idx, (thresh, amount) in enumerate(self.thresholds):
            if not self.given[idx] and fraction >= thresh:
                self.given[idx] = True
                reward += amount

        self.best_fraction = max(self.best_fraction, fraction)
        terminated = all(self.given)
        truncated = (not terminated) and self.steps >= self.MAX_STEPS

        obs = f"Cycle result — total served: {total_served} vehicles.\n" + "\n".join(lines)
        if terminated:
            obs += "\nOPTIMAL SPLIT FOUND. Flow maximized."
        elif truncated:
            obs += f"\nOut of steps. Best achieved: {round(self.best_fraction * 100)}% of optimal flow."
        else:
            obs += f"\n{self.MAX_STEPS - self.steps} step(s) remaining."

        info = {}
        return obs, reward, terminated, truncated, info
