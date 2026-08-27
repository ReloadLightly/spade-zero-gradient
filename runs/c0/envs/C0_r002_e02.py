import random
import re


class CaravanLoadoutEnv:
    GOODS = ["silk", "spice", "salt", "dates"]
    WEIGHTS = {"silk": 2, "spice": 3, "salt": 1, "dates": 4}
    PRICES = {"silk": 3, "spice": 4, "salt": 2, "dates": 5}
    PROFIT_LO, PROFIT_HI = 1, 14
    SCOUTS_TOTAL = 3
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.capacity = 0
        self.budget = 0
        self.profits = {}
        self.revealed = set()
        self.scouts_left = self.SCOUTS_TOTAL
        self.loaded = {}
        self.cap_used = 0
        self.bud_used = 0
        self.steps = 0
        self.done = False
        self.optimal_profit = 0
        self.milestone_awarded = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.capacity = self.rng.randint(18, 26)
        self.budget = self.rng.randint(22, 32)
        self.profits = {g: self.rng.randint(self.PROFIT_LO, self.PROFIT_HI) for g in self.GOODS}
        self.revealed = set()
        self.scouts_left = self.SCOUTS_TOTAL
        self.loaded = {g: 0 for g in self.GOODS}
        self.cap_used = 0
        self.bud_used = 0
        self.steps = 0
        self.done = False
        self.milestone_awarded = False
        self.optimal_profit = self._optimal_profit()

        goods_desc = ", ".join(
            f"{g} ({self.WEIGHTS[g]}w/{self.PRICES[g]}c)" for g in self.GOODS
        )
        obs = (
            "You are provisioning a caravan at the oasis market before a desert crossing.\n"
            "Goal: allocate your coin budget and camel weight capacity among four trade "
            "goods to maximize total profit delivered at the far bazaar. Each good's "
            "profit-per-unit is hidden until you scout it.\n"
            f"Camel weight capacity: {self.capacity}. Coin budget: {self.budget}.\n"
            f"Goods (weight/unit, price/unit): {goods_desc}.\n"
            f"Every good's true profit-per-unit lies between {self.PROFIT_LO} and "
            f"{self.PROFIT_HI} coins, but you only have {self.SCOUTS_TOTAL} scout uses "
            "total to reveal exact values for 3 of the 4 goods — one good must stay a mystery.\n"
            "Actions (exactly one per turn):\n"
            "  appraise <good>   - reveal that good's true profit/unit (spends 1 scout)\n"
            "  load <good> <n>   - buy n units of that good onto the camel\n"
            "  depart            - lock in your load and end the trip\n"
            f"You have {self.MAX_STEPS} steps total. Step 0/{self.MAX_STEPS}."
        )
        return obs, {"capacity": self.capacity, "budget": self.budget}

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        text = (action or "").strip().lower()
        parts = text.split()

        if not parts:
            obs = self._malformed()
        elif parts[0] == "appraise" and len(parts) == 2:
            obs, reward = self._do_appraise(parts[1])
        elif parts[0] == "load" and len(parts) == 3:
            obs, reward = self._do_load(parts[1], parts[2])
        elif parts[0] == "depart" and len(parts) == 1:
            obs, reward = self._do_depart()
        else:
            obs = self._malformed()

        terminated = self.done
        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += f"\nOut of steps ({self.MAX_STEPS}/{self.MAX_STEPS}). The caravan departs unplanned."

        info = {
            "capacity_used": self.cap_used,
            "budget_used": self.bud_used,
            "scouts_left": self.scouts_left,
            "optimal_profit": self.optimal_profit,
        }
        return obs, reward, terminated, truncated, info

    def _malformed(self):
        return (
            "Malformed action. Use: 'appraise <good>' | 'load <good> <units>' | 'depart'.\n"
            f"Step {self.steps}/{self.MAX_STEPS}."
        )

    def _do_appraise(self, good):
        if good not in self.GOODS:
            return self._malformed(), 0.0
        if good in self.revealed:
            obs = (
                f"You already know {good}'s profit: {self.profits[good]} coins/unit. "
                f"No scout spent. Scouts left: {self.scouts_left}.\n"
                f"Step {self.steps}/{self.MAX_STEPS}."
            )
            return obs, 0.0
        if self.scouts_left <= 0:
            obs = (
                "No scouts remaining — that good's profit stays hidden.\n"
                f"Step {self.steps}/{self.MAX_STEPS}."
            )
            return obs, 0.0

        self.scouts_left -= 1
        self.revealed.add(good)
        reward = 0.0
        if len(self.revealed) == self.SCOUTS_TOTAL and not self.milestone_awarded:
            self.milestone_awarded = True
            reward = 0.2
        obs = (
            f"Scouted {good}: profit is {self.profits[good]} coins/unit. "
            f"Scouts left: {self.scouts_left}. Revealed so far: {sorted(self.revealed)}.\n"
            f"Capacity used {self.cap_used}/{self.capacity}, budget used {self.bud_used}/{self.budget}.\n"
            f"Step {self.steps}/{self.MAX_STEPS}."
        )
        return obs, reward

    def _do_load(self, good, n_str):
        if good not in self.GOODS or not re.fullmatch(r"\d+", n_str):
            return self._malformed(), 0.0
        n = int(n_str)
        w_cost = n * self.WEIGHTS[good]
        b_cost = n * self.PRICES[good]
        cap_left = self.capacity - self.cap_used
        bud_left = self.budget - self.bud_used

        if n == 0:
            obs = f"Loaded 0 units of {good} — nothing changed.\nStep {self.steps}/{self.MAX_STEPS}."
            return obs, 0.0
        if w_cost > cap_left or b_cost > bud_left:
            max_by_weight = cap_left // self.WEIGHTS[good]
            max_by_budget = bud_left // self.PRICES[good]
            obs = (
                f"Cannot load {n} {good}: needs {w_cost}w/{b_cost}c but only "
                f"{cap_left}w/{bud_left}c remain (max affordable now: "
                f"{min(max_by_weight, max_by_budget)} units).\n"
                f"Step {self.steps}/{self.MAX_STEPS}."
            )
            return obs, 0.0

        self.loaded[good] += n
        self.cap_used += w_cost
        self.bud_used += b_cost
        obs = (
            f"Loaded {n} units of {good}. Camel now carries: "
            f"{ {g: v for g, v in self.loaded.items() if v} }.\n"
            f"Capacity used {self.cap_used}/{self.capacity}, budget used {self.bud_used}/{self.budget}.\n"
            f"Step {self.steps}/{self.MAX_STEPS}."
        )
        return obs, 0.0

    def _do_depart(self):
        self.done = True
        actual_profit = sum(self.loaded[g] * self.profits[g] for g in self.GOODS)
        ratio = 1.0 if self.optimal_profit == 0 else min(1.0, actual_profit / self.optimal_profit)
        reward = 0.8 * ratio
        obs = (
            f"The caravan departs. Delivered profit: {actual_profit} coins "
            f"(best possible with full information: {self.optimal_profit} coins).\n"
            f"Final load: { {g: v for g, v in self.loaded.items() if v} }.\n"
            f"Step {self.steps}/{self.MAX_STEPS}. Trip complete."
        )
        return obs, reward

    def _optimal_profit(self):
        cap, bud = self.capacity, self.budget
        dp = [[0] * (bud + 1) for _ in range(cap + 1)]
        for good in self.GOODS:
            w, p, prof = self.WEIGHTS[good], self.PRICES[good], self.profits[good]
            if w > cap or p > bud:
                continue
            for wi in range(w, cap + 1):
                row = dp[wi]
                prev = dp[wi - w]
                for bi in range(p, bud + 1):
                    candidate = prev[bi - p] + prof
                    if candidate > row[bi]:
                        row[bi] = candidate
        return dp[cap][bud]
