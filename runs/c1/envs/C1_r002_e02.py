import re


class SaltRoadRelayEnv:
    MAX_STEPS = 10
    LEG_REWARD = 0.1
    BONUS = 0.6

    def __init__(self):
        self.rng = None

    def reset(self, seed=None):
        self.rng = __import__("random").Random(seed)
        self.capacity = self.rng.randint(8, 10)
        legs = [self.rng.randint(2, 5) for _ in range(4)]
        while sum(legs) <= self.capacity:
            idx = self.rng.randrange(4)
            if legs[idx] < min(6, self.capacity):
                legs[idx] += 1
        self.legs = legs
        self.prices = [0, self.rng.randint(1, 6), self.rng.randint(1, 6), self.rng.randint(1, 6)]
        self.known = [True, False, False, False]
        self.optimal_cost = self._solve_optimal()

        self.stock = 0
        self.spent = 0
        self.wp_index = 0
        self.steps_used = 0
        self.done = False

        obs = (
            "SALT ROAD RELAY. Guide a caravan from Oasis W0 through W1, W2, W3 to the "
            "final Oasis W4, buying water along the way. Carrying capacity: "
            f"{self.capacity} units (never exceed it). The four legs cost exactly "
            f"{self.legs[0]}, {self.legs[1]}, {self.legs[2]}, {self.legs[3]} water to "
            "cross, in order; if your stock would go negative crossing a leg, the run fails.\n"
            f"W0's water is free. W1 is tagged {self._tag(1)}, W2 is tagged {self._tag(2)}, "
            f"W3 is tagged {self._tag(3)} (cheap=1-2/unit, moderate=3-4, pricey=5-6) but exact "
            "prices are hidden until inspected.\n"
            "At each waypoint you may act once, then you automatically move to the next: "
            "'inspect' reveals the exact price here (no cost); 'buy N' purchases N units at "
            "this waypoint's price (N=0 allowed) and advances you to the next waypoint, "
            f"consuming that leg's water cost. You have {self.MAX_STEPS} total actions."
        )
        return obs, {}

    def _tag(self, i):
        p = self.prices[i]
        return "cheap" if p <= 2 else ("moderate" if p <= 4 else "pricey")

    def _solve_optimal(self):
        dp = {0: 0}
        for i in range(4):
            price = self.prices[i]
            after_buy = {}
            for w, cost in dp.items():
                for wprime in range(w, self.capacity + 1):
                    c = cost + price * (wprime - w)
                    if wprime not in after_buy or c < after_buy[wprime]:
                        after_buy[wprime] = c
            leg = self.legs[i]
            after_leg = {}
            for w, cost in after_buy.items():
                rem = w - leg
                if rem >= 0 and (rem not in after_leg or cost < after_leg[rem]):
                    after_leg[rem] = cost
            dp = after_leg
        return min(dp.values())

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps_used += 1
        text = (action or "").strip().lower()

        if text == "inspect":
            obs, reward = self._do_inspect()
            terminated, truncated = False, False
        else:
            m = re.match(r"^buy\s+(\d+)$", text)
            if not m:
                obs = ("Malformed action. Use 'inspect' or 'buy N' where N is a "
                       "non-negative integer.")
                reward = 0.0
                terminated, truncated = False, False
            else:
                n = int(m.group(1))
                if n > self.capacity - self.stock:
                    obs = (f"Cannot buy {n}: capacity is {self.capacity}, you carry "
                           f"{self.stock}, room for at most {self.capacity - self.stock}.")
                    reward = 0.0
                    terminated, truncated = False, False
                else:
                    obs, reward, terminated = self._do_buy(n)
                    truncated = False

        if not terminated and self.steps_used >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += (f" Step budget exhausted at waypoint W{self.wp_index}; "
                    "run incomplete.")
        if terminated:
            self.done = True

        info = {"wp_index": self.wp_index, "stock": self.stock, "spent": self.spent}
        return obs, reward, terminated, truncated, info

    def _do_inspect(self):
        i = self.wp_index
        if i == 0:
            return "W0's water is free (price 0); no need to inspect.", 0.0
        if self.known[i]:
            return f"Already known: W{i} charges {self.prices[i]}/unit.", 0.0
        self.known[i] = True
        return f"You inspect the well. W{i} charges exactly {self.prices[i]}/unit.", 0.0

    def _do_buy(self, n):
        i = self.wp_index
        price = self.prices[i]
        self.spent += price * n
        self.stock += n
        leg = self.legs[i]
        remaining = self.stock - leg

        if remaining < 0:
            obs = (f"You buy {n} at W{i} (stock {self.stock}) and set out, but run dry "
                   f"crossing the leg to W{i + 1} (needed {leg}, had {self.stock}). "
                   f"Total spent: {self.spent}.")
            return obs, 0.0, True

        self.stock = remaining
        self.wp_index += 1
        reward = self.LEG_REWARD

        if self.wp_index == 4:
            ratio = min(1.0, self.optimal_cost / self.spent) if self.spent > 0 else 1.0
            bonus = self.BONUS * ratio
            reward += bonus
            obs = (f"You buy {n} at W{i} and cross into the Final Oasis (W4) with "
                   f"{self.stock} water to spare! Total spent: {self.spent} "
                   f"(true optimum was {self.optimal_cost}). Journey complete.")
            return obs, reward, True

        obs = (f"You buy {n} at W{i}, cross the leg, and arrive at W{self.wp_index} "
               f"with {self.stock} water. W{self.wp_index} is tagged "
               f"{self._tag(self.wp_index)}. Spent so far: {self.spent}.")
        return obs, reward, False
