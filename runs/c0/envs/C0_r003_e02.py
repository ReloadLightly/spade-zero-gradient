import random
import itertools
import string


class LadderBudgetEnv:
    """Buy ladder rungs of mixed prices within a budget to maximize total height."""

    MAX_STEPS = 10
    N_RUNGS = 6
    PRICE_LO, PRICE_HI = 3, 15
    HEIGHT_LO, HEIGHT_HI = 2, 9

    def __init__(self):
        self.rng = None
        self.prices = []
        self.heights = []
        self.budget = 0
        self.ids = []
        self.optimal_height = 0
        self.optimal_price = 0
        self.inspected = set()
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.ids = list(string.ascii_uppercase[: self.N_RUNGS])
        self.steps = 0
        self.done = False
        self.inspected = set()

        for _attempt in range(200):
            prices = [self.rng.randint(self.PRICE_LO, self.PRICE_HI) for _ in range(self.N_RUNGS)]
            heights = [self.rng.randint(self.HEIGHT_LO, self.HEIGHT_HI) for _ in range(self.N_RUNGS)]
            total_price = sum(prices)
            cheapest_two = sum(sorted(prices)[:2])
            budget = self.rng.randint(cheapest_two, total_price - 1)

            best_height, best_price, best_size = 0, 0, 0
            for r in range(1, self.N_RUNGS + 1):
                for combo in itertools.combinations(range(self.N_RUNGS), r):
                    tp = sum(prices[i] for i in combo)
                    if tp > budget:
                        continue
                    th = sum(heights[i] for i in combo)
                    if th > best_height or (th == best_height and tp < best_price):
                        best_height, best_price, best_size = th, tp, r

            total_height = sum(heights)
            if 2 <= best_size <= 4 and 0 < best_height < total_height:
                self.prices, self.heights, self.budget = prices, heights, budget
                self.optimal_height, self.optimal_price = best_height, best_price
                break
        else:
            self.prices, self.heights, self.budget = prices, heights, budget
            self.optimal_height, self.optimal_price = best_height, best_price

        obs = (
            "Ladder yard: {n} rungs for sale, labeled {ids}. "
            "Prices run roughly {plo}-{phi} coins, heights roughly {hlo}-{hhi} feet. "
            "Your budget is {budget} coins. Buy a subset of rungs to maximize total "
            "height WITHOUT exceeding budget.\n"
            "Actions: 'INSPECT <id>' reveals a rung's exact price and height "
            "(costs a step). 'BUY <id,id,...>' submits your final purchase and "
            "ends the episode if it fits the budget. You have {max_steps} steps total."
        ).format(
            n=self.N_RUNGS,
            ids=",".join(self.ids),
            plo=self.PRICE_LO,
            phi=self.PRICE_HI,
            hlo=self.HEIGHT_LO,
            hhi=self.HEIGHT_HI,
            budget=self.budget,
            max_steps=self.MAX_STEPS,
        )
        info = {"budget": self.budget, "n_rungs": self.N_RUNGS}
        return obs, info

    def _parse_ids(self, text):
        raw = text.replace(",", " ").split()
        seen = []
        for tok in raw:
            u = tok.strip().upper()
            if u and u not in seen:
                seen.append(u)
        return seen

    def step(self, action):
        self.steps += 1
        info = {}

        if self.done:
            return "Episode already over.", 0.0, True, False, info

        text = (action or "").strip()
        parts = text.split(None, 1)
        verb = parts[0].upper() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if verb == "INSPECT":
            ids = self._parse_ids(rest)
            if len(ids) != 1 or ids[0] not in self.ids:
                obs = "Malformed INSPECT: give exactly one valid rung id, e.g. 'INSPECT C'."
                return self._maybe_truncate(obs, 0.0, info)
            rid = ids[0]
            idx = self.ids.index(rid)
            self.inspected.add(rid)
            obs = "Rung {r}: price={p} coins, height={h} feet. (Inspected so far: {s})".format(
                r=rid, p=self.prices[idx], h=self.heights[idx],
                s=",".join(sorted(self.inspected)),
            )
            return self._maybe_truncate(obs, 0.0, info)

        if verb == "BUY":
            ids = self._parse_ids(rest)
            if not ids or any(i not in self.ids for i in ids):
                obs = "Malformed BUY: list one or more valid rung ids, e.g. 'BUY A,C,D'."
                return self._maybe_truncate(obs, 0.0, info)

            idxs = [self.ids.index(i) for i in ids]
            total_price = sum(self.prices[i] for i in idxs)
            total_height = sum(self.heights[i] for i in idxs)

            if total_price > self.budget:
                obs = (
                    "Purchase {ids} costs {tp} coins, over budget by {over} coins. "
                    "Not submitted -- choose a cheaper combination."
                ).format(ids=",".join(ids), tp=total_price, over=total_price - self.budget)
                return self._maybe_truncate(obs, 0.0, info)

            ratio = min(1.0, total_height / self.optimal_height) if self.optimal_height else 0.0
            reward = 0.2 + 0.8 * ratio
            self.done = True
            obs = (
                "Purchased {ids} for {tp}/{budget} coins, total height {th} feet "
                "(best possible within budget: {opt} feet). Episode complete."
            ).format(
                ids=",".join(ids), tp=total_price, budget=self.budget,
                th=total_height, opt=self.optimal_height,
            )
            return obs, reward, True, False, info

        obs = "Unknown action. Use 'INSPECT <id>' or 'BUY <id,id,...>'."
        return self._maybe_truncate(obs, 0.0, info)

    def _maybe_truncate(self, obs, reward, info):
        if self.steps >= self.MAX_STEPS:
            self.done = True
            return obs + " Step limit reached; episode truncated.", reward, False, True, info
        return obs, reward, False, False, info
