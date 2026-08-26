import random
import re
import itertools


class HikePackKnapsackEnv:
    ITEM_IDS = ["A", "B", "C", "D", "E", "F"]
    ITEM_NAMES = [
        "Trail Mix Pack",
        "Energy Bar",
        "Dehydrated Rice",
        "Jerky Pouch",
        "Peanut Butter Packet",
        "Freeze-Dried Fruit",
    ]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.max_steps = 10
        self.step_count = 0
        self.done = False
        self.inspected = set()
        self.inspect_reward_count = 0

        weights = [self.rng.randint(60, 320) for _ in self.ITEM_IDS]
        cal_lo, cal_hi, true_cal = [], [], []
        for _ in self.ITEM_IDS:
            lo = self.rng.randint(80, 400)
            hi = lo + self.rng.randint(50, 250)
            cal_lo.append(lo)
            cal_hi.append(hi)
            true_cal.append(self.rng.randint(lo, hi))

        self.items = {}
        for i, iid in enumerate(self.ITEM_IDS):
            self.items[iid] = {
                "name": self.ITEM_NAMES[i],
                "weight": weights[i],
                "cal_lo": cal_lo[i],
                "cal_hi": cal_hi[i],
                "true_cal": true_cal[i],
            }

        total_weight = sum(weights)
        frac = self.rng.uniform(0.45, 0.6)
        budget = int(total_weight * frac)
        budget = max(budget, min(weights) + 20)
        budget = min(budget, total_weight - 1)
        self.weight_budget = budget

        best = 0
        ids = self.ITEM_IDS
        for r in range(len(ids) + 1):
            for combo in itertools.combinations(ids, r):
                w = sum(self.items[i]["weight"] for i in combo)
                if w <= self.weight_budget:
                    c = sum(self.items[i]["true_cal"] for i in combo)
                    if c > best:
                        best = c
        self.optimal_cal = max(best, 1)

        return self._initial_obs(), {}

    def _initial_obs(self):
        lines = [
            "HIKE PACKING: pick a subset of food items to carry without exceeding "
            "the weight budget, maximizing total calories.",
            f"Weight budget: {self.weight_budget}g. Step limit: {self.max_steps}.",
            "Each item's weight is printed on the label (known exactly). Its "
            "calorie count is hidden behind a range until you read the nutrition label.",
            "Actions: 'inspect <id>' reveals one item's exact calories (1 step). "
            "'pack <id> <id> ...' commits your final selection and ends the trip (1 step).",
            "Items:",
        ]
        for iid in self.ITEM_IDS:
            it = self.items[iid]
            lines.append(
                f"  {iid}: {it['name']} - weight {it['weight']}g, "
                f"calories in [{it['cal_lo']}, {it['cal_hi']}] kcal (unconfirmed)"
            )
        return "\n".join(lines)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        parts = text.split(None, 1)

        if not parts:
            obs = "Malformed action. Use 'inspect <id>' or 'pack <id> <id> ...'."
            truncated = self.step_count >= self.max_steps
            self.done = self.done or truncated
            return obs, 0.0, False, truncated, {}

        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd == "inspect":
            iid = rest.strip().upper()
            if iid not in self.items:
                obs = (
                    f"Malformed action: unknown item '{rest.strip()}'. "
                    f"Use: inspect <id> where id is one of {self.ITEM_IDS}."
                )
                truncated = self.step_count >= self.max_steps
                self.done = self.done or truncated
                return obs, 0.0, False, truncated, {}

            reward = 0.0
            if iid not in self.inspected:
                self.inspected.add(iid)
                if self.inspect_reward_count < 4:
                    reward = 0.05
                    self.inspect_reward_count += 1

            item = self.items[iid]
            obs = (
                f"Label check on {iid} ({item['name']}): weight {item['weight']}g "
                f"(known), calories = {item['true_cal']} kcal (confirmed). "
                f"Inspected so far: {sorted(self.inspected)}. "
                f"Steps used: {self.step_count}/{self.max_steps}."
            )
            truncated = self.step_count >= self.max_steps
            self.done = self.done or truncated
            return obs, reward, False, truncated, {}

        elif cmd == "pack":
            raw = [x for x in re.split(r"[,\s]+", rest.strip()) if x]
            ids_clean = [x.upper() for x in raw]
            valid = (
                bool(ids_clean)
                and all(x in self.items for x in ids_clean)
                and len(set(ids_clean)) == len(ids_clean)
            )
            if not valid:
                obs = (
                    "Malformed pack action: give distinct valid item ids, "
                    "e.g. 'pack A C E'."
                )
                truncated = self.step_count >= self.max_steps
                self.done = self.done or truncated
                return obs, 0.0, False, truncated, {}

            total_weight = sum(self.items[i]["weight"] for i in ids_clean)
            total_cal = sum(self.items[i]["true_cal"] for i in ids_clean)
            self.done = True

            if total_weight > self.weight_budget:
                obs = (
                    f"Over budget: packed {total_weight}g exceeds the "
                    f"{self.weight_budget}g limit. Trip aborted, 0 packing reward."
                )
                return obs, 0.0, True, False, {}

            frac = max(0.0, min(1.0, total_cal / self.optimal_cal))
            reward = round(0.8 * frac, 4)
            obs = (
                f"Packed {ids_clean}: weight {total_weight}/{self.weight_budget}g, "
                f"calories {total_cal} (best possible under budget: "
                f"{self.optimal_cal}). Packing reward: {reward:.2f}."
            )
            return obs, reward, True, False, {}

        else:
            obs = "Malformed action. Use 'inspect <id>' or 'pack <id> <id> ...'."
            truncated = self.step_count >= self.max_steps
            self.done = self.done or truncated
            return obs, 0.0, False, truncated, {}
