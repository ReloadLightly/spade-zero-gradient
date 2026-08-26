import random
import re


class HikingSupplyOptimizerEnv:
    """Pack-weight-vs-calories knapsack with hidden per-item stats."""

    _NAMES = [
        "Freeze-Dried Meal", "Trail Mix", "Energy Bar", "Canned Stew",
        "Peanut Butter Packet", "Dehydrated Soup", "Beef Jerky",
        "Granola Bar", "Instant Oatmeal", "Protein Powder",
    ]
    _MILESTONES = [(0.0, "valid", 0.2), (0.7, "seventy", 0.3),
                   (0.9, "ninety", 0.3), (1.0, "optimal", 0.2)]
    _LABELS = [(1.0, "OPTIMAL"), (0.9, "EXCELLENT"), (0.7, "GOOD"),
               (0.5, "FAIR"), (0.0, "POOR")]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.n = 6
        names = self.rng.sample(self._NAMES, self.n)
        self.items = []
        for nm in names:
            w = self.rng.randint(2, 9)
            cal = self.rng.randint(80, 450)
            self.items.append({"name": nm, "weight": w, "calories": cal,
                                "inspected": False})
        total_w = sum(it["weight"] for it in self.items)
        self.capacity = max(self.items, key=lambda it: it["weight"])["weight"]
        self.capacity = max(self.capacity,
                             round(total_w * self.rng.uniform(0.5, 0.65)))
        self.optimal_calories = self._best_calories()
        while self.optimal_calories <= 0:
            self.capacity += 1
            self.optimal_calories = self._best_calories()
        self.step_count = 0
        self.awarded = set()
        self.best_calories_seen = 0
        self.terminated = False
        obs = self._intro()
        return obs, {"capacity": self.capacity, "n_items": self.n}

    def _best_calories(self):
        best = 0
        for mask in range(1 << self.n):
            w = c = 0
            for i in range(self.n):
                if mask & (1 << i):
                    w += self.items[i]["weight"]
                    c += self.items[i]["calories"]
            if w <= self.capacity and c > best:
                best = c
        return best

    def _intro(self):
        lines = [
            "You are packing supplies for a hike. Choose a subset of the "
            "gear below that maximizes total calories without exceeding "
            "the pack's weight capacity.",
            f"Weight capacity: {self.capacity} units.",
            "Gear (weight and calories hidden until inspected):",
        ]
        for i, it in enumerate(self.items, 1):
            lines.append(f"  {i}. {it['name']}")
        lines += [
            "Actions (exactly one per turn):",
            "  INSPECT <id>  - reveal an item's weight and calories.",
            "  PACK <id,id,...>  - submit a selection of item ids to try "
            "(you may retry after a valid attempt).",
            "You have 10 steps total to reach the best possible calorie "
            "total.",
        ]
        return "\n".join(lines)

    def _label(self, frac):
        for thresh, lab in self._LABELS:
            if frac >= thresh:
                return lab
        return "POOR"

    def step(self, action):
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {}
        self.step_count += 1
        action = (action or "").strip()
        m_inspect = re.fullmatch(r"(?i)INSPECT\s+(\d+)", action)
        m_pack = re.fullmatch(r"(?i)PACK\s+([\d,\s]+)", action)

        if m_inspect:
            obs, reward, terminated = self._do_inspect(int(m_inspect.group(1)))
        elif m_pack:
            ids = [s for s in re.split(r"[,\s]+", m_pack.group(1).strip()) if s]
            obs, reward, terminated = self._do_pack(ids)
        else:
            obs = ("Unrecognized action. Use 'INSPECT <id>' or "
                   "'PACK <id,id,...>'.")
            reward, terminated = 0.0, False

        truncated = False
        if not terminated and self.step_count >= 10:
            truncated = True
        self.terminated = terminated
        return obs, reward, terminated, truncated, {
            "step": self.step_count, "best_calories": self.best_calories_seen,
            "optimal_calories": self.optimal_calories,
        }

    def _do_inspect(self, idx):
        if not (1 <= idx <= self.n):
            return (f"No item {idx}. Valid ids are 1-{self.n}.", 0.0, False)
        it = self.items[idx - 1]
        it["inspected"] = True
        ratio = it["calories"] / it["weight"]
        obs = (f"Item {idx} ({it['name']}): weight {it['weight']}, "
               f"calories {it['calories']} (ratio {ratio:.1f} cal/weight).")
        return obs, 0.0, False

    def _do_pack(self, ids):
        seen = set()
        parsed = []
        for s in ids:
            if not s.isdigit():
                return (f"'{s}' is not a valid item id.", 0.0, False)
            i = int(s)
            if not (1 <= i <= self.n):
                return (f"No item {i}. Valid ids are 1-{self.n}.", 0.0, False)
            if i in seen:
                return (f"Item {i} was selected more than once.", 0.0, False)
            seen.add(i)
            parsed.append(i)
        if not parsed:
            return ("PACK requires at least one item id.", 0.0, False)

        weight = sum(self.items[i - 1]["weight"] for i in parsed)
        calories = sum(self.items[i - 1]["calories"] for i in parsed)
        if weight > self.capacity:
            over = weight - self.capacity
            return (f"That selection weighs {weight}, exceeding capacity "
                    f"by {over}. Not a valid pack.", 0.0, False)

        self.best_calories_seen = max(self.best_calories_seen, calories)
        frac = calories / self.optimal_calories
        reward = 0.0
        for thresh, name, val in self._MILESTONES:
            if frac >= thresh and name not in self.awarded:
                self.awarded.add(name)
                reward += val
        terminated = "optimal" in self.awarded
        label = self._label(frac)
        remaining = self.capacity - weight
        if terminated:
            obs = (f"Valid pack: weight {weight}/{self.capacity}, "
                   f"calories {calories}. Efficiency: {label}. This is the "
                   "best achievable total. Success!")
        else:
            obs = (f"Valid pack: weight {weight}/{self.capacity} "
                   f"(spare capacity {remaining}), calories {calories}. "
                   f"Efficiency: {label}. You may INSPECT more items or "
                   "PACK again to improve.")
        return obs, reward, terminated
