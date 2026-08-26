import random
import re


class HikeSupplyKnapsackEnv:
    LETTERS = "ABCDEFGH"
    NAME_POOL = [
        "Trail Mix Pouch", "Freeze-Dried Chili", "Energy Bar Box",
        "Dehydrated Fruit Bag", "Peanut Butter Packet", "Instant Oatmeal Pack",
        "Jerky Strip Sleeve", "Protein Bar Case", "Summit Sausage Roll",
        "Backcountry Cheese Wedge", "Dried Noodle Brick", "Camp Cocoa Tin",
    ]
    MAX_STEPS = 10
    MAX_INSPECTIONS = 6

    def __init__(self):
        self.rng = None
        self.names = []
        self.weights = []
        self.calories = []
        self.tags = []
        self.capacity = 0
        self.opt = 0
        self.inspected = set()
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        chosen_names = self.rng.sample(self.NAME_POOL, 8)
        order = list(range(8))
        self.rng.shuffle(order)
        self.names = [chosen_names[i] for i in order]
        self.weights = [self.rng.randint(2, 9) for _ in range(8)]
        self.calories = [self.rng.randint(60, 380) for _ in range(8)]
        total_weight = sum(self.weights)
        self.capacity = max(8, round(0.5 * total_weight))
        self.opt = self._knapsack_optimum()
        ratios = [self.calories[i] / self.weights[i] for i in range(8)]
        order_by_ratio = sorted(range(8), key=lambda i: ratios[i])
        third = 8 // 3
        self.tags = [""] * 8
        for rank, idx in enumerate(order_by_ratio):
            if rank < third:
                self.tags[idx] = "sparse"
            elif rank >= 8 - third:
                self.tags[idx] = "dense"
            else:
                self.tags[idx] = "balanced"
        self.inspected = set()
        self.steps = 0
        self.done = False

        lines = [
            "TRAILHEAD PACK BENCH",
            f"Goal: choose a subset of the 8 supplies below to MAXIMIZE total "
            f"calories without exceeding a pack weight capacity of {self.capacity} units.",
            "Exact weight and calorie values are hidden until you INSPECT an item.",
            f"You may INSPECT at most {self.MAX_INSPECTIONS} of the 8 items "
            f"(each INSPECT and the final PACK all count as one action).",
            f"You have at most {self.MAX_STEPS} actions total.",
            "Action formats: 'INSPECT <letter>' to reveal one item's stats, "
            "or 'PACK <letters>' (e.g. 'PACK A C F') to submit your final load and end the episode.",
            "Candidates (letter: name [density tag]):",
        ]
        for i in range(8):
            lines.append(f"  {self.LETTERS[i]}: {self.names[i]} [tag: {self.tags[i]}]")
        lines.append(
            "Density tags are a rough calorie-per-weight guide: 'dense' items pack "
            "the most calories per unit weight, 'sparse' the least."
        )
        return "\n".join(lines), {}

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        reward = 0.0
        terminated = False
        observation = ""

        m_inspect = re.match(r"^INSPECT\s+([A-Ha-h])\s*$", text)
        m_pack = re.match(r"^PACK\s+(.+)$", text, re.IGNORECASE)

        if m_inspect:
            letter = m_inspect.group(1).upper()
            idx = self.LETTERS.index(letter)
            if letter not in self.inspected and len(self.inspected) >= self.MAX_INSPECTIONS:
                observation = (
                    f"Inspection budget exhausted ({self.MAX_INSPECTIONS} used). "
                    "You must now PACK using what you know."
                )
            else:
                self.inspected.add(letter)
                w = self.weights[idx]
                c = self.calories[idx]
                observation = (
                    f"{letter} ({self.names[idx]}): weight={w}, calories={c}, "
                    f"ratio={c / w:.1f} cal/weight. "
                    f"Inspected so far: {len(self.inspected)}/{self.MAX_INSPECTIONS}."
                )
        elif m_pack:
            letters = []
            seen = set()
            valid = True
            for ch in re.findall(r"[A-Ha-h]", m_pack.group(1)):
                up = ch.upper()
                if up not in self.LETTERS:
                    valid = False
                    break
                if up not in seen:
                    seen.add(up)
                    letters.append(up)
            if not valid or not letters:
                observation = (
                    "Malformed PACK action. Use letters A-H only, e.g. 'PACK A C F'."
                )
            else:
                idxs = [self.LETTERS.index(l) for l in letters]
                weight_sum = sum(self.weights[i] for i in idxs)
                cal_sum = sum(self.calories[i] for i in idxs)
                feasible = weight_sum <= self.capacity
                terminated = True
                self.done = True
                if not feasible:
                    reward = 0.0
                    observation = (
                        f"PACK {''.join(letters)} weighs {weight_sum} > capacity "
                        f"{self.capacity}. Overweight pack fails. Episode over."
                    )
                else:
                    ratio_to_opt = min(1.0, cal_sum / self.opt) if self.opt > 0 else 1.0
                    reward = 0.2 + 0.8 * ratio_to_opt
                    observation = (
                        f"PACK {''.join(letters)}: weight={weight_sum}/{self.capacity} "
                        f"(feasible), calories={cal_sum} (best possible was {self.opt}). "
                        f"Score: {reward:.3f}. Episode over."
                    )
        else:
            observation = (
                "Unrecognized action. Use 'INSPECT <letter>' or 'PACK <letters>', "
                "e.g. 'INSPECT B' or 'PACK A C F'."
            )

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            observation += " Step limit reached without a PACK submission. Episode over."

        return observation, reward, terminated, truncated, {}

    def _knapsack_optimum(self):
        dp = [0] * (self.capacity + 1)
        for w, c in zip(self.weights, self.calories):
            for cap in range(self.capacity, w - 1, -1):
                candidate = dp[cap - w] + c
                if candidate > dp[cap]:
                    dp[cap] = candidate
        return dp[self.capacity]
