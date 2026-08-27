import random
import re


class HiddenCapacityExpeditionEnv:
    MAX_STEPS = 10
    N_ITEMS = 6

    def __init__(self):
        self.rng = None
        self.step_count = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False

        weights = self.rng.sample(range(3, 10), self.N_ITEMS)
        values = [self.rng.randint(2, 9) for _ in range(self.N_ITEMS)]
        self.items = list(zip(weights, values))
        total = sum(weights)

        self.cap_min = max(max(weights), (total * 45) // 100)
        self.cap_max = (total * 70) // 100
        if self.cap_max <= self.cap_min:
            self.cap_max = self.cap_min + 1
        self.true_capacity = self.rng.randint(self.cap_min, self.cap_max)

        self.optimal_value = self._best_value(self.true_capacity)

        self.lower_bound = self.cap_min
        self.upper_bound = self.cap_max
        init_width = self.upper_bound - self.lower_bound
        self.threshold1 = max(1, init_width // 2)
        self.threshold2 = max(1, init_width // 4)
        self.milestone1_done = False
        self.milestone2_done = False

        item_lines = "\n".join(
            f"  item {i}: weight {w}, value {v}" for i, (w, v) in enumerate(self.items)
        )
        obs = (
            "EXPEDITION SLED PACKING\n"
            f"You have {self.N_ITEMS} items:\n{item_lines}\n"
            f"The sled's maximum weight capacity is a hidden integer known only to be "
            f"between {self.cap_min} and {self.cap_max} (inclusive).\n"
            "Before committing, you may test candidate loads:\n"
            "  TEST i,j,k   -- propose a subset of item indices; you learn its total "
            "weight and whether it FITS or is TOO HEAVY for the true capacity.\n"
            "When ready, commit once and only once:\n"
            "  PACK i,j,k   -- final subset. If its weight exceeds the true capacity "
            "you score nothing; otherwise you score by how close its value is to the "
            "best possible load that would have fit.\n"
            f"You have {self.MAX_STEPS} actions total (TEST and PACK both count). "
            "PACK ends the episode."
        )
        return obs, {}

    def _weight_value(self, indices):
        w = sum(self.items[i][0] for i in indices)
        v = sum(self.items[i][1] for i in indices)
        return w, v

    def _best_value(self, capacity):
        best = 0
        n = self.N_ITEMS
        for mask in range(1 << n):
            w = 0
            v = 0
            for i in range(n):
                if mask & (1 << i):
                    w += self.items[i][0]
                    v += self.items[i][1]
            if w <= capacity and v > best:
                best = v
        return best

    def _parse_indices(self, text):
        parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
        indices = []
        for p in parts:
            if not p.isdigit():
                return None
            i = int(p)
            if i < 0 or i >= self.N_ITEMS:
                return None
            indices.append(i)
        if not indices or len(set(indices)) != len(indices):
            return None
        return indices

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        m = re.match(r"^(TEST|PACK)\s+(.*)$", text, re.IGNORECASE)
        truncated = self.step_count >= self.MAX_STEPS

        if not m:
            obs = "Malformed action. Use 'TEST i,j,k' or 'PACK i,j,k' with item indices."
            if truncated:
                self.done = True
                return obs, 0.0, False, True, {}
            return obs, 0.0, False, False, {}

        verb = m.group(1).upper()
        indices = self._parse_indices(m.group(2))
        if indices is None:
            obs = "Malformed indices. Use distinct item indices 0-5, e.g. 'TEST 0,2,4'."
            if truncated:
                self.done = True
                return obs, 0.0, False, True, {}
            return obs, 0.0, False, False, {}

        w, v = self._weight_value(indices)

        if verb == "TEST":
            fits = w <= self.true_capacity
            if fits:
                self.lower_bound = max(self.lower_bound, w)
            else:
                self.upper_bound = min(self.upper_bound, w - 1)

            reward = 0.0
            width = self.upper_bound - self.lower_bound
            if not self.milestone1_done and width <= self.threshold1:
                self.milestone1_done = True
                reward += 0.2
            if not self.milestone2_done and width <= self.threshold2:
                self.milestone2_done = True
                reward += 0.2

            verdict = "FITS" if fits else "TOO HEAVY"
            obs = (
                f"Tested items {indices}: total weight {w} -> {verdict}. "
                f"Capacity now known to be between {self.lower_bound} and {self.upper_bound}."
            )
            if truncated:
                self.done = True
                return obs, reward, False, True, {}
            return obs, reward, False, False, {}

        self.done = True
        if w > self.true_capacity:
            obs = (
                f"PACK failed: items {indices} weigh {w}, exceeding the true capacity "
                f"of {self.true_capacity}. Expedition aborted with nothing scored."
            )
            return obs, 0.0, True, False, {}

        if v >= self.optimal_value:
            reward = 0.6
        elif v >= (self.optimal_value * 85) // 100:
            reward = 0.3
        else:
            reward = 0.1

        obs = (
            f"PACK committed: items {indices}, weight {w} (capacity was {self.true_capacity}), "
            f"value {v}. Best possible value within true capacity was {self.optimal_value}."
        )
        return obs, reward, True, False, {}
