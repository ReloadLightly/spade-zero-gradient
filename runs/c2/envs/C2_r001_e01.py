import random


class CARuleDetectiveEnv:
    def __init__(self):
        self.width = 9
        self.step_limit = 10
        self.gen_target = 3
        self.num_targets = 3
        self.rng = None
        self.table = {}
        self.initial_row = []
        self.targets = []
        self.secret_values = []
        self.steps = 0
        self.queries_used = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        canonical_keys = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        while True:
            bits = {k: self.rng.randint(0, 1) for k in canonical_keys}
            if len(set(bits.values())) > 1:
                break
        self.table = {}
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    key = (a, b, c) if a <= c else (c, b, a)
                    self.table[(a, b, c)] = bits[key]

        self.initial_row = [self.rng.randint(0, 1) for _ in range(self.width)]
        self.targets = sorted(self.rng.sample(range(self.width), self.num_targets))

        row = list(self.initial_row)
        for _ in range(self.gen_target):
            row = self._advance(row)
        self.secret_values = [row[p] for p in self.targets]

        self.steps = 0
        self.queries_used = 0
        self.done = False

        row_str = "".join(str(v) for v in self.initial_row)
        obs = (
            "You are debugging a 1-dimensional binary cellular automaton with a hidden "
            f"update rule. The grid has {self.width} cells at positions 0..{self.width - 1}. "
            "Cells outside the grid are always treated as 0 (fixed zero boundary). "
            f"Generation 0 is: {row_str}. Each cell's value at the next generation is an "
            "unknown function of its own current value and its left and right neighbors' "
            "current values; you do not know this function yet. "
            "Send 'QUERY abc' (abc = three characters, each 0 or 1, e.g. QUERY 010) to learn "
            "what the rule outputs when the left neighbor is a, the cell itself is b, and the "
            "right neighbor is c. "
            f"When ready, send 'PREDICT p:v p:v p:v' giving your predicted value (0 or 1) for "
            f"generation {self.gen_target} at exactly these three positions: {self.targets}. "
            f"You have {self.step_limit} actions total; every QUERY or PREDICT attempt, valid "
            "or not, counts as one action. The episode ends when you submit a valid PREDICT or "
            "when actions run out."
        )
        return obs, {}

    def _advance(self, row):
        n = len(row)
        new_row = []
        for i in range(n):
            a = row[i - 1] if i - 1 >= 0 else 0
            b = row[i]
            c = row[i + 1] if i + 1 < n else 0
            new_row.append(self.table[(a, b, c)])
        return new_row

    def _finish(self, obs, reward):
        if self.steps >= self.step_limit:
            self.done = True
            return obs + " Step limit reached; episode truncated.", reward, False, True, {}
        return obs, reward, False, False, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        parts = text.split()

        if not parts:
            return self._finish(
                "Empty action. Use 'QUERY abc' or 'PREDICT p:v p:v p:v'.", 0.0
            )

        verb = parts[0].upper()

        if verb == "QUERY":
            if len(parts) != 2 or len(parts[1]) != 3 or any(ch not in "01" for ch in parts[1]):
                return self._finish(
                    "Malformed QUERY. Use exactly 'QUERY abc' where abc is three characters, "
                    "each 0 or 1 (e.g., QUERY 010).",
                    0.0,
                )
            a, b, c = (int(ch) for ch in parts[1])
            val = self.table[(a, b, c)]
            self.queries_used += 1
            obs = (
                f"Neighborhood {parts[1]} -> next value {val}. "
                f"({self.steps}/{self.step_limit} actions used, {self.queries_used} queries so far)"
            )
            return self._finish(obs, 0.0)

        if verb == "PREDICT":
            pairs = parts[1:]
            parsed = {}
            valid = True
            for p in pairs:
                if ":" not in p:
                    valid = False
                    break
                pos_s, val_s = p.split(":", 1)
                if not pos_s.isdigit() or val_s not in ("0", "1"):
                    valid = False
                    break
                parsed[int(pos_s)] = int(val_s)
            if not valid or set(parsed.keys()) != set(self.targets):
                return self._finish(
                    "Malformed PREDICT. Provide exactly one value for each target position "
                    f"{self.targets} in the form 'PREDICT p:v p:v p:v'.",
                    0.0,
                )
            correct = sum(
                1 for t, sv in zip(self.targets, self.secret_values) if parsed[t] == sv
            )
            reward = correct / len(self.targets)
            self.done = True
            obs = (
                f"Prediction submitted: {correct}/{len(self.targets)} correct. "
                "Episode complete."
            )
            return obs, reward, True, False, {"correct": correct}

        return self._finish(
            "Unknown action verb. Use 'QUERY abc' or 'PREDICT p:v p:v p:v'.", 0.0
        )
