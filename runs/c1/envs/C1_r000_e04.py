import random
import re


class FractalPlantEnv:
    MAX_N = 9
    TARGETS = {4: 0.3, 5: 0.3, 6: 0.4}
    STEP_LIMIT = 10

    def __init__(self):
        self.rng = None
        self.true_counts = []
        self.revealed = set()
        self.awarded = set()
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        a = self.rng.choice([1, 2])
        b = self.rng.choice([1, 2])
        c0 = self.rng.choice([1, 2, 3])
        c1 = self.rng.choice([1, 2, 3])
        counts = [c0, c1]
        for i in range(2, self.MAX_N + 1):
            counts.append(a * counts[i - 1] + b * counts[i - 2])
        self.true_counts = counts
        self.revealed = {0, 1}
        self.awarded = set()
        self.steps = 0
        self.done = False

        obs = (
            "A fractal plant grows generation by generation. "
            f"Generation 0 has {c0} branch tips. Generation 1 has {c1} branch tips. "
            "Every later generation's branch count is produced by a fixed hidden rule "
            "applied to the two immediately preceding generations' counts. "
            "Goal: correctly predict the branch counts at Generations 4, 5, and 6 "
            "BEFORE they have been revealed to you (partial credit for each one you "
            "get right: 0.3, 0.3, 0.4). "
            "Actions (send exactly one per turn):\n"
            "  GROW                -> reveals the branch count at the next unrevealed generation, in order.\n"
            "  PREDICT <n> <value> -> guess the branch count at generation n (n must not "
            "already be revealed to you). A correct guess reveals and confirms that "
            "generation's count. A wrong guess only tells you whether your value was "
            "too high or too low, and does NOT reveal the true count, so you may try "
            "again.\n"
            f"You have {self.STEP_LIMIT} steps total. Begin."
        )
        return obs, {"max_generation": self.MAX_N, "step_limit": self.STEP_LIMIT}

    def _grow_obs(self):
        parts = []
        for n in sorted(self.revealed):
            parts.append(f"Gen {n}: {self.true_counts[n]}")
        return "; ".join(parts)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        reward = 0.0
        terminated = False

        m_grow = re.match(r'^GROW$', action, re.IGNORECASE)
        m_pred = re.match(r'^PREDICT\s+(-?\d+)\s+(-?\d+)\s*$', action, re.IGNORECASE)

        if m_grow:
            highest = max(self.revealed)
            nxt = highest + 1
            if nxt > self.MAX_N:
                obs = "The observed range is fully grown; GROW has nothing left to reveal."
            else:
                self.revealed.add(nxt)
                obs = (
                    f"The plant grows. Generation {nxt} has {self.true_counts[nxt]} "
                    f"branch tips. Revealed so far -> {self._grow_obs()}"
                )
        elif m_pred:
            n = int(m_pred.group(1))
            value = int(m_pred.group(2))
            if n < 0 or n > self.MAX_N:
                obs = f"Invalid generation {n}. Valid range is 0 to {self.MAX_N}."
            elif n in self.revealed:
                obs = (
                    f"Generation {n} is already known to you "
                    f"(it has {self.true_counts[n]} branch tips) — predict an "
                    "unrevealed generation instead."
                )
            else:
                actual = self.true_counts[n]
                if value == actual:
                    self.revealed.add(n)
                    obs = f"Correct. Generation {n} indeed has {actual} branch tips."
                    if n in self.TARGETS and n not in self.awarded:
                        reward = self.TARGETS[n]
                        self.awarded.add(n)
                        obs += f" (+{reward:.2f} reward)"
                elif value > actual:
                    obs = f"Incorrect: your guess for generation {n} is too high. Try again or gather more evidence."
                else:
                    obs = f"Incorrect: your guess for generation {n} is too low. Try again or gather more evidence."
        else:
            obs = (
                "Malformed action. Use exactly 'GROW' or 'PREDICT <n> <value>' "
                "with integer arguments."
            )

        if set(self.TARGETS) <= self.awarded:
            terminated = True
            self.done = True
            obs += " All target generations correctly predicted — episode complete."

        truncated = False
        if not terminated and self.steps >= self.STEP_LIMIT:
            truncated = True
            self.done = True
            obs += " Step limit reached."

        return obs, reward, terminated, truncated, {"revealed": sorted(self.revealed)}
