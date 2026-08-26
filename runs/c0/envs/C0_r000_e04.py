import random
import re


class FractalBranchEnv:
    def __init__(self):
        self.rng = None
        self.family = None
        self.params = {}
        self.revealed = {}
        self.max_reveal_n = 1
        self.max_grows = 3
        self.grows_used = 0
        self.checkpoint_n = 5
        self.target_n = 7
        self.checkpoint_used = False
        self.steps = 0
        self.max_steps = 10
        self.done = False

    def _compute(self, n):
        fam = self.family
        p = self.params
        if fam == "arithmetic":
            return p["b0"] + n * p["d"]
        if fam == "geometric":
            return p["b0"] * (p["r"] ** n)
        if fam == "fibonacci":
            a, b = p["b0"], p["b1"]
            if n == 0:
                return a
            if n == 1:
                return b
            for _ in range(n - 1):
                a, b = b, a + b
            return b
        if fam == "quadratic":
            return p["b0"] + p["c"] * n * (n + 1) // 2
        raise ValueError("unknown family")

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.family = self.rng.choice(
            ["arithmetic", "geometric", "fibonacci", "quadratic"]
        )
        if self.family == "arithmetic":
            self.params = {"b0": self.rng.randint(2, 6), "d": self.rng.randint(2, 5)}
        elif self.family == "geometric":
            self.params = {"b0": self.rng.randint(2, 4), "r": self.rng.randint(2, 3)}
        elif self.family == "fibonacci":
            b0 = self.rng.randint(2, 4)
            b1 = self.rng.randint(b0 + 1, b0 + 4)
            self.params = {"b0": b0, "b1": b1}
        else:
            self.params = {"b0": self.rng.randint(2, 5), "c": self.rng.randint(1, 3)}

        self.revealed = {0: self._compute(0), 1: self._compute(1)}
        self.max_reveal_n = 1
        self.grows_used = 0
        self.checkpoint_used = False
        self.steps = 0
        self.done = False

        obs = (
            "A fractal plant's branch count grows by a fixed, consistent rule from "
            "generation to generation.\n"
            f"Generation 0: {self.revealed[0]} branches.\n"
            f"Generation 1: {self.revealed[1]} branches.\n"
            "Goal: determine the branch count at generation 7.\n"
            "Actions (send exactly one per turn):\n"
            "  GROW               - reveal the next generation's branch count "
            "(usable up to 3 times, revealing generations 2, 3, then 4)\n"
            "  CHECK <value>      - usable once, at any time: predict the branch "
            "count at generation 5 before it is otherwise revealed (worth 0.3 "
            "reward); the true value is shown afterward either way\n"
            "  PREDICT <value>    - submit your final answer for generation 7's "
            "branch count (worth 0.7 reward) and end the episode\n"
            "You have at most 10 turns total."
        )
        info = {"family": self.family}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        tokens = text.split()
        cmd = tokens[0].upper() if tokens else ""

        reward = 0.0
        terminated = False
        obs = ""

        if cmd == "GROW":
            if self.grows_used >= self.max_grows:
                obs = (
                    "No further growth is available (limit of 3 reached). "
                    "Use CHECK or PREDICT."
                )
            else:
                self.grows_used += 1
                next_n = self.max_reveal_n + 1
                value = self._compute(next_n)
                self.revealed[next_n] = value
                self.max_reveal_n = next_n
                obs = f"Generation {next_n}: {value} branches."

        elif cmd == "CHECK":
            if len(tokens) < 2 or not re.fullmatch(r"-?\d+", tokens[1]):
                obs = "Malformed CHECK. Use: CHECK <integer>"
            elif self.checkpoint_used:
                obs = "You already used your one checkpoint prediction."
            else:
                self.checkpoint_used = True
                guess = int(tokens[1])
                true_val = self._compute(self.checkpoint_n)
                correct = guess == true_val
                reward = 0.3 if correct else 0.0
                verdict = "correct" if correct else "incorrect"
                obs = (
                    f"Checkpoint prediction {verdict}. "
                    f"Generation {self.checkpoint_n}: {true_val} branches."
                )

        elif cmd == "PREDICT":
            if len(tokens) < 2 or not re.fullmatch(r"-?\d+", tokens[1]):
                obs = "Malformed PREDICT. Use: PREDICT <integer>"
            else:
                guess = int(tokens[1])
                true_val = self._compute(self.target_n)
                correct = guess == true_val
                reward = 0.7 if correct else 0.0
                terminated = True
                self.done = True
                verdict = "Correct!" if correct else "Incorrect."
                obs = (
                    f"{verdict} Generation {self.target_n} actually has "
                    f"{true_val} branches. Episode complete."
                )

        else:
            obs = (
                "Unrecognized action. Valid commands: GROW, CHECK <integer>, "
                "PREDICT <integer>."
            )

        truncated = False
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            self.done = True
            obs += " Turn limit reached; episode ends without a final prediction."

        info = {
            "family": self.family,
            "grows_used": self.grows_used,
            "checkpoint_used": self.checkpoint_used,
        }
        return obs, reward, terminated, truncated, info
