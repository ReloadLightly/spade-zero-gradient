import random


class MirrorRotateSymmetryEnv:
    ALPHABET = "abcdefgh"
    LENGTH = 6
    TRANSFORMS = ["mirror", "rotate_half", "swap_pairs", "alpha_reflect", "reverse_alpha_reflect"]
    MAX_PEEKS = 4
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.secret_name = None
        self.reference = None
        self.target = None
        self.witness = None
        self.target_answer = None
        self.peeked = {}
        self.consistent = set()
        self.best_consistent_size = None
        self.steps = 0
        self.done = False

    def _apply(self, name, s):
        n = len(s)
        lo = ord(self.ALPHABET[0])
        hi = ord(self.ALPHABET[-1])
        if name == "mirror":
            return s[::-1]
        if name == "rotate_half":
            h = n // 2
            return s[h:] + s[:h]
        if name == "swap_pairs":
            chars = list(s)
            for i in range(0, n - 1, 2):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return "".join(chars)
        if name == "alpha_reflect":
            return "".join(chr(lo + hi - ord(c)) for c in s)
        if name == "reverse_alpha_reflect":
            reflected = "".join(chr(lo + hi - ord(c)) for c in s)
            return reflected[::-1]
        raise ValueError(name)

    def _rand_string(self, distinct):
        if distinct:
            letters = list(self.ALPHABET)
            self.rng.shuffle(letters)
            return "".join(letters[: self.LENGTH])
        return "".join(self.rng.choice(self.ALPHABET) for _ in range(self.LENGTH))

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.secret_name = self.rng.choice(self.TRANSFORMS)
        self.reference = self._rand_string(distinct=True)
        while True:
            self.target = self._rand_string(distinct=False)
            if self.target != self.reference:
                break
        self.witness = self._apply(self.secret_name, self.reference)
        self.target_answer = self._apply(self.secret_name, self.target)
        self.peeked = {}
        self.consistent = set(self.TRANSFORMS)
        self.best_consistent_size = len(self.consistent)
        self.steps = 0
        self.done = False

        obs = (
            "SYMMETRY LAB. A hidden transformation T (one of 5 fixed symmetry "
            "operations on 6-letter strings over alphabet a-h) was applied to "
            f"reference string R = '{self.reference}' to produce a hidden witness "
            "string W = T(R). You cannot see W directly.\n"
            f"Goal: determine T well enough to predict T(TARGET) for the held-out "
            f"string TARGET = '{self.target}'.\n"
            "Actions (exactly one per turn):\n"
            f"  PEEK:<index>    reveal W[index] for index in 0..{self.LENGTH-1} "
            f"(each index usable once, at most {self.MAX_PEEKS} peeks total).\n"
            "  ANSWER:<string> submit your predicted 6-letter T(TARGET); ends the episode.\n"
            f"You have {self.MAX_STEPS} turns total."
        )
        info = {"reference": self.reference, "target": self.target}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if action.upper().startswith("PEEK:"):
            if len(self.peeked) >= self.MAX_PEEKS:
                obs = f"Peek budget ({self.MAX_PEEKS}) exhausted. Submit ANSWER:<string> instead."
            else:
                idx_str = action[5:].strip()
                if not idx_str.lstrip("-").isdigit():
                    obs = f"Malformed PEEK: '{idx_str}' is not an integer. Use PEEK:<0-{self.LENGTH-1}>."
                else:
                    idx = int(idx_str)
                    if idx < 0 or idx >= self.LENGTH:
                        obs = f"Index {idx} out of range 0..{self.LENGTH-1}."
                    elif idx in self.peeked:
                        obs = f"Index {idx} already peeked: W[{idx}] = '{self.peeked[idx]}'. Choose a new index."
                    else:
                        revealed = self.witness[idx]
                        self.peeked[idx] = revealed
                        self.consistent = {
                            name for name in self.consistent
                            if self._apply(name, self.reference)[idx] == revealed
                        }
                        obs = (
                            f"W[{idx}] = '{revealed}'. Consistent-hypothesis count: "
                            f"{len(self.consistent)} of {len(self.TRANSFORMS)}."
                        )
                        if len(self.consistent) < self.best_consistent_size:
                            reward += 0.1
                            self.best_consistent_size = len(self.consistent)
        elif action.upper().startswith("ANSWER:"):
            guess = action[7:].strip().lower()
            terminated = True
            self.done = True
            if guess == self.target_answer:
                reward += 0.6
                obs = f"Correct. T(TARGET) = '{self.target_answer}'. Episode solved."
            else:
                obs = (
                    f"Incorrect. You answered '{guess}', the true value was "
                    f"'{self.target_answer}'. Episode over."
                )
        else:
            obs = "Malformed action. Use 'PEEK:<index>' or 'ANSWER:<string>'."

        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode truncated."

        return obs, reward, terminated, truncated, info
