import random


class BalanceGemEnv:
    def __init__(self):
        self.n = 8
        self.max_steps = 8

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.fake_index = self.rng.randrange(self.n)
        self.fake_dir = self.rng.choice(['H', 'L'])
        self.steps = 0
        self.done = False
        self.milestones_awarded = 0
        self.candidates = set(
            (i, d) for i in range(self.n) for d in ('H', 'L')
        )
        info = {}
        obs = (
            "You are a gem authenticator. There are 8 gems, indices 0-7. "
            "Exactly one gem is FAKE: it weighs either strictly more (H) or "
            "strictly less (L) than the other 7 identical genuine gems -- "
            "you do not know which gem, nor which direction.\n"
            "Action format:\n"
            "  WEIGH <left indices> | <right indices>\n"
            "    e.g. put three gems on each side, comma-separated, "
            "separated by a single '|'. Groups must be nonempty, equal "
            "size, and disjoint.\n"
            "  Result is one of LEFT_HEAVIER, RIGHT_HEAVIER, BALANCED.\n"
            "When confident, submit your final answer with:\n"
            "  GUESS <index> <H|L>\n"
            "    naming the fake gem's index and whether it is heavier "
            "or lighter.\n"
            "GUESS ends the episode immediately, right or wrong -- you "
            "get exactly one shot at it.\n"
            f"You have {self.max_steps} total actions (weighings plus "
            "the final guess, combined)."
        )
        return obs, info

    def _malformed(self, msg):
        truncated = self.steps >= self.max_steps
        if truncated:
            self.done = True
        return msg, 0.0, False, truncated, {}

    def _parse_indices(self, s):
        s = s.strip()
        if not s:
            raise ValueError("empty group")
        toks = [t.strip() for t in s.split(',')]
        result = []
        seen = set()
        for t in toks:
            core = t[1:] if t.startswith('-') else t
            if not core.isdigit():
                raise ValueError(f"'{t}' is not an integer")
            v = int(t)
            if not (0 <= v < self.n):
                raise ValueError(f"index {v} out of range 0-{self.n - 1}")
            if v in seen:
                raise ValueError(f"duplicate index {v} in same group")
            seen.add(v)
            result.append(v)
        return result

    def _predict(self, i, d, left, right):
        off = 1 if d == 'H' else -1
        l = off if i in left else 0
        r = off if i in right else 0
        if l > r:
            return "LEFT_HEAVIER"
        if l < r:
            return "RIGHT_HEAVIER"
        return "BALANCED"

    def _handle_weigh(self, rest):
        parts = rest.split('|')
        if len(parts) != 2:
            return self._malformed(
                "WEIGH needs exactly one '|' separating two groups, "
                "e.g. left indices, a '|', then right indices."
            )
        try:
            left = self._parse_indices(parts[0])
            right = self._parse_indices(parts[1])
        except ValueError as e:
            return self._malformed(f"Bad indices: {e}")
        if not left or not right:
            return self._malformed("Both groups must be nonempty.")
        if len(left) != len(right):
            return self._malformed("Both groups must be the same size.")
        if set(left) & set(right):
            return self._malformed(
                "Groups must be disjoint -- no gem on both sides."
            )

        outcome = self._predict(self.fake_index, self.fake_dir, left, right)

        new_candidates = {
            c for c in self.candidates
            if self._predict(c[0], c[1], left, right) == outcome
        }
        reward = 0.0
        if len(new_candidates) < len(self.candidates) and self.milestones_awarded < 3:
            reward = 0.15
            self.milestones_awarded += 1
        self.candidates = new_candidates

        truncated = self.steps >= self.max_steps
        if truncated:
            self.done = True
        obs = (
            f"Result: {outcome}. Consistent (gem, direction) possibilities "
            f"remaining: {len(self.candidates)}."
        )
        if truncated:
            obs += " Step limit reached without a guess -- episode over."
        return obs, reward, False, truncated, {}

    def _handle_guess(self, rest):
        tokens = rest.split()
        if len(tokens) != 2:
            return self._malformed(
                "GUESS needs exactly an index and a direction, "
                "e.g. an index then H or L."
            )
        idx_tok, dir_tok = tokens
        core = idx_tok[1:] if idx_tok.startswith('-') else idx_tok
        if not core.isdigit():
            return self._malformed("Index must be an integer 0-7.")
        idx = int(idx_tok)
        if not (0 <= idx < self.n):
            return self._malformed(f"Index must be between 0 and {self.n - 1}.")
        dtok = dir_tok.upper()
        if dtok in ("H", "HEAVIER"):
            d = "H"
        elif dtok in ("L", "LIGHTER"):
            d = "L"
        else:
            return self._malformed("Direction must be H (heavier) or L (lighter).")

        self.done = True
        correct = (idx == self.fake_index and d == self.fake_dir)
        true_word = "heavier" if self.fake_dir == "H" else "lighter"
        if correct:
            reward = 0.55
            obs = f"Correct! Gem {idx} is fake and {true_word}. Episode complete."
        else:
            reward = 0.0
            obs = (
                f"Incorrect guess. The true fake was gem {self.fake_index} "
                f"({true_word}). Episode complete."
            )
        return obs, reward, True, False, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.steps += 1
        text = (action or "").strip()
        upper = text.upper()
        if upper.startswith("WEIGH"):
            return self._handle_weigh(text[5:].strip())
        elif upper.startswith("GUESS"):
            return self._handle_guess(text[5:].strip())
        else:
            truncated = self.steps >= self.max_steps
            if truncated:
                self.done = True
            return (
                "Malformed action. Use 'WEIGH a,b,c | d,e,f' or "
                "'GUESS <index> <H|L>'.",
                0.0, False, truncated, {}
            )
