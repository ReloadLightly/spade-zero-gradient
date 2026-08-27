import re


class GemBalanceEnv:
    WEIGH_RE = re.compile(r'^\s*WEIGH\s+([0-9,\s]+)\s+VS\s+([0-9,\s]+)\s*$', re.IGNORECASE)
    GUESS_RE = re.compile(r'^\s*GUESS\s+(\d+)\s+(HEAVIER|LIGHTER)\s*$', re.IGNORECASE)
    MAX_STEPS = 10
    N = 8
    INFORMATIVE_CAP = 3
    INFORMATIVE_REWARD = 0.1
    GUESS_REWARD = 0.7

    def __init__(self):
        self.rng = None
        self.gems = list(range(1, self.N + 1))
        self.fake = None
        self.direction = None
        self.candidates = set()
        self.step_count = 0
        self.informative_used = 0
        self.done = False

    def reset(self, seed=None):
        import random
        self.rng = random.Random(seed)
        self.fake = self.rng.randrange(self.N) + 1
        self.direction = self.rng.choice(['heavier', 'lighter'])
        self.candidates = {(g, d) for g in self.gems for d in ('heavier', 'lighter')}
        self.step_count = 0
        self.informative_used = 0
        self.done = False
        obs = (
            "GEM AUTHENTICATION. There are 8 gems labeled 1-8. Exactly one is a "
            "counterfeit whose true weight differs from all the others; it may be "
            "HEAVIER or LIGHTER than genuine gems, but you do not know which, nor "
            "which gem it is. You have a balance scale.\n"
            "Actions (send exactly one per turn, plain text):\n"
            "  WEIGH <left> VS <right>   e.g. WEIGH 1,2,3 VS 4,5,6\n"
            "    (left/right are comma-separated gem labels, equal counts, no repeats,\n"
            "     no gem on both sides). Reply reports LEFT heavier, RIGHT heavier, or BALANCED.\n"
            "  GUESS <gem> <HEAVIER|LIGHTER>   e.g. GUESS 5 LIGHTER\n"
            "    This is a final, one-shot answer: it ends the episode immediately.\n"
            "You have at most 10 total actions. Weigh first; guess only once you are certain."
        )
        return obs, {}

    def _simulate(self, fake_gem, fake_dir, left, right):
        def w(x):
            if x == fake_gem:
                return 1 if fake_dir == 'heavier' else -1
            return 0
        l = sum(w(x) for x in left)
        r = sum(w(x) for x in right)
        if l > r:
            return 'LEFT heavier'
        if r > l:
            return 'RIGHT heavier'
        return 'BALANCED'

    def _parse_weigh(self, m):
        left_raw, right_raw = m.group(1), m.group(2)
        try:
            left = [int(x) for x in left_raw.split(',') if x.strip() != '']
            right = [int(x) for x in right_raw.split(',') if x.strip() != '']
        except ValueError:
            return None
        if not left or not right:
            return None
        if len(left) != len(right):
            return None
        if len(set(left)) != len(left) or len(set(right)) != len(right):
            return None
        if set(left) & set(right):
            return None
        if any(g not in self.gems for g in left + right):
            return None
        return left, right

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        action = action if isinstance(action, str) else str(action)
        wm = self.WEIGH_RE.match(action)
        gm = None if wm else self.GUESS_RE.match(action)

        if wm:
            parsed = self._parse_weigh(wm)
            if parsed is None:
                obs = (
                    "Malformed WEIGH action. Use: WEIGH <left> VS <right> with equal-size, "
                    "disjoint, comma-separated gem labels from 1-8, e.g. WEIGH 1,2 VS 3,4."
                )
            else:
                left, right = parsed
                outcome = self._simulate(self.fake, self.direction, left, right)
                before = len(self.candidates)
                self.candidates = {
                    (g, d) for (g, d) in self.candidates
                    if self._simulate(g, d, left, right) == outcome
                }
                after = len(self.candidates)
                if after < before and self.informative_used < self.INFORMATIVE_CAP:
                    reward = self.INFORMATIVE_REWARD
                    self.informative_used += 1
                obs = (
                    f"Result: {outcome}. Consistent (gem, direction) hypotheses "
                    f"remaining: {after}."
                )
                info['candidates_remaining'] = after
        elif gm:
            guess_gem = int(gm.group(1))
            guess_dir = gm.group(2).lower()
            terminated = True
            self.done = True
            if guess_gem == self.fake and guess_dir == self.direction:
                reward = self.GUESS_REWARD
                obs = (
                    f"Correct! Gem {guess_gem} is the counterfeit and it is "
                    f"{guess_dir}. You win."
                )
            else:
                reward = 0.0
                obs = (
                    f"Incorrect: gem {guess_gem} being {guess_dir} is not "
                    "consistent with the evidence gathered. Episode over."
                )
        else:
            obs = (
                "Malformed action. Use 'WEIGH <left> VS <right>' (e.g. WEIGH 1,2 VS 3,4) "
                "or 'GUESS <gem> <HEAVIER|LIGHTER>' (e.g. GUESS 5 LIGHTER)."
            )

        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode over."

        return obs, reward, terminated, truncated, info
