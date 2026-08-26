import math
import random
import re


class CounterfeitGemBalanceEnv:
    N = 9
    MAX_STEPS = 10
    SCORED_WEIGHINGS = 3

    def __init__(self):
        self.rng = None
        self.fake_gem = None
        self.fake_dir = None
        self.step_count = 0
        self.weigh_count = 0
        self.candidates = []
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.fake_gem = self.rng.randrange(self.N)
        self.fake_dir = self.rng.choice(['HEAVIER', 'LIGHTER'])
        self.step_count = 0
        self.weigh_count = 0
        self.done = False
        self.candidates = [(g, d) for g in range(self.N) for d in ('HEAVIER', 'LIGHTER')]
        return self._opening_observation(), {}

    def _opening_observation(self):
        return (
            f"You examine {self.N} gems, labeled 0-{self.N - 1}, laid on a table. "
            "Exactly one is counterfeit: it looks identical to the rest but is "
            "either slightly HEAVIER or slightly LIGHTER than the others (you do "
            "not know which gem, or which direction). All genuine gems weigh "
            "exactly the same as each other.\n"
            "You have a two-pan balance. Actions:\n"
            "  WEIGH a,b,...|x,y,...  -- place the left-of-'|' gems on the left "
            "pan and the right-of-'|' gems on the right pan (equal counts each "
            "side, no gem repeated within one weighing). Returns LEFT, RIGHT, "
            "or BALANCED.\n"
            "  SUBMIT g,DIRECTION      -- declare gem g is the counterfeit and "
            "DIRECTION is HEAVIER or LIGHTER. Ends the episode.\n"
            f"You have {self.MAX_STEPS} steps total. Only your first "
            f"{self.SCORED_WEIGHINGS} weighings earn partial credit, so make "
            "them count before you submit."
        )

    def _parse_weigh(self, arg):
        if '|' not in arg:
            return None
        left_s, right_s = arg.split('|', 1)
        left = [int(x) for x in re.findall(r'\d+', left_s)]
        right = [int(x) for x in re.findall(r'\d+', right_s)]
        if not left or not right or len(left) != len(right):
            return None
        combined = left + right
        if len(set(combined)) != len(combined):
            return None
        if any(i < 0 or i >= self.N for i in combined):
            return None
        return left, right

    def _outcome(self, gem, direction, left, right):
        if gem in left:
            return 'LEFT' if direction == 'HEAVIER' else 'RIGHT'
        if gem in right:
            return 'RIGHT' if direction == 'HEAVIER' else 'LEFT'
        return 'BALANCED'

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        upper = text.upper()

        if upper.startswith('WEIGH'):
            parsed = self._parse_weigh(text[5:].strip())
            if parsed is None:
                obs = ("Malformed WEIGH. Use: WEIGH a,b,...|x,y,... with equal, "
                       "distinct gem indices on each side, no gem on both sides.")
                return self._maybe_truncate(obs, 0.0)

            left, right = parsed
            actual_outcome = self._outcome(self.fake_gem, self.fake_dir, left, right)
            prev_count = len(self.candidates)
            self.candidates = [
                (g, d) for (g, d) in self.candidates
                if self._outcome(g, d, left, right) == actual_outcome
            ]
            reward = 0.0
            if self.weigh_count < self.SCORED_WEIGHINGS:
                threshold = math.ceil(prev_count / 3)
                if len(self.candidates) <= threshold:
                    reward = 0.2
            self.weigh_count += 1

            obs = (
                f"Result: {actual_outcome}. Gem-direction pairs still consistent "
                f"with all evidence so far: {len(self.candidates)} (was {prev_count})."
            )
            return self._maybe_truncate(obs, reward)

        if upper.startswith('SUBMIT'):
            m = re.search(r'(\d+)\s*,\s*(HEAVIER|LIGHTER)', upper)
            if not m:
                obs = "Malformed SUBMIT. Use: SUBMIT g,HEAVIER or SUBMIT g,LIGHTER."
                return self._maybe_truncate(obs, 0.0)
            gem = int(m.group(1))
            direction = m.group(2)
            self.done = True
            if gem == self.fake_gem and direction == self.fake_dir:
                obs = f"Correct. Gem {gem} was counterfeit and {direction}."
                return obs, 0.4, True, False, {}
            if gem == self.fake_gem:
                obs = (f"Gem {gem} was indeed counterfeit, but it was "
                       f"{self.fake_dir}, not {direction}.")
                return obs, 0.1, True, False, {}
            obs = (f"Wrong. Gem {gem} was genuine; the counterfeit was "
                   f"gem {self.fake_gem} ({self.fake_dir}).")
            return obs, 0.0, True, False, {}

        obs = "Unrecognized action. Use WEIGH a,b,...|x,y,... or SUBMIT g,DIRECTION."
        return self._maybe_truncate(obs, 0.0)

    def _maybe_truncate(self, obs, reward):
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            return obs + " Step limit reached.", reward, False, True, {}
        return obs, reward, False, False, {}
