import re
import random


class AuthenticGemBalanceEnv:
    """Balance-scale deduction: find the one authentic gem among N fakes."""

    N = 7
    MAX_STEPS = 10
    FAKE_WEIGHT = 10
    OFFSET = 3
    THRESHOLDS = (7, 3, 1)

    WEIGH_RE = re.compile(r'^WEIGH\s+([0-9,\s]+)\s+VS\s+([0-9,\s]+)$')
    GUESS_RE = re.compile(r'^GUESS\s+(\d+)$')

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.authentic_index = self.rng.randint(1, self.N)
        self.direction = self.rng.choice([1, -1])
        self.candidates = {(i, d) for i in range(1, self.N + 1) for d in (1, -1)}
        self.pending_thresholds = list(self.THRESHOLDS)
        self.step_count = 0
        self.finished = False

        obs = (
            f"You must identify the one AUTHENTIC gem among {self.N} gems "
            f"numbered 1..{self.N}. The other {self.N - 1} gems are identical "
            f"fakes of equal weight. The authentic gem weighs a different "
            f"amount than the fakes -- it may be HEAVIER or LIGHTER, and you "
            f"do not know which.\n"
            f"You have a two-pan balance scale. Actions:\n"
            f"  WEIGH <left indices> VS <right indices>  -- e.g. 'WEIGH 1,2 VS 3,4'. "
            f"The two groups must be disjoint, non-empty, and the SAME size "
            f"(1 to {self.N // 2} gems per side). Result is one of "
            f"LEFT_HEAVIER, RIGHT_HEAVIER, BALANCED.\n"
            f"  GUESS <index>  -- name the authentic gem's index. This ends the episode.\n"
            f"You have {self.MAX_STEPS} total actions. Malformed actions still "
            f"consume a step."
        )
        return obs, {"num_gems": self.N, "step_limit": self.MAX_STEPS}

    def _weight(self, i, idx, d):
        return self.FAKE_WEIGHT + (d * self.OFFSET if i == idx else 0)

    def _true_weight(self, i):
        return self._weight(i, self.authentic_index, self.direction)

    def _outcome(self, left, right, idx, d):
        lsum = sum(self._weight(i, idx, d) for i in left)
        rsum = sum(self._weight(i, idx, d) for i in right)
        if lsum > rsum:
            return "LEFT_HEAVIER"
        if rsum > lsum:
            return "RIGHT_HEAVIER"
        return "BALANCED"

    def _parse_group(self, text):
        parts = [p for p in re.split(r'[,\s]+', text.strip()) if p]
        return [int(p) for p in parts]

    def step(self, action):
        if self.finished:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip().upper()
        reward = 0.0
        terminated = False

        gm = self.GUESS_RE.match(text)
        wm = self.WEIGH_RE.match(text)

        if gm:
            guess = int(gm.group(1))
            if guess == self.authentic_index:
                reward = 0.7
                obs = f"Correct! Gem {guess} is authentic."
                info = {"success": True}
            else:
                reward = 0.0
                obs = f"Incorrect. Gem {guess} is not authentic."
                info = {"success": False}
            self.finished = True
            terminated = True
            truncated = False
            return obs, reward, terminated, truncated, info

        elif wm:
            try:
                left = self._parse_group(wm.group(1))
                right = self._parse_group(wm.group(2))
            except ValueError:
                left = right = None

            valid = (
                left is not None and right is not None
                and len(left) == len(right)
                and 1 <= len(left) <= self.N // 2
                and len(set(left)) == len(left)
                and len(set(right)) == len(right)
                and not (set(left) & set(right))
                and all(1 <= i <= self.N for i in left + right)
            )

            if not valid:
                obs = (
                    "Invalid weighing: groups must be disjoint, non-empty, "
                    f"equal size (max {self.N // 2} each), with indices in "
                    f"1..{self.N}. No result recorded."
                )
            else:
                outcome = self._outcome(
                    left, right, self.authentic_index, self.direction
                )
                new_candidates = {
                    (idx, d) for (idx, d) in self.candidates
                    if self._outcome(left, right, idx, d) == outcome
                }
                self.candidates = new_candidates

                for t in list(self.pending_thresholds):
                    if len(self.candidates) <= t:
                        reward += 0.1
                        self.pending_thresholds.remove(t)

                obs = f"Result: {outcome}."
        else:
            obs = (
                "Unrecognized action. Use 'WEIGH a,b VS c,d' or 'GUESS <index>'."
            )

        truncated = self.step_count >= self.MAX_STEPS
        if truncated:
            self.finished = True
            obs += " Step limit reached; episode over."

        return obs, reward, False, truncated, {}
