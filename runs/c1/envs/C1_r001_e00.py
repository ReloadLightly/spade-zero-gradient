import random


class JewelersBalanceEnv:
    """Find the one counterfeit gem (and whether it's heavy or light) via balance-scale weighings."""

    GEMS = ["AMBER", "BERYL", "CITRINE", "DIAMOND", "EMERALD", "FLUORITE"]
    MAX_STEPS = 10
    WEIGH_BUDGET_REWARD = 0.6
    GUESS_REWARD = 0.4

    def __init__(self):
        self.rng = None
        self.fake_index = None
        self.fake_dir = None
        self.candidates = set()
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.fake_index = self.rng.randrange(len(self.GEMS))
        self.fake_dir = self.rng.choice(["HEAVIER", "LIGHTER"])
        self.candidates = {
            (i, d) for i in range(len(self.GEMS)) for d in ("HEAVIER", "LIGHTER")
        }
        self.steps = 0
        self.done = False

        gem_list = ", ".join(self.GEMS)
        obs = (
            "You are a gem appraiser. Among six gems ({}), exactly one is a "
            "counterfeit: it weighs either HEAVIER or LIGHTER than the five "
            "genuine gems (which all weigh identically). You do not know "
            "which gem or which direction.\n\n"
            "You have a two-pan balance scale. Actions (you have {} steps total):\n"
            "  WEIGH <left_group> <right_group> -- comma-separated gem names, "
            "no spaces inside a group, equal group sizes, no overlap. "
            "Example: WEIGH AMBER,BERYL CITRINE,DIAMOND\n"
            "  GUESS <gem> <HEAVIER|LIGHTER> -- your final verdict, ends the episode.\n\n"
            "Each WEIGH returns LEFT_HEAVIER, RIGHT_HEAVIER, or EQUAL. "
            "Gems left off the scale still matter -- an EQUAL result tells you "
            "something about them too. Reward comes from how much you narrow "
            "the true possibilities and from a correct final GUESS."
        ).format(gem_list, self.MAX_STEPS)
        return obs, {}

    def _gem_index(self, name):
        name = name.strip().upper()
        if name in self.GEMS:
            return self.GEMS.index(name)
        return None

    def _result_for(self, gem_idx, direction, left_idx, right_idx):
        if gem_idx in left_idx:
            return "LEFT_HEAVIER" if direction == "HEAVIER" else "RIGHT_HEAVIER"
        elif gem_idx in right_idx:
            return "RIGHT_HEAVIER" if direction == "HEAVIER" else "LEFT_HEAVIER"
        else:
            return "EQUAL"

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        parts = text.split()

        if not parts:
            obs = "Empty action. Use WEIGH <left> <right> or GUESS <gem> <direction>."
            return self._maybe_truncate(obs, 0.0)

        cmd = parts[0].upper()

        if cmd == "WEIGH":
            if len(parts) != 3:
                obs = (
                    "Malformed WEIGH. Format: WEIGH <left_group> <right_group>, "
                    "e.g. WEIGH AMBER,BERYL CITRINE,DIAMOND"
                )
                return self._maybe_truncate(obs, 0.0)

            left_names = [n for n in parts[1].split(",") if n]
            right_names = [n for n in parts[2].split(",") if n]
            left_idx = [self._gem_index(n) for n in left_names]
            right_idx = [self._gem_index(n) for n in right_names]

            if (
                not left_names or not right_names
                or len(left_idx) != len(left_names) or None in left_idx
                or len(right_idx) != len(right_names) or None in right_idx
                or len(left_idx) != len(right_idx)
                or set(left_idx) & set(right_idx)
                or len(set(left_idx)) != len(left_idx)
                or len(set(right_idx)) != len(right_idx)
            ):
                obs = (
                    "Invalid weighing: groups must use real gem names, be "
                    "equal-sized, and not share a gem. Try again."
                )
                return self._maybe_truncate(obs, 0.0)

            left_idx_set, right_idx_set = set(left_idx), set(right_idx)
            actual_result = self._result_for(
                self.fake_index, self.fake_dir, left_idx_set, right_idx_set
            )

            before = len(self.candidates)
            self.candidates = {
                (i, d) for (i, d) in self.candidates
                if self._result_for(i, d, left_idx_set, right_idx_set) == actual_result
            }
            after = len(self.candidates)
            reward = self.WEIGH_BUDGET_REWARD * (before - after) / 11.0

            obs = (
                "Scale result: {}. {} gem/direction possibilities remain consistent."
            ).format(actual_result, after)
            return self._maybe_truncate(obs, reward)

        elif cmd == "GUESS":
            if len(parts) != 3 or parts[2].upper() not in ("HEAVIER", "LIGHTER"):
                obs = "Malformed GUESS. Format: GUESS <gem> <HEAVIER|LIGHTER>"
                return self._maybe_truncate(obs, 0.0)

            gem_idx = self._gem_index(parts[1])
            direction = parts[2].upper()
            if gem_idx is None:
                obs = "Unknown gem name in GUESS. Use one of: " + ", ".join(self.GEMS)
                return self._maybe_truncate(obs, 0.0)

            self.done = True
            if gem_idx == self.fake_index and direction == self.fake_dir:
                obs = "Correct! {} was the counterfeit and it was {}.".format(
                    self.GEMS[gem_idx], direction
                )
                return obs, self.GUESS_REWARD, True, False, {}
            else:
                obs = "Incorrect. The counterfeit was {} ({}).".format(
                    self.GEMS[self.fake_index], self.fake_dir
                )
                return obs, 0.0, True, False, {}

        else:
            obs = "Unrecognized command. Use WEIGH <left> <right> or GUESS <gem> <direction>."
            return self._maybe_truncate(obs, 0.0)

    def _maybe_truncate(self, obs, reward):
        if self.steps >= self.MAX_STEPS:
            self.done = True
            return obs + " Step limit reached.", reward, False, True, {}
        return obs, reward, False, False, {}
