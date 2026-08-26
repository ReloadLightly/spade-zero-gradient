import random
import re
import math


class GemBalanceEnv:
    GEMS = "ABCDEFGH"

    WEIGH_RE = re.compile(r'^WEIGH\s+([A-Za-z]+)\s+([A-Za-z]+)$', re.IGNORECASE)
    GUESS_RE = re.compile(r'^GUESS\s+([A-Za-z])\s+(HEAVIER|LIGHTER)$', re.IGNORECASE)

    def __init__(self):
        self.rng = None
        self.fake_gem = None
        self.direction = None
        self.step_count = 0
        self.remaining = set()
        self.total_hypotheses = 0
        self.terminated = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.fake_gem = self.rng.choice(self.GEMS)
        self.direction = self.rng.choice(['HEAVIER', 'LIGHTER'])
        self.step_count = 0
        self.remaining = set((g, d) for g in self.GEMS for d in ('HEAVIER', 'LIGHTER'))
        self.total_hypotheses = len(self.remaining)
        self.terminated = False

        observation = (
            "You are a gem authenticator with 8 gems labeled A B C D E F G H. "
            "Exactly one gem is counterfeit: it weighs slightly more OR slightly less "
            "than the other seven identical genuine gems, but you do not know which gem "
            "it is or whether it is heavier or lighter.\n"
            "Each turn, send exactly one action:\n"
            "  WEIGH <left> <right>  -- put disjoint, equal-size groups of gem letters on "
            "the two pans, e.g. 'WEIGH AB CD'. Reply tells you LEFT_HEAVIER, RIGHT_HEAVIER, "
            "or BALANCED.\n"
            "  GUESS <letter> <HEAVIER|LIGHTER>  -- e.g. 'GUESS F LIGHTER'. Ends the episode.\n"
            "You have at most 10 turns. Weigh strategically, then submit your GUESS."
        )
        return observation, {}

    def _predict(self, gem, direction, left_set, right_set):
        if gem in left_set:
            return 'LEFT_HEAVIER' if direction == 'HEAVIER' else 'RIGHT_HEAVIER'
        if gem in right_set:
            return 'RIGHT_HEAVIER' if direction == 'HEAVIER' else 'LEFT_HEAVIER'
        return 'BALANCED'

    def step(self, action):
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()

        weigh_match = self.WEIGH_RE.match(action)
        guess_match = self.GUESS_RE.match(action)

        if weigh_match:
            left_str = weigh_match.group(1).upper()
            right_str = weigh_match.group(2).upper()
            left_set = set(left_str)
            right_set = set(right_str)

            valid = (
                len(left_str) == len(set(left_str))
                and len(right_str) == len(set(right_str))
                and left_set.issubset(set(self.GEMS))
                and right_set.issubset(set(self.GEMS))
                and len(left_str) == len(right_str)
                and len(left_str) > 0
                and left_set.isdisjoint(right_set)
            )

            if not valid:
                observation = (
                    "Malformed weighing: sides must be non-empty, equal-size, disjoint "
                    "groups of distinct letters from A-H. Example: 'WEIGH AB CD'. Try again."
                )
                truncated = self.step_count >= 10
                if truncated:
                    self.terminated = True
                return observation, 0.0, False, truncated, {}

            outcome = self._predict(self.fake_gem, self.direction, left_set, right_set)

            before = len(self.remaining)
            self.remaining = {
                (g, d) for (g, d) in self.remaining
                if self._predict(g, d, left_set, right_set) == outcome
            }
            after = len(self.remaining)

            reward = 0.4 * (math.log(before) - math.log(after)) / math.log(self.total_hypotheses)
            reward = max(0.0, reward)

            observation = (
                f"Result: {outcome}. {after} (gem, direction) explanations remain "
                f"consistent with everything seen so far."
            )
            truncated = self.step_count >= 10
            if truncated:
                self.terminated = True
            return observation, reward, False, truncated, {}

        if guess_match:
            letter = guess_match.group(1).upper()
            direction = guess_match.group(2).upper()

            if letter not in self.GEMS:
                observation = "Unknown gem letter. Gems are labeled A-H. Try again."
                truncated = self.step_count >= 10
                if truncated:
                    self.terminated = True
                return observation, 0.0, False, truncated, {}

            correct = (letter == self.fake_gem and direction == self.direction)
            reward = 0.6 if correct else 0.0
            self.terminated = True

            if correct:
                observation = f"Correct! Gem {letter} is the counterfeit and is {direction}."
            else:
                observation = (
                    f"Incorrect. Gem {letter} being {direction} does not match the evidence. "
                    f"The episode has ended."
                )
            return observation, reward, True, False, {}

        observation = (
            "Unrecognized action. Use 'WEIGH <left> <right>' or "
            "'GUESS <letter> <HEAVIER|LIGHTER>'."
        )
        truncated = self.step_count >= 10
        if truncated:
            self.terminated = True
        return observation, 0.0, False, truncated, {}
