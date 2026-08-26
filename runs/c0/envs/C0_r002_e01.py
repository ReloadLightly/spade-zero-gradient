import math
import random
import re


class PlanetaryConjunctionEnv:
    """Deduce hidden planetary periods from aggregate alignment counts,
    then report the first day all three planets are simultaneously home."""

    PERIOD_RANGE = (2, 6)  # inclusive; each hidden period drawn from here
    MAX_QUERY_DAY = 999
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.periods = []
        self.true_lcm = 0
        self.step_count = 0
        self.seen_counts = set()
        self.novel_rewards_given = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        lo, hi = self.PERIOD_RANGE
        self.periods = self.rng.sample(range(lo, hi + 1), 3)
        self.true_lcm = math.lcm(*self.periods)
        self.step_count = 0
        self.seen_counts = set()
        self.novel_rewards_given = 0
        self.done = False

        obs = (
            "Three newly charted planets each return to their home position "
            "every P days, where each P is a hidden whole number between "
            f"{lo} and {hi} (inclusive), and no two planets share the same P.\n"
            "You have two available actions:\n"
            "  PROBE <day>   - day is a positive integer; learn how many of "
            "the 3 planets (0, 1, 2, or 3) are at their home position on that day.\n"
            "  ANSWER <day>  - state the smallest day > 0 on which all 3 "
            "planets are simultaneously home. This ends the episode.\n"
            f"You have {self.MAX_STEPS} total actions before the episode ends. "
            "Malformed actions still consume a step."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        truncated_after = self.step_count >= self.MAX_STEPS
        text = (action or "").strip()

        m_probe = re.match(r"^PROBE\s+(-?\d+)\s*$", text, re.IGNORECASE)
        m_answer = re.match(r"^ANSWER\s+(-?\d+)\s*$", text, re.IGNORECASE)

        if m_probe:
            day = int(m_probe.group(1))
            if day < 1 or day > self.MAX_QUERY_DAY:
                obs = (
                    f"Invalid PROBE day '{day}'. Day must be an integer "
                    f"between 1 and {self.MAX_QUERY_DAY}."
                )
                terminated, truncated = False, truncated_after
                if truncated:
                    self.done = True
                return obs, 0.0, terminated, truncated, {}

            count = sum(1 for p in self.periods if day % p == 0)
            reward = 0.0
            if count not in self.seen_counts:
                self.seen_counts.add(count)
                if self.novel_rewards_given < 2:
                    reward = 0.2
                    self.novel_rewards_given += 1

            obs = f"Day {day}: {count} of the 3 planets are at their home position."
            terminated, truncated = False, truncated_after
            if truncated:
                self.done = True
                obs += " Step limit reached; episode truncated."
            return obs, reward, terminated, truncated, {}

        if m_answer:
            day = int(m_answer.group(1))
            self.done = True
            if day < 1 or day > self.MAX_QUERY_DAY:
                obs = (
                    f"Invalid ANSWER day '{day}'. Episode ended without a "
                    f"valid answer. Hidden periods were {sorted(self.periods)}; "
                    f"true first conjunction was day {self.true_lcm}."
                )
                return obs, 0.0, True, False, {}

            correct = day == self.true_lcm
            reward = 0.6 if correct else 0.0
            if correct:
                obs = (
                    f"Correct! Day {day} is the first day all 3 planets are "
                    f"simultaneously home. Hidden periods were "
                    f"{sorted(self.periods)}."
                )
            else:
                obs = (
                    f"Incorrect. You answered {day}; the true first "
                    f"conjunction was day {self.true_lcm}. Hidden periods "
                    f"were {sorted(self.periods)}."
                )
            return obs, reward, True, False, {}

        obs = (
            f"Unrecognized action '{text}'. Use 'PROBE <day>' or "
            "'ANSWER <day>' with a positive integer day."
        )
        terminated, truncated = False, truncated_after
        if truncated:
            self.done = True
        return obs, 0.0, terminated, truncated, {}
