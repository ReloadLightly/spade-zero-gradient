import math
import random
import re


class TwinOrbitAlmanacEnv:
    """Infer two hidden integer orbital periods from modular residue
    readings, then predict the day of their next joint conjunction."""

    MIN_PERIOD = 4
    MAX_PERIOD = 20
    MIN_DAY = 1
    MAX_DAY = 1000
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.period_a = None
        self.period_b = None
        self.conjunction_day = None
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        candidates = list(range(self.MIN_PERIOD, self.MAX_PERIOD + 1))
        self.period_a = self.rng.choice(candidates)
        remaining = [p for p in candidates if p != self.period_a]
        self.period_b = self.rng.choice(remaining)
        self.conjunction_day = (self.period_a * self.period_b) // math.gcd(
            self.period_a, self.period_b
        )
        self.steps = 0
        self.done = False

        obs = (
            "TWIN ORBIT ALMANAC. Two planets, Aster and Borea, each orbit "
            "with a fixed but hidden whole-number period (in days), each "
            "period between 4 and 20 inclusive, and the two periods are "
            "different. Both planets passed through their Home Mark "
            "together on Day 0 (that was the last conjunction).\n"
            "Goal: determine each planet's exact period and the exact day "
            "of the NEXT time both planets are at their Home Mark "
            "simultaneously (their next conjunction), then submit all "
            "three numbers.\n"
            "Action format (exactly one per turn):\n"
            "  READ <day>   -- inspect both planets on the given day "
            "(1 to 1000). Reports how many days into its current lap each "
            "planet is (0 means it is exactly at its Home Mark).\n"
            "  ANSWER <period_a> <period_b> <day> -- submit your final "
            "answer: Aster's period, Borea's period, and the day of the "
            "next joint conjunction. This ends the episode.\n"
            "You have 10 actions total, ANSWER included. Choose your "
            "readings wisely."
        )
        return obs, {}

    def _corrective(self, message):
        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        obs = message
        if truncated:
            obs += " Step budget exhausted; episode truncated."
            self.done = True
        return obs, 0.0, False, truncated, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        text = (action or "").strip()
        read_match = re.fullmatch(
            r"(?i)READ\s+(-?\d+)", text
        )
        answer_match = re.fullmatch(
            r"(?i)ANSWER\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)", text
        )

        if read_match:
            day = int(read_match.group(1))
            if day < self.MIN_DAY or day > self.MAX_DAY:
                return self._corrective(
                    f"Invalid day {day}. Day must be between "
                    f"{self.MIN_DAY} and {self.MAX_DAY}."
                )
            self.steps += 1
            r_a = day % self.period_a
            r_b = day % self.period_b
            home_a = r_a == 0
            home_b = r_b == 0
            lines = [f"Day {day}:"]
            lines.append(
                f"  Aster is {r_a} day(s) into its current lap"
                f"{' -- AT HOME MARK' if home_a else ''}."
            )
            lines.append(
                f"  Borea is {r_b} day(s) into its current lap"
                f"{' -- AT HOME MARK' if home_b else ''}."
            )
            if home_a and home_b:
                lines.append(
                    "  *** Both planets are at their Home Mark together "
                    "on this day. ***"
                )
            truncated = self.steps >= self.MAX_STEPS
            obs = "\n".join(lines)
            if truncated:
                obs += "\nStep budget exhausted; episode truncated."
                self.done = True
            return obs, 0.0, False, truncated, {}

        if answer_match:
            self.steps += 1
            guess_a = int(answer_match.group(1))
            guess_b = int(answer_match.group(2))
            guess_day = int(answer_match.group(3))

            reward = 0.0
            correct_a = guess_a == self.period_a
            correct_b = guess_b == self.period_b
            correct_day = guess_day == self.conjunction_day
            if correct_a:
                reward += 0.3
            if correct_b:
                reward += 0.3
            if correct_day:
                reward += 0.4

            lines = ["ANSWER received."]
            lines.append(
                f"  Aster's period: {'correct' if correct_a else 'incorrect'}."
            )
            lines.append(
                f"  Borea's period: {'correct' if correct_b else 'incorrect'}."
            )
            lines.append(
                f"  Next conjunction day: "
                f"{'correct' if correct_day else 'incorrect'}."
            )
            lines.append(f"Episode reward: {reward:.2f}")
            self.done = True
            return "\n".join(lines), reward, True, False, {}

        return self._corrective(
            "Unrecognized action. Use 'READ <day>' or "
            "'ANSWER <period_a> <period_b> <day>'."
        )
