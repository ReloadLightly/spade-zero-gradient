import random
import math
import re
from functools import reduce


class PlanetaryConjunctionEnv:
    PLANETS = ["A", "B", "C"]
    MIN_PERIOD = 3
    MAX_PERIOD = 9
    MAX_DAY_QUERY = 200
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.periods = {}
        self.step_count = 0
        self.reported = set()
        self.done = False
        self.grand_conjunction = None

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        chosen = self.rng.sample(
            range(self.MIN_PERIOD, self.MAX_PERIOD + 1), len(self.PLANETS)
        )
        self.periods = dict(zip(self.PLANETS, chosen))
        self.grand_conjunction = reduce(
            lambda a, b: a * b // math.gcd(a, b), self.periods.values()
        )
        self.step_count = 0
        self.reported = set()
        self.done = False

        obs = (
            "PLANETARY CONJUNCTION TRACKER\n"
            "Three planets (A, B, C) orbit with hidden, distinct integer periods "
            f"between {self.MIN_PERIOD} and {self.MAX_PERIOD} days (inclusive). Each "
            "planet passes through its 'home marker' position exactly on every "
            "multiple of its own period, and all three started at their home "
            "marker together on day 0.\n"
            "GOAL: (1) determine each planet's period, and (2) determine the day "
            "of the next Grand Conjunction, the smallest day after 0 on which all "
            "three planets are simultaneously at their home marker.\n"
            "ACTIONS (send exactly one per turn):\n"
            f"  SCAN <day>            - probe a day (1-{self.MAX_DAY_QUERY}); learn "
            "which planets are at their home marker that day.\n"
            "  REPORT <planet> <n>   - claim planet <planet> (A/B/C) has period n; "
            "correct claims are rewarded once each.\n"
            "  ANSWER <day>          - submit your final Grand Conjunction day; "
            "ends the episode.\n"
            f"You have {self.MAX_STEPS} actions total; budget them across "
            "scanning, reporting, and your final answer."
        )
        info = {"step": self.step_count, "max_steps": self.MAX_STEPS}
        return obs, info

    def _corrective(self, message):
        self.step_count += 1
        truncated = self.step_count >= self.MAX_STEPS
        if truncated:
            self.done = True
            message += " Step limit reached; episode over."
        info = {"step": self.step_count, "max_steps": self.MAX_STEPS}
        return message, 0.0, False, truncated, info

    def step(self, action):
        if self.done:
            return (
                "Episode already finished.",
                0.0,
                True,
                False,
                {"step": self.step_count},
            )

        text = (action or "").strip()
        m_scan = re.fullmatch(r"(?i)SCAN\s+(-?\d+)", text)
        m_report = re.fullmatch(r"(?i)REPORT\s+([A-Ca-c])\s+(\d+)", text)
        m_answer = re.fullmatch(r"(?i)ANSWER\s+(\d+)", text)

        if m_scan:
            day = int(m_scan.group(1))
            if day < 1 or day > self.MAX_DAY_QUERY:
                return self._corrective(
                    f"Malformed SCAN: day must be between 1 and {self.MAX_DAY_QUERY}."
                )
            self.step_count += 1
            home = [p for p in self.PLANETS if day % self.periods[p] == 0]
            if home:
                obs = f"Day {day}: planet(s) {', '.join(home)} at home marker."
            else:
                obs = f"Day {day}: no planet at its home marker."
            truncated = self.step_count >= self.MAX_STEPS
            if truncated:
                self.done = True
                obs += " Step limit reached; episode over."
            info = {"step": self.step_count, "max_steps": self.MAX_STEPS}
            return obs, 0.0, False, truncated, info

        if m_report:
            planet = m_report.group(1).upper()
            guess = int(m_report.group(2))
            self.step_count += 1
            correct = self.periods[planet] == guess
            reward = 0.0
            if correct and planet not in self.reported:
                reward = 0.2
                self.reported.add(planet)
                obs = f"Confirmed: planet {planet}'s period is {guess} days. (+{reward:.1f})"
            elif correct:
                obs = f"Correct, but planet {planet} was already confirmed; no further reward."
            else:
                obs = f"Planet {planet}'s period is not {guess}. Keep probing."
            truncated = self.step_count >= self.MAX_STEPS
            if truncated:
                self.done = True
                obs += " Step limit reached; episode over."
            info = {"step": self.step_count, "max_steps": self.MAX_STEPS}
            return obs, reward, False, truncated, info

        if m_answer:
            guess = int(m_answer.group(1))
            self.step_count += 1
            self.done = True
            correct = guess == self.grand_conjunction
            reward = 0.4 if correct else 0.0
            if correct:
                obs = f"Correct! The Grand Conjunction falls on day {guess}. Episode complete."
            else:
                obs = f"Incorrect. Day {guess} is not the Grand Conjunction. Episode over."
            info = {
                "step": self.step_count,
                "max_steps": self.MAX_STEPS,
                "periods": dict(self.periods),
            }
            return obs, reward, True, False, info

        return self._corrective(
            "Malformed action. Use 'SCAN <day>', 'REPORT <planet> <n>', or 'ANSWER <day>'."
        )
