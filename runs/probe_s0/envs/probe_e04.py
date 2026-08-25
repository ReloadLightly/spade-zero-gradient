import re
import math
import random


class ConjunctionCipherEnv:
    """
    Two hidden planets A and B start aligned (in conjunction) on day 0.
    Each has a hidden integer orbital period (in days). The solver may
    query the orbital phase of either planet on any day, expressed as a
    reduced fraction of its period, then must identify both periods and
    compute the day of a specified future conjunction.
    """

    MAX_STEPS = 10

    _PARAM_CHOICES = [
        (2, 2), (2, 3), (2, 4),
        (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3), (4, 4),
    ]

    def __init__(self):
        self._rng = None
        self._period = {}
        self._n_target = None
        self._step_count = 0
        self._done = False
        self._claimed_period = {"A": False, "B": False}
        self._guessed_correct = {"A": False, "B": False}

    def reset(self, seed=None):
        self._rng = random.Random(seed)
        d, k = self._rng.choice(self._PARAM_CHOICES)
        period_a = k * d
        period_b = (k + 1) * d
        self._period = {"A": period_a, "B": period_b}
        self._n_target = self._rng.randint(2, 4)
        self._step_count = 0
        self._done = False
        self._claimed_period = {"A": False, "B": False}
        self._guessed_correct = {"A": False, "B": False}

        obs = (
            "Two planets, A and B, orbit with unknown integer periods (in whole "
            "days) and were exactly aligned (conjunction #0) on day 0.\n"
            "Actions (one per line, exactly this form):\n"
            "  PHASE A <day>        -- reveals planet A's orbital phase on <day> "
            "as a reduced fraction remainder/period\n"
            "  PHASE B <day>        -- same for planet B\n"
            "  GUESS_PERIOD A <n>   -- claim planet A's period is <n> (integer "
            "days); correct claims pay out once\n"
            "  GUESS_PERIOD B <n>   -- claim planet B's period is <n>\n"
            "  ANSWER <day>         -- final answer: the day of conjunction #"
            f"{self._n_target} (counting day 0 as conjunction #0); ends the episode\n"
            f"You have {self.MAX_STEPS} actions total. <day> and <n> are "
            "non-negative integers."
        )
        return obs, {}

    def step(self, action):
        if self._done:
            return "Episode already finished.", 0.0, True, False, {}

        self._step_count += 1
        text = (action or "").strip()

        m = re.match(r"^PHASE\s+([ABab])\s+(\d+)$", text)
        if m:
            planet = m.group(1).upper()
            day = int(m.group(2))
            period = self._period[planet]
            remainder = day % period
            if remainder == 0:
                frac = "0/1"
            else:
                g = math.gcd(remainder, period)
                frac = f"{remainder // g}/{period // g}"
            obs = f"Planet {planet} phase on day {day}: {frac}"
            return obs, 0.0, False, self._maybe_truncate(), {}

        m = re.match(r"^GUESS_PERIOD\s+([ABab])\s+(\d+)$", text)
        if m:
            planet = m.group(1).upper()
            guess = int(m.group(2))
            actual = self._period[planet]
            if guess == actual:
                if self._claimed_period[planet]:
                    obs = f"Correct, but planet {planet}'s period was already confirmed."
                    reward = 0.0
                else:
                    self._claimed_period[planet] = True
                    self._guessed_correct[planet] = True
                    obs = f"Confirmed: planet {planet}'s period is {actual} days."
                    reward = 0.25
            else:
                obs = f"Incorrect: planet {planet}'s period is not {guess}."
                reward = 0.0
            return obs, reward, False, self._maybe_truncate(), {}

        m = re.match(r"^ANSWER\s+(\d+)$", text)
        if m:
            answer = int(m.group(1))
            lcm = self._period["A"] * self._period["B"] // math.gcd(
                self._period["A"], self._period["B"]
            )
            target = self._n_target * lcm
            self._done = True
            if answer == target:
                obs = f"Correct! Conjunction #{self._n_target} falls on day {target}."
                reward = 0.5
            else:
                obs = f"Incorrect. That is not the day of conjunction #{self._n_target}."
                reward = 0.0
            return obs, reward, True, False, {}

        obs = (
            "Malformed action. Use one of: PHASE A <day>, PHASE B <day>, "
            "GUESS_PERIOD A <n>, GUESS_PERIOD B <n>, ANSWER <day>."
        )
        return obs, 0.0, False, self._maybe_truncate(), {}

    def _maybe_truncate(self):
        if self._step_count >= self.MAX_STEPS:
            self._done = True
            return True
        return False
