import random
import re


class TwoRitesAlmanacEnv:
    """Infer two hidden periodic festival calendars from probed days,
    then predict their occurrence at three held-out future dates."""

    CAL_LEN = 30
    MAX_STEPS = 10
    PERIOD_CHOICES = (3, 4, 5)

    def __init__(self):
        self.rng = None
        self.p1 = self.a1 = None
        self.p2 = self.a2 = None
        self.holdout = None
        self.step_count = 0
        self.terminated = False

    def _ember(self, day):
        return day % self.p1 == self.a1

    def _frost(self, day):
        return day % self.p2 == self.a2

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.p1 = self.rng.choice(self.PERIOD_CHOICES)
        self.a1 = self.rng.randrange(self.p1)
        self.p2 = self.rng.choice(self.PERIOD_CHOICES)
        self.a2 = self.rng.randrange(self.p2)
        self.holdout = sorted(self.rng.sample(range(31, 46), 3))
        self.step_count = 0
        self.terminated = False

        obs = (
            "ALMANAC OF TWO RITES\n"
            "Two festivals, Ember Rite and Frost Rite, each recur on a hidden "
            "fixed period with a hidden phase (each period is one of 3, 4, or "
            "5 days). A day may show either, both, or neither festival.\n\n"
            f"You may query any calendar day from 1 to {self.CAL_LEN} to see "
            "which festivals occur that day. You have "
            f"{self.MAX_STEPS} total actions (queries plus your final "
            "submission).\n\n"
            "ACTION FORMATS:\n"
            "  QUERY <day>            -- e.g. QUERY 7\n"
            "  SUBMIT d1=<code> d2=<code> d3=<code>\n"
            "    where <code> is one of E, F, EF, N (Ember only, Frost only, "
            "both, neither)\n\n"
            f"Your final SUBMIT must cover exactly these three future days, "
            f"in any order: {', '.join(str(d) for d in self.holdout)}.\n"
            "SUBMIT ends the episode; reward is granted per correctly "
            "predicted festival-bit across the three days."
        )
        return obs, {}

    def _corrective(self, msg):
        remaining = self.MAX_STEPS - self.step_count
        return f"{msg} ({remaining} actions remaining.)", 0.0, False, False, {}

    def step(self, action):
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()

        m = re.match(r"^QUERY\s+(\d+)$", text, re.IGNORECASE)
        if m:
            day = int(m.group(1))
            if not (1 <= day <= self.CAL_LEN):
                return self._checked_truncate(
                    f"Invalid day. Choose a day from 1 to {self.CAL_LEN}."
                )
            e = self._ember(day)
            f = self._frost(day)
            if e and f:
                label = "Ember YES, Frost YES"
            elif e:
                label = "Ember YES, Frost NO"
            elif f:
                label = "Ember NO, Frost YES"
            else:
                label = "Ember NO, Frost NO"
            remaining = self.MAX_STEPS - self.step_count
            obs = f"Day {day}: {label}. ({remaining} actions remaining.)"
            if remaining <= 0:
                return obs, 0.0, False, True, {}
            return obs, 0.0, False, False, {}

        m = re.match(r"^SUBMIT\s+(.+)$", text, re.IGNORECASE)
        if m:
            tokens = m.group(1).split()
            parsed = {}
            valid = True
            for tok in tokens:
                tm = re.match(r"^(\d+)=([A-Za-z]{1,2})$", tok)
                if not tm:
                    valid = False
                    break
                day = int(tm.group(1))
                code = tm.group(2).upper()
                if code in ("FE",):
                    code = "EF"
                if code not in ("E", "F", "EF", "N"):
                    valid = False
                    break
                parsed[day] = code
            if not valid or sorted(parsed.keys()) != self.holdout:
                return self._checked_truncate(
                    "Malformed SUBMIT. Provide exactly the three listed "
                    "days, each as day=CODE with CODE in {E, F, EF, N}."
                )

            correct_bits = 0
            for day in self.holdout:
                code = parsed[day]
                pred_e = "E" in code
                pred_f = "F" in code
                if pred_e == self._ember(day):
                    correct_bits += 1
                if pred_f == self._frost(day):
                    correct_bits += 1

            reward = correct_bits / (2 * len(self.holdout))
            self.terminated = True
            obs = (
                f"Submission graded: {correct_bits}/{2 * len(self.holdout)} "
                "festival-bits correct across the three days."
            )
            return obs, reward, True, False, {}

        return self._checked_truncate(
            "Unrecognized action. Use 'QUERY <day>' or "
            "'SUBMIT d1=<code> d2=<code> d3=<code>'."
        )

    def _checked_truncate(self, msg):
        obs, reward, terminated, truncated, info = self._corrective(msg)
        if self.step_count >= self.MAX_STEPS:
            return obs, reward, terminated, True, info
        return obs, reward, terminated, truncated, info
