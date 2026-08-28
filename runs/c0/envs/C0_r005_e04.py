import random


class WeekdayLeapCipherEnv:
    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
    QUERY_MIN, QUERY_MAX = 0, 15
    STEP_LIMIT = 10

    def __init__(self):
        self.rng = None
        self.w0 = None
        self.m = None
        self.r = None
        self.target_year = None
        self.steps = 0
        self.rule_used = False
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.w0 = self.rng.randrange(7)
        self.m = self.rng.choice([3, 4, 5, 6])
        self.r = self.rng.randrange(self.m)
        self.target_year = self.rng.randint(20, 35)
        self.steps = 0
        self.rule_used = False
        self.done = False

        obs = (
            "WEEKDAY CIPHER. A calendar starts at year 0. Every year is either "
            "'common' (shifts the Jan-1 weekday by 1 day) or 'leap' (shifts it "
            "by 2 days), following a hidden rule of the form "
            "'year MOD m == r' for some unknown m in {3,4,5,6} and r in [0,m).\n"
            f"You may query Jan 1's weekday for any year in [{self.QUERY_MIN}, "
            f"{self.QUERY_MAX}] with: QUERY <year>\n"
            "Once, you may submit your inferred rule with: RULE <m> <r> "
            "(scored 0.3 if both match exactly; usable only once, so gather "
            "evidence first).\n"
            f"Finally, predict the Jan-1 weekday of year {self.target_year} "
            "(outside the queryable range) with: PREDICT <WeekdayName> "
            "(scored 0.7 if correct). PREDICT ends the episode.\n"
            f"You have {self.STEP_LIMIT} total actions across all commands."
        )
        return obs, {"target_year": self.target_year, "step_limit": self.STEP_LIMIT}

    def _is_leap(self, year):
        return year % self.m == self.r

    def _weekday_index(self, year):
        idx = self.w0
        for k in range(0, year):
            idx = (idx + (2 if self._is_leap(k) else 1)) % 7
        return idx

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        tokens = text.split()
        reward = 0.0
        terminated = False

        if not tokens:
            obs = "Empty action. Use QUERY <year>, RULE <m> <r>, or PREDICT <WeekdayName>."
        else:
            cmd = tokens[0].upper()

            if cmd == "QUERY" and len(tokens) == 2:
                try:
                    year = int(tokens[1])
                except ValueError:
                    year = None
                if year is None or not (self.QUERY_MIN <= year <= self.QUERY_MAX):
                    obs = f"Invalid year. Query an integer in [{self.QUERY_MIN}, {self.QUERY_MAX}]."
                else:
                    wd = self.WEEKDAYS[self._weekday_index(year)]
                    obs = f"Year {year}: Jan 1 is a {wd}."

            elif cmd == "RULE" and len(tokens) == 3:
                if self.rule_used:
                    obs = "Rule already submitted once; that guess is locked in. Continue with QUERY or submit PREDICT."
                else:
                    self.rule_used = True
                    try:
                        m_guess, r_guess = int(tokens[1]), int(tokens[2])
                        correct = (m_guess == self.m and r_guess == self.r)
                    except ValueError:
                        correct = False
                    if correct:
                        reward = 0.3
                        obs = "Rule guess correct."
                    else:
                        obs = "Rule guess incorrect."

            elif cmd == "PREDICT" and len(tokens) == 2:
                guess = tokens[1].strip().capitalize()
                actual = self.WEEKDAYS[self._weekday_index(self.target_year)]
                terminated = True
                self.done = True
                if guess == actual:
                    reward = 0.7
                    obs = f"Correct! Year {self.target_year}'s Jan 1 is {actual}. Episode complete."
                else:
                    obs = f"Incorrect. Episode complete."

            else:
                obs = ("Unrecognized action. Use exactly: QUERY <year>, "
                       "RULE <m> <r>, or PREDICT <WeekdayName>.")

        truncated = False
        if not terminated and self.steps >= self.STEP_LIMIT:
            truncated = True
            self.done = True

        return obs, reward, terminated, truncated, {"steps": self.steps}
