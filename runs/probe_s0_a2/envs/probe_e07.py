import random


class FestivalCycleEnv:
    """Solver infers a hidden repeating festival cycle from probes and predicts future festival days."""

    MIN_PERIOD = 3
    MAX_PERIOD = 7
    MAX_DAY = 100
    STEP_LIMIT = 10

    def __init__(self):
        self.rng = None
        self.period = None
        self.phase = None
        self.reference_day = None
        self.known_day = None
        self.target_days = None
        self.steps = 0
        self.done = False
        self.known_festivals = set()
        self.milestone_awarded = False

    def _is_festival(self, day):
        return (day - 1) % self.period == self.phase

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.period = self.rng.randint(self.MIN_PERIOD, self.MAX_PERIOD)
        self.phase = self.rng.randint(0, self.period - 1)
        self.reference_day = self.rng.randint(20, 40)
        self.known_day = self.phase + 1
        self.known_festivals = {self.known_day}
        self.milestone_awarded = False
        self.steps = 0
        self.done = False

        targets = []
        d = self.reference_day + 1
        while len(targets) < 3:
            if self._is_festival(d):
                targets.append(d)
            d += 1
        self.target_days = targets

        obs = (
            "FESTIVAL CYCLE PUZZLE\n"
            "A festival recurs on a hidden fixed cycle: every P days starting from some\n"
            "offset, a festival is held (P is an unknown integer between 3 and 7).\n"
            f"Recorded fact: Day {self.known_day} is a confirmed festival day.\n"
            f"GOAL: determine the cycle and report the next THREE festival days that\n"
            f"occur strictly after day {self.reference_day}, in increasing order.\n"
            "ACTIONS:\n"
            "  CHECK <day>            -- ask whether <day> (integer 1-100) is a festival day\n"
            "  ANSWER <d1> <d2> <d3>  -- submit the next three festival days after day "
            f"{self.reference_day} (ends the episode)\n"
            f"You have {self.STEP_LIMIT} total actions (CHECK + ANSWER combined)."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        parts = action.split()

        reward = 0.0
        terminated = False
        truncated = False

        if not parts:
            obs = "Empty action. Use 'CHECK <day>' or 'ANSWER <d1> <d2> <d3>'."
        elif parts[0].upper() == "CHECK":
            if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
                obs = "Malformed CHECK. Format: CHECK <day> (integer 1-100)."
            else:
                day = int(parts[1])
                if day < 1 or day > self.MAX_DAY:
                    obs = f"Day out of range. Use a day between 1 and {self.MAX_DAY}."
                else:
                    is_fest = self._is_festival(day)
                    if is_fest:
                        if day not in self.known_festivals:
                            self.known_festivals.add(day)
                            if not self.milestone_awarded and len(self.known_festivals) >= 2:
                                reward += 0.25
                                self.milestone_awarded = True
                        obs = f"Day {day}: FESTIVAL."
                    else:
                        obs = f"Day {day}: not a festival."
        elif parts[0].upper() == "ANSWER":
            if len(parts) != 4 or not all(p.lstrip("-").isdigit() for p in parts[1:]):
                obs = "Malformed ANSWER. Format: ANSWER <d1> <d2> <d3> (three integers)."
            else:
                guesses = [int(p) for p in parts[1:4]]
                correct = sum(1 for g, t in zip(guesses, self.target_days) if g == t)
                reward += 0.25 * correct
                terminated = True
                obs = (
                    f"ANSWER submitted: {guesses}. Correct matches (in order): {correct}/3. "
                    f"True next festival days after day {self.reference_day}: {self.target_days}."
                )
        else:
            obs = "Unknown action. Use 'CHECK <day>' or 'ANSWER <d1> <d2> <d3>'."

        if not terminated and self.steps >= self.STEP_LIMIT:
            truncated = True
            obs += f" Step limit ({self.STEP_LIMIT}) reached without a submitted ANSWER."

        if terminated or truncated:
            self.done = True

        return obs, reward, terminated, truncated, {}
