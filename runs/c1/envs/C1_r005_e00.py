import random


class MissingWorkingEnv:
    STATIONS = ["Ashford", "Bramwell", "Corley", "Dunholt", "Elmswick"]
    CANDIDATES = ["Amber", "Birch", "Cedar", "Delta", "Ember"]
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.ghost = self.rng.choice(self.CANDIDATES)
        self.reals = [c for c in self.CANDIDATES if c != self.ghost]
        self.skip_of = {name: self.rng.choice(self.STATIONS) for name in self.reals}
        self.logs = {
            st: sorted(name for name in self.reals if self.skip_of[name] != st)
            for st in self.STATIONS
        }
        self.confirmed = set()
        self.steps = 0
        self.done = False
        obs = (
            "Five services are printed in the working timetable for this branch "
            f"line: {', '.join(self.CANDIDATES)}. Four are genuine trains that "
            "each call at four of the line's five stations (skipping exactly "
            "one station apiece). One name is a clerical ghost entry that "
            "never actually ran and appears at NO station's arrival log.\n"
            f"Stations, in order: {', '.join(self.STATIONS)}.\n"
            "Find the ghost entry.\n"
            "Actions (one per turn):\n"
            "  LOG <station>      - see every service recorded at that station\n"
            "  CONFIRM <service>  - declare a service genuine (verifiable, "
            "rewarded once per correct call)\n"
            "  ANSWER <service>   - final, irreversible: name the ghost entry\n"
            f"You have {self.MAX_STEPS} steps total; ANSWER ends the episode."
        )
        return obs, {}

    def _find(self, token, options):
        token = token.strip().lower()
        for opt in options:
            if opt.lower() == token:
                return opt
        return None

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        terminated = False

        parts = (action or "").strip().split(None, 1)
        verb = parts[0].upper() if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        if verb == "LOG" and arg:
            station = self._find(arg, self.STATIONS)
            if station is None:
                obs = (
                    f"No such station '{arg.strip()}'. Stations are: "
                    f"{', '.join(self.STATIONS)}."
                )
            else:
                seen = self.logs[station]
                obs = (
                    f"{station} arrival log records: {', '.join(seen)}."
                    if seen else f"{station} arrival log records: (none)."
                )
        elif verb == "CONFIRM" and arg:
            name = self._find(arg, self.CANDIDATES)
            if name is None:
                obs = (
                    f"'{arg.strip()}' is not a printed service name. Choices: "
                    f"{', '.join(self.CANDIDATES)}."
                )
            elif name in self.confirmed:
                obs = f"You already confirmed {name}."
            elif name == self.ghost:
                obs = (
                    f"No evidence confirms {name} as genuine yet — think again "
                    "before ruling it in."
                )
            else:
                self.confirmed.add(name)
                reward = 0.15
                obs = (
                    f"Confirmed: {name} is a genuine service. "
                    f"({len(self.confirmed)}/{len(self.reals)} genuine services confirmed.)"
                )
        elif verb == "ANSWER" and arg:
            name = self._find(arg, self.CANDIDATES)
            if name is None:
                obs = (
                    f"'{arg.strip()}' is not a printed service name. Choices: "
                    f"{', '.join(self.CANDIDATES)}."
                )
            else:
                terminated = True
                self.done = True
                if name == self.ghost:
                    reward = 0.4
                    obs = f"Correct — {name} is the ghost entry. Case closed."
                else:
                    obs = (
                        f"Incorrect. {name} was in fact a genuine service. "
                        "The working timetable's ghost entry goes unproven."
                    )
        else:
            obs = (
                "Unrecognized action. Use 'LOG <station>', "
                "'CONFIRM <service>', or 'ANSWER <service>'."
            )

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True

        return obs, reward, terminated, truncated, {}
