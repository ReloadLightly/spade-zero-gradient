import random


class VillageCouncilEnv:
    NAME_POOL = ["Ardo", "Bela", "Coro", "Dima", "Elin", "Faro", "Gwen", "Hale"]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.n = 4
        self.names = self.rng.sample(self.NAME_POOL, self.n)
        self.lookup = {name.lower(): i for i, name in enumerate(self.names)}
        self.k = self.rng.choice([1, 2, 3])

        idxs = list(range(self.n))
        self.rng.shuffle(idxs)
        knight_set = set(idxs[: self.k])
        self.is_knight = [i in knight_set for i in range(self.n)]

        self.answered = [False] * self.n
        self.step_count = 0
        self.max_steps = 8
        self.done = False

        roster = ", ".join(self.names)
        obs = (
            "You are investigating a village where every inhabitant is either "
            "a knight (always tells the truth) or a knave (always lies).\n"
            f"Villagers: {roster}.\n"
            f"The council records confirm exactly {self.k} of these {self.n} "
            "villagers are knights.\n\n"
            "Goal: determine every villager's true type.\n"
            "Actions (exactly one per turn):\n"
            "  ASK <name1> <name2>  -- ask name1 whether name2 is a knight; "
            "you hear name1's claim, which may be true or false.\n"
            "  ANSWER <name> KNIGHT  or  ANSWER <name> KNAVE  -- lock in your "
            "verdict for one villager. Each villager can be answered once.\n"
            f"You have {self.max_steps} actions total. Episode ends when every "
            "villager has been answered or the action limit is reached."
        )
        return obs, {"knights_total": self.k}

    def step(self, action):
        self.step_count += 1
        tokens = (action or "").strip().split()

        reward = 0.0
        terminated = False
        truncated = False
        info = {"steps_used": self.step_count}

        if len(tokens) == 3 and tokens[0].upper() == "ASK":
            obs = self._handle_ask(tokens[1], tokens[2])
        elif len(tokens) == 3 and tokens[0].upper() == "ANSWER":
            obs, reward = self._handle_answer(tokens[1], tokens[2].upper())
        else:
            obs = (
                "Malformed action. Use 'ASK <name1> <name2>' or "
                "'ANSWER <name> KNIGHT'/'ANSWER <name> KNAVE'."
            )

        if all(self.answered):
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        info["villagers_answered"] = sum(self.answered)
        return obs, reward, terminated, truncated, info

    def _handle_ask(self, raw1, raw2):
        i1 = self.lookup.get(raw1.lower())
        i2 = self.lookup.get(raw2.lower())
        if i1 is None or i2 is None:
            return f"Unknown villager name in ASK. Choices are: {', '.join(self.names)}."
        if i1 == i2:
            return "A villager cannot be asked about themselves. Pick two different names."

        same_type = self.is_knight[i1] == self.is_knight[i2]
        name1, name2 = self.names[i1], self.names[i2]
        if same_type:
            return f'{name1} says: "Yes, {name2} is a knight."'
        return f'{name1} says: "No, {name2} is a knave."'

    def _handle_answer(self, raw_name, verdict):
        idx = self.lookup.get(raw_name.lower())
        if idx is None:
            return f"Unknown villager name in ANSWER. Choices are: {', '.join(self.names)}.", 0.0
        if verdict not in ("KNIGHT", "KNAVE"):
            return "Verdict must be KNIGHT or KNAVE.", 0.0
        if self.answered[idx]:
            return f"{self.names[idx]} has already been given a final verdict.", 0.0

        self.answered[idx] = True
        claimed_knight = verdict == "KNIGHT"
        correct = claimed_knight == self.is_knight[idx]
        reward = (1.0 / self.n) if correct else 0.0
        result = "correct" if correct else "recorded"
        return f"Verdict for {self.names[idx]} {result}.", reward
