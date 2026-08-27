import random


class TreasureMapDeductionEnv:
    REGIONS = ["North", "South"]
    TERRAINS = ["Forest", "Cave", "Shore"]
    ELEVATIONS = ["Low", "High"]
    DISTANCES = ["Near", "Mid", "Far"]
    CATEGORIES = ["REGION", "TERRAIN", "ELEVATION", "DISTANCE"]
    NUM_SITES = 8

    def __init__(self):
        self.rng = None
        self.sites = []
        self.attrs = {}
        self.secret = None
        self.asked = set()
        self.revealed = {}
        self.candidates = set()
        self.milestones_hit = set()
        self.steps = 0
        self.max_steps = 10
        self.done = False

    def _random_sig(self):
        return {
            "REGION": self.rng.choice(self.REGIONS),
            "TERRAIN": self.rng.choice(self.TERRAINS),
            "ELEVATION": self.rng.choice(self.ELEVATIONS),
            "DISTANCE": self.rng.choice(self.DISTANCES),
        }

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.sites = [f"S{i + 1}" for i in range(self.NUM_SITES)]
        secret_idx = self.rng.randrange(self.NUM_SITES)
        self.secret = self.sites[secret_idx]

        twin_size = self.rng.randint(2, 4)
        other_indices = [i for i in range(self.NUM_SITES) if i != secret_idx]
        self.rng.shuffle(other_indices)
        twin_indices = set([secret_idx] + other_indices[: twin_size - 1])

        secret_sig = self._random_sig()

        self.attrs = {}
        for i, sid in enumerate(self.sites):
            if i in twin_indices:
                self.attrs[sid] = dict(secret_sig)
            else:
                sig = self._random_sig()
                tries = 0
                while sig == secret_sig and tries < 50:
                    sig = self._random_sig()
                    tries += 1
                self.attrs[sid] = sig

        self.asked = set()
        self.revealed = {}
        self.candidates = set(self.sites)
        self.milestones_hit = set()
        self.steps = 0
        self.done = False

        table_lines = []
        for sid in self.sites:
            a = self.attrs[sid]
            table_lines.append(
                f"  {sid}: region={a['REGION']}, terrain={a['TERRAIN']}, "
                f"elevation={a['ELEVATION']}, distance={a['DISTANCE']}"
            )
        table = "\n".join(table_lines)

        obs = (
            "TREASURE MAP DEDUCTION\n"
            "A treasure is buried at exactly one of these 8 sites. You know every site's "
            "attributes below, but not which site holds the treasure.\n"
            f"{table}\n\n"
            "Goal: name the treasure's site.\n"
            "Actions (one per turn):\n"
            "  ASK <CATEGORY>  -- CATEGORY is one of REGION, TERRAIN, ELEVATION, DISTANCE. "
            "Reveals the treasure site's value for that category.\n"
            "  GUESS <SITE>    -- SITE is one of S1..S8. Ends the episode.\n"
            f"You have at most {self.max_steps} actions total. Malformed actions are "
            "corrected and do not consume a step."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        parts = (action or "").strip().split()
        reward = 0.0

        if len(parts) == 2 and parts[0].upper() == "ASK" and parts[1].upper() in self.CATEGORIES:
            cat = parts[1].upper()
            self.steps += 1
            already_asked = cat in self.asked
            self.asked.add(cat)
            value = self.attrs[self.secret][cat]
            self.revealed[cat] = value
            self.candidates = {sid for sid in self.candidates if self.attrs[sid][cat] == value}

            if not already_asked:
                if "first_ask" not in self.milestones_hit:
                    self.milestones_hit.add("first_ask")
                    reward += 0.1
                if len(self.candidates) <= 5 and "half" not in self.milestones_hit:
                    self.milestones_hit.add("half")
                    reward += 0.1
                if len(self.candidates) <= 4 and "shortlist" not in self.milestones_hit:
                    self.milestones_hit.add("shortlist")
                    reward += 0.1
                if len(self.asked) == 4 and "complete" not in self.milestones_hit:
                    self.milestones_hit.add("complete")
                    reward += 0.1

            obs = (
                f"{cat} of the treasure site: {value}. "
                f"Sites still consistent with everything revealed so far: {len(self.candidates)}."
            )
            truncated = self.steps >= self.max_steps
            return obs, reward, False, truncated, {}

        if len(parts) == 2 and parts[0].upper() == "GUESS" and parts[1].upper() in self.sites:
            self.steps += 1
            guess = parts[1].upper()
            self.done = True
            if guess == self.secret:
                reward = 0.6
                result = "Correct!"
            elif len(self.asked) > 0 and guess in self.candidates:
                reward = 0.3
                result = "Incorrect, but consistent with every clue you gathered."
            else:
                reward = 0.0
                result = "Incorrect."
            obs = f"You guessed {guess}. The treasure was at {self.secret}. {result}"
            return obs, reward, True, False, {}

        return (
            "Malformed action. Use 'ASK <CATEGORY>' (REGION/TERRAIN/ELEVATION/DISTANCE) "
            "or 'GUESS <SITE>' (S1..S8).",
            0.0,
            False,
            False,
            {},
        )
