import random


class TimelineWitnessEnv:
    EVENT_POOL = [
        ("W", "the forcing open of the study window"),
        ("L", "the arrival of the family lawyer"),
        ("S", "the unlocking of the wall safe"),
        ("P", "the victim's final phone call"),
        ("F", "the footsteps on the back stairs"),
    ]
    WITNESS_POOL = [
        "the butler",
        "the gardener",
        "the housekeeper",
        "the driver",
        "the next-door neighbor",
    ]

    def __init__(self):
        self.rng = None
        self.codes = [c for c, _ in self.EVENT_POOL]
        self.desc = dict(self.EVENT_POOL)
        self.true_order = []
        self.witness_names = []
        self.facts = {}
        self.credited = set()
        self.step_count = 0
        self.max_steps = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        order = self.codes[:]
        self.rng.shuffle(order)
        self.true_order = order

        chosen = self.rng.sample(self.WITNESS_POOL, 4)
        self.witness_names = chosen
        self.facts = {}
        for i in range(4):
            a, b = self.true_order[i], self.true_order[i + 1]
            if self.rng.random() < 0.5:
                text = (
                    f"{self.desc[a].capitalize()} happened immediately "
                    f"before {self.desc[b]}."
                )
            else:
                text = (
                    f"{self.desc[b].capitalize()} happened immediately "
                    f"after {self.desc[a]}."
                )
            self.facts[chosen[i]] = text

        self.credited = set()
        self.step_count = 0
        return self._opening_obs(), {}

    def _opening_obs(self):
        lines = [
            "Investigate the order of five events on the night of the murder.",
            "Events (code: description):",
        ]
        for c, d in self.EVENT_POOL:
            lines.append(f"  {c}: {d}")
        lines.append(
            "Four witnesses each know one fact linking two events as "
            "immediate neighbors in the true order. Interview them, then "
            "submit the full timeline."
        )
        lines.append("Witnesses available: " + ", ".join(self.witness_names))
        lines.append("Actions:")
        lines.append("  ASK <witness name>   e.g. ASK the butler")
        lines.append(
            "  ORDER <5 codes separated by spaces>   e.g. ORDER "
            + " ".join(self.codes)
        )
        lines.append(f"You have {self.max_steps} steps total.")
        return "\n".join(lines)

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        action = (action or "").strip()
        parts = action.split(None, 1)
        verb = parts[0].upper() if parts else ""

        if verb == "ASK" and len(parts) > 1:
            name = parts[1].strip().lower()
            match = None
            for w in self.witness_names:
                if w.lower() == name:
                    match = w
                    break
            if match is None:
                obs = (
                    "No such witness. Available witnesses: "
                    + ", ".join(self.witness_names)
                    + "."
                )
            else:
                obs = f"{match.capitalize()} says: {self.facts[match]}"

        elif verb == "ORDER" and len(parts) > 1:
            tokens = [t.upper() for t in parts[1].replace(",", " ").split()]
            if len(tokens) != 5 or set(tokens) != set(self.codes):
                obs = (
                    "Invalid order. Submit exactly the 5 codes "
                    f"{self.codes} each once, e.g. ORDER "
                    + " ".join(self.codes)
                    + "."
                )
            else:
                newly = 0
                for i in range(5):
                    if tokens[i] == self.true_order[i] and i not in self.credited:
                        self.credited.add(i)
                        newly += 1
                reward = 0.2 * newly
                correct_count = len(self.credited)
                if correct_count == 5:
                    terminated = True
                    obs = (
                        "Correct! The true timeline was "
                        + " ".join(self.true_order)
                        + ". Case closed."
                    )
                else:
                    obs = (
                        f"{correct_count} of 5 positions confirmed correct "
                        "so far (no detail on which). Keep investigating."
                    )
            info["correct_positions"] = len(self.credited)

        else:
            obs = (
                "Malformed action. Use 'ASK <witness name>' or "
                "'ORDER <5 codes separated by spaces>'."
            )

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            obs += (
                "\nStep limit reached. The true timeline was "
                + " ".join(self.true_order)
                + "."
            )

        return obs, reward, terminated, truncated, info
