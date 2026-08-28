import random


class MurderTimelineDeductionEnv:
    EVENT_POOL = [
        "the back door was found unlocked",
        "a scream was heard from the study",
        "the victim's car left the driveway",
        "the security light switched on in the garden",
        "a glass shattered in the kitchen",
        "the study lamp was seen flickering off",
        "a set of muddy footprints appeared on the porch",
    ]

    LABELS = ["A", "B", "C", "D", "E"]

    def __init__(self):
        self.rng = None
        self.true_order = []
        self.event_text = {}
        self.witnesses = []
        self.asked = set()
        self.step_count = 0
        self.max_steps = 10
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        chosen = self.rng.sample(self.EVENT_POOL, 5)
        shuffled_events = chosen[:]
        self.rng.shuffle(shuffled_events)
        self.event_text = {label: text for label, text in zip(self.LABELS, shuffled_events)}

        self.true_order = self.LABELS[:]
        self.rng.shuffle(self.true_order)

        pair_clues = [(self.true_order[i], self.true_order[i + 1]) for i in range(4)]
        bookend = (self.true_order[0], self.true_order[4])
        all_clues = pair_clues + [bookend]
        self.rng.shuffle(all_clues)

        self.witnesses = []
        for idx, (a, b) in enumerate(all_clues, start=1):
            if self.rng.random() < 0.5:
                text = (f"Witness {idx} recalls: \"{self.event_text[a]}\" "
                        f"happened before \"{self.event_text[b]}\".")
            else:
                text = (f"Witness {idx} recalls: \"{self.event_text[b]}\" "
                        f"happened after \"{self.event_text[a]}\".")
            self.witnesses.append(text)

        self.asked = set()
        self.step_count = 0
        self.done = False

        event_list = "; ".join(f"{label}: {self.event_text[label]}" for label in self.LABELS)
        obs = (
            "Reconstruct the true chronological order of five events on the night of the murder. "
            f"Events: {event_list}. There are 5 witnesses (numbered 1-5); each holds one clue "
            "about the relative order of two events, but does not say which position they occupy. "
            "Goal: output the full order from earliest to latest. "
            "Actions: 'ASK <n>' (n = 1-5) to hear a witness's clue, or "
            "'SUBMIT <order>' e.g. 'SUBMIT C,A,E,B,D' to give your final ordering — this ends "
            "the episode. You have 10 steps total."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()
        parts = action.split(None, 1)
        verb = parts[0].upper() if parts else ""
        reward = 0.0
        terminated = False

        if verb == "ASK" and len(parts) > 1:
            arg = parts[1].strip()
            if arg.isdigit() and 1 <= int(arg) <= len(self.witnesses):
                n = int(arg)
                if n in self.asked:
                    obs = (f"Witness {n} has nothing new — you already heard this clue: "
                           f"{self.witnesses[n - 1]}")
                else:
                    self.asked.add(n)
                    obs = self.witnesses[n - 1]
            else:
                obs = "Malformed ASK: use 'ASK <n>' where n is a witness number from 1 to 5."
        elif verb == "SUBMIT" and len(parts) > 1:
            guess = [x.strip().upper() for x in parts[1].split(",")]
            if len(guess) != 5 or sorted(guess) != sorted(self.LABELS):
                obs = "Malformed SUBMIT: give all five labels A,B,C,D,E exactly once, comma-separated."
            else:
                if guess[0] == self.true_order[0]:
                    reward += 0.2
                if guess[4] == self.true_order[4]:
                    reward += 0.2
                if guess[1:4] == self.true_order[1:4]:
                    reward += 0.3
                if guess == self.true_order:
                    reward += 0.3
                terminated = True
                self.done = True
                if guess == self.true_order:
                    obs = f"Correct! The true order was {','.join(self.true_order)}."
                else:
                    obs = f"Incorrect. The true order was {','.join(self.true_order)}."
        else:
            obs = "Malformed action. Use 'ASK <n>' or 'SUBMIT <a,b,c,d,e>'."

        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            obs += " Step limit reached — episode over."

        return obs, reward, terminated, truncated, {}
