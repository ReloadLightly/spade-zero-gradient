import random
import string


class GemAuthenticationEnv:
    GEM_POOL = (
        "Ruby", "Sapphire", "Emerald", "Amethyst", "Topaz", "Opal",
        "Garnet", "Peridot", "Aquamarine", "Citrine", "Tourmaline",
        "Zircon", "Spinel", "Beryl", "Jade", "Onyx", "Moonstone",
        "Turquoise", "Lapis", "Pearl",
    )

    def __init__(self):
        self.n = 15
        self.max_steps = 10
        self.labels = list(string.ascii_uppercase[:self.n])
        self.rng = None
        self.gem_names = {}
        self.fake_index = None
        self.candidates = set()
        self.milestones_hit = set()
        self.milestone_thresholds = (5, 2, 1)
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        names = self.rng.sample(self.GEM_POOL, self.n)
        self.gem_names = dict(zip(self.labels, names))
        self.fake_index = self.rng.randrange(self.n)
        self.candidates = set(range(self.n))
        self.milestones_hit = set()
        self.steps = 0
        self.done = False

        roster = ", ".join(f"{lab}={self.gem_names[lab]}" for lab in self.labels)
        observation = (
            "You are appraising 15 gems, labeled A through O, laid out on a table:\n"
            f"{roster}\n"
            "Exactly one gem is a counterfeit and is HEAVIER than every genuine gem; "
            "all 14 genuine gems weigh exactly the same as each other.\n"
            "Find the counterfeit using a two-pan balance scale, in at most "
            f"{self.max_steps} actions total.\n"
            "Actions (send exactly one per turn):\n"
            "  WEIGH <labels> VS <labels>  -- e.g. 'WEIGH A,B,C VS D,E,F'. "
            "The two groups must be non-empty, the same size, and share no label. "
            "The scale reports LEFT_HEAVIER, RIGHT_HEAVIER, or BALANCED.\n"
            "  GUESS <label>  -- e.g. 'GUESS G'. Names your final answer for the "
            "counterfeit and ends the game (right or wrong).\n"
            "Track what each result rules in or out yourself -- the scale only "
            "ever reports which side (if either) tipped down."
        )
        return observation, {}

    def step(self, action):
        if self.done:
            return "The appraisal has already ended.", 0.0, True, False, {}

        self.steps += 1
        text = (action or "").strip()
        upper = text.upper()

        if upper.startswith("WEIGH"):
            reward, observation, terminated = self._handle_weigh(text)
        elif upper.startswith("GUESS"):
            reward, observation, terminated = self._handle_guess(text)
        else:
            reward, terminated = 0.0, False
            observation = (
                "Unrecognized action. Use 'WEIGH <labels> VS <labels>' or "
                "'GUESS <label>'."
            )

        truncated = False
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            observation += (
                f" Out of actions ({self.max_steps} used) -- the appraisal "
                "ends unresolved."
            )

        if terminated or truncated:
            self.done = True

        return observation, reward, terminated, truncated, {"step": self.steps}

    def _parse_group(self, chunk):
        parts = [p.strip().upper() for p in chunk.split(",") if p.strip()]
        if not parts:
            return None
        if any(p not in self.labels for p in parts):
            return None
        if len(parts) != len(set(parts)):
            return None
        return parts

    def _handle_weigh(self, text):
        body = text[len("WEIGH"):].strip()
        if " VS " not in body.upper():
            return 0.0, (
                "Malformed WEIGH action. Format: 'WEIGH A,B VS C,D' with two "
                "equal-size, non-overlapping groups of labels."
            ), False

        idx = body.upper().index(" VS ")
        left_raw, right_raw = body[:idx], body[idx + 4:]
        left = self._parse_group(left_raw)
        right = self._parse_group(right_raw)

        if left is None or right is None or len(left) != len(right):
            return 0.0, (
                "Invalid weighing: both sides must be non-empty, use known "
                "labels A-O, contain no repeats, and be the same size."
            ), False

        if set(left) & set(right):
            return 0.0, "Invalid weighing: a label cannot appear on both sides.", False

        left_idx = {ord(c) - ord('A') for c in left}
        right_idx = {ord(c) - ord('A') for c in right}

        if self.fake_index in left_idx:
            outcome = "LEFT_HEAVIER"
            self.candidates = self.candidates & left_idx
        elif self.fake_index in right_idx:
            outcome = "RIGHT_HEAVIER"
            self.candidates = self.candidates & right_idx
        else:
            outcome = "BALANCED"
            self.candidates = self.candidates - (left_idx | right_idx)

        reward = 0.0
        for threshold in self.milestone_thresholds:
            if threshold not in self.milestones_hit and len(self.candidates) <= threshold:
                self.milestones_hit.add(threshold)
                reward += 0.2

        observation = f"{outcome}. ({self.steps}/{self.max_steps} actions used.)"
        return reward, observation, False

    def _handle_guess(self, text):
        body = text[len("GUESS"):].strip().upper()
        if body not in self.labels:
            return 0.0, (
                "Invalid guess: name exactly one label A-O, e.g. 'GUESS G'."
            ), False

        guessed = ord(body) - ord('A')
        if guessed == self.fake_index:
            return 0.4, (
                f"Correct -- {self.gem_names[body]} ({body}) was the "
                "counterfeit. Appraisal complete."
            ), True

        correct_label = self.labels[self.fake_index]
        return 0.0, (
            f"Incorrect. {self.gem_names[body]} ({body}) was genuine; the "
            f"counterfeit was {self.gem_names[correct_label]} ({correct_label}). "
            "Appraisal complete."
        ), True
