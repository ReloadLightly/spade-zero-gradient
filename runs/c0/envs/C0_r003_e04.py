import random


class SymmetryDetectiveEnv:
    ALPHABET = ['A', 'B', 'C', 'D']
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.map = {}
        self.targets = []
        self.target_labels = []
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.steps = 0
        self.done = False

        unassigned = list(self.ALPHABET)
        self.map = {}
        while unassigned:
            x = unassigned.pop(0)
            if unassigned and self.rng.random() < 0.6:
                y = self.rng.choice(unassigned)
                unassigned.remove(y)
                self.map[x] = y
                self.map[y] = x
            else:
                self.map[x] = x

        labels = [True, True, False, False]
        self.rng.shuffle(labels)
        self.targets = [self._make_string(lbl) for lbl in labels]
        self.target_labels = labels

        obs = self._format_intro()
        return obs, {}

    def _make_string(self, symmetric):
        s = [None] * 6
        for i in range(3):
            letter = self.rng.choice(self.ALPHABET)
            s[i] = letter
            s[5 - i] = self.map[letter]
        if not symmetric:
            idx = self.rng.choice([0, 1, 2])
            correct = self.map[s[idx]]
            choices = [c for c in self.ALPHABET if c != correct]
            s[5 - idx] = self.rng.choice(choices)
        return ''.join(s)

    def _format_intro(self):
        lines = []
        lines.append("SYMMETRY DETECTIVE")
        lines.append(
            "A secret 180-degree rotation rule maps each letter in {A,B,C,D} to "
            "another letter (or itself); it is an involution (apply it twice, "
            "get the original letter back)."
        )
        lines.append(
            "A length-6 string is ROTATION-SYMMETRIC if, for each mirrored "
            "position pair (0,5), (1,4), (2,3), the rule applied to the left "
            "letter equals the right letter."
        )
        lines.append("Classify these 4 target strings as symmetric (T) or not (F):")
        for i, t in enumerate(self.targets, 1):
            lines.append(f"  {i}: {t}")
        lines.append(f"You have {self.MAX_STEPS} steps total. Actions:")
        lines.append(
            "  EXAMPLE <string>  - a length-6 string of letters A-D; returns "
            "MATCH/MISMATCH for each of the 3 mirrored position-pairs."
        )
        lines.append(
            "  CLASSIFY <T/F> <T/F> <T/F> <T/F>  - final answer for targets "
            "1-4 in order. Ends the episode; scores 0.25 per correct answer."
        )
        lines.append("Use EXAMPLE to gather evidence before you CLASSIFY.")
        return "\n".join(lines)

    def step(self, action):
        if self.done:
            return ("Episode already finished.", 0.0, True, False, {})

        self.steps += 1
        tokens = str(action).strip().split()
        reward = 0.0
        terminated = False
        obs = ""

        if not tokens:
            obs = "Empty action. Use EXAMPLE <string> or CLASSIFY <T/F> <T/F> <T/F> <T/F>."
        else:
            cmd = tokens[0].upper()

            if cmd == "EXAMPLE":
                if len(tokens) != 2 or len(tokens[1]) != 6 or \
                        not all(c.upper() in self.ALPHABET for c in tokens[1]):
                    obs = (
                        "Malformed EXAMPLE. Provide exactly one string of length 6 "
                        "using only letters A, B, C, D. Example: EXAMPLE AABBCD"
                    )
                else:
                    s = tokens[1].upper()
                    lines = [f"Feedback for EXAMPLE {s}:"]
                    for i in range(3):
                        a, b = s[i], s[5 - i]
                        match = self.map[a] == b
                        lines.append(
                            f"  pair({i},{5 - i}) [{a},{b}]: "
                            f"{'MATCH' if match else 'MISMATCH'}"
                        )
                    obs = "\n".join(lines)

            elif cmd == "CLASSIFY":
                valid = len(tokens) == 5 and all(
                    t.upper() in ("T", "F") for t in tokens[1:]
                )
                if not valid:
                    obs = (
                        "Malformed CLASSIFY. Provide exactly 4 answers, each T or F, "
                        "e.g.: CLASSIFY T F T F"
                    )
                else:
                    answers = [t.upper() == "T" for t in tokens[1:5]]
                    correct = sum(
                        1 for a, b in zip(answers, self.target_labels) if a == b
                    )
                    reward = 0.25 * correct
                    terminated = True
                    self.done = True
                    lines = [
                        f"CLASSIFY submitted: {correct}/4 correct.",
                        "True labels: " + ", ".join(
                            "T" if lbl else "F" for lbl in self.target_labels
                        ),
                        "Secret rule: " + ", ".join(
                            f"{k}->{v}" for k, v in sorted(self.map.items())
                        ),
                    ]
                    obs = "\n".join(lines)

            else:
                obs = (
                    f"Unknown command '{tokens[0]}'. Use EXAMPLE or CLASSIFY."
                )

        truncated = (not terminated) and (self.steps >= self.MAX_STEPS)
        if truncated:
            self.done = True
            obs += "\nStep limit reached. Episode truncated."

        return obs, reward, terminated, truncated, {}
