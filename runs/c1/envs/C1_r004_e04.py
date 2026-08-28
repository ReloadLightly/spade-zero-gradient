class NineCastersRelayEnv:
    def __init__(self):
        self.rng = None
        self.a = None
        self.b = None
        self.roots = []
        self.targets = []
        self.step_count = 0
        self.answered = False
        self.visible_len = 10
        self.total_len = 20

    @staticmethod
    def _digital_root(n):
        if n <= 0:
            return 0
        return 1 + (n - 1) % 9

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.a = self.rng.randint(2, 8)
        self.b = self.rng.randint(1, 8)
        x0 = self.rng.randint(100, 999)
        self.roots = [self._digital_root(x0)]
        for _ in range(1, self.total_len):
            nxt = self._digital_root(self.a * self.roots[-1] + self.b)
            self.roots.append(nxt)
        candidates = list(range(12, self.total_len))
        self.rng.shuffle(candidates)
        self.targets = sorted(candidates[:3])
        self.step_count = 0
        self.answered = False

        obs = (
            "NINE-CASTERS' RELAY. A hidden sequence x0, x1, x2, ... was built by a "
            "secret rule x_(i+1) = a*x_i + b, for fixed hidden integers a and b. "
            "Only the DIGITAL ROOT (the repeated digit-sum, reduced to a single "
            f"digit 1-9) of each of the first {self.visible_len} numbers "
            f"(indices 0-{self.visible_len - 1}) was ever recorded; later entries "
            f"were lost. Index 0's recorded root is {self.roots[0]}.\n"
            f"Your task: predict what the digital roots WOULD have been at the lost "
            f"indices {self.targets[0]}, {self.targets[1]}, {self.targets[2]} had the "
            "log continued under the same rule.\n"
            "Actions (one per step):\n"
            f"  QUERY <i>   - reveal the recorded root at index i (0-{self.visible_len - 1})\n"
            "  ANSWER <d0> <d1> <d2> - submit your three predicted roots (each 1-9), "
            "in order, for the target indices above; ends the episode.\n"
            "You have 10 steps total."
        )
        return obs, {}

    def step(self, action):
        self.step_count += 1
        text = (action or "").strip()
        tokens = text.split()

        if not tokens:
            return self._malformed("Empty action.")

        cmd = tokens[0].upper()

        if cmd == "QUERY":
            if len(tokens) != 2:
                return self._malformed("Use: QUERY <index>")
            try:
                idx = int(tokens[1])
            except ValueError:
                return self._malformed("Index must be an integer.")
            if idx < 0 or idx >= self.visible_len:
                obs = (
                    f"Index {idx} is outside the recorded log "
                    f"(only indices 0-{self.visible_len - 1} were recorded)."
                )
                return obs, 0.0, False, self._is_truncated(), {}
            obs = f"Recorded root at index {idx} is {self.roots[idx]}."
            return obs, 0.0, False, self._is_truncated(), {}

        if cmd == "ANSWER":
            if len(tokens) != 4:
                return self._malformed("Use: ANSWER <d0> <d1> <d2>")
            try:
                guesses = [int(t) for t in tokens[1:]]
            except ValueError:
                return self._malformed("Predicted roots must be integers 1-9.")
            if any(g < 1 or g > 9 for g in guesses):
                return self._malformed("Predicted roots must be integers 1-9.")

            self.answered = True
            correct = 0
            lines = []
            for tgt, guess in zip(self.targets, guesses):
                truth = self.roots[tgt]
                ok = guess == truth
                correct += int(ok)
                lines.append(
                    f"index {tgt}: guessed {guess}, actual {truth} -> "
                    f"{'correct' if ok else 'wrong'}"
                )
            reward = correct / 3.0
            obs = "Relay unsealed.\n" + "\n".join(lines) + f"\nScore: {correct}/3."
            return obs, reward, True, False, {}

        return self._malformed("Unknown command. Use QUERY or ANSWER.")

    def _malformed(self, msg):
        obs = msg + " No step consumed beyond this attempt."
        return obs, 0.0, False, self._is_truncated(), {}

    def _is_truncated(self):
        return (not self.answered) and self.step_count >= 10
