import random


class RhythmProgressionEnv:
    N = 16
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.terminated = False
        self.count_rewarded = False
        self.submitted = False

        k1 = self.rng.choice([3, 4, 5])
        d = self.rng.choice([1, 2])
        self.k_seq = [k1 + i * d for i in range(5)]  # bars 1..5
        self.target_k = self.k_seq[4]
        self.target_pattern = self._pattern(self.target_k)

        bars_text = []
        for i in range(3):
            bars_text.append(
                f"Bar {i + 1} ({self.k_seq[i]} hits): {self._pattern(self.k_seq[i])}"
            )

        obs = (
            "GROOVE RECONSTRUCTION. Each bar has 16 sixteenth-note steps (positions 0-15), "
            "shown left to right as 'X' (hit) or '.' (rest). The hit COUNT rises by a fixed "
            "amount from bar to bar. Within any bar, the hits are always spread as evenly as "
            "possible across the 16 steps by one fixed, undisclosed rule that depends only on "
            "the hit count.\n\n"
            + "\n".join(bars_text)
            + "\n\nGoal: reconstruct Bar 5 exactly, as a 16-character string of X/. .\n\n"
            "Actions (send exactly one per turn):\n"
            "  PROBE <k> <pos>   - ask whether step <pos> (integer 0-15) is a hit in the "
            "16-step pattern for hit-count <k> (integer 1-16), under the same fixed rule.\n"
            "  COUNT <k>         - declare your answer for how many hits Bar 5 contains.\n"
            "  SUBMIT <pattern>  - submit your final 16-character X/. guess for Bar 5. This "
            "ends the episode.\n"
            f"You have {self.MAX_STEPS} steps total; PROBE, COUNT, and SUBMIT each use one "
            "step. SUBMIT may only be used once."
        )
        return obs, {}

    def _pattern(self, k):
        hits = set((i * self.N) // k for i in range(k))
        return "".join("X" if i in hits else "." for i in range(self.N))

    def step(self, action):
        if self.terminated:
            return "Episode already over.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        parts = text.split()
        reward = 0.0
        obs = ""

        if not parts:
            obs = "Empty action. Use PROBE <k> <pos>, COUNT <k>, or SUBMIT <pattern>."
        else:
            cmd = parts[0].upper()

            if cmd == "PROBE" and len(parts) == 3:
                try:
                    k = int(parts[1])
                    pos = int(parts[2])
                except ValueError:
                    k = pos = None
                if k is None or not (1 <= k <= 16) or not (0 <= pos <= 15):
                    obs = "Malformed PROBE. Use: PROBE <k 1-16> <pos 0-15>."
                else:
                    pat = self._pattern(k)
                    verdict = "HIT" if pat[pos] == "X" else "REST"
                    obs = f"PROBE k={k} pos={pos} -> {verdict}"

            elif cmd == "COUNT" and len(parts) == 2:
                try:
                    k = int(parts[1])
                except ValueError:
                    k = None
                if k is None or not (1 <= k <= 60):
                    obs = "Malformed COUNT. Use: COUNT <integer hit count>."
                elif k == self.target_k:
                    if not self.count_rewarded:
                        reward = 0.3
                        self.count_rewarded = True
                        obs = "COUNT correct. Bar 5's hit count is confirmed."
                    else:
                        obs = "COUNT correct (already rewarded once)."
                else:
                    obs = "COUNT incorrect for Bar 5's hit count."

            elif cmd == "SUBMIT" and len(parts) == 2:
                guess = parts[1].upper()
                if len(guess) != self.N or any(c not in "X." for c in guess):
                    obs = "Malformed SUBMIT. Send exactly 16 characters made of X and . only."
                else:
                    self.submitted = True
                    self.terminated = True
                    matches = sum(1 for a, b in zip(guess, self.target_pattern) if a == b)
                    if guess == self.target_pattern:
                        reward = 0.7
                        obs = (
                            f"SUBMIT exact match! Bar 5 was: {self.target_pattern}. "
                            "Episode complete."
                        )
                    else:
                        reward = 0.7 * (matches / self.N)
                        obs = (
                            f"SUBMIT incorrect. {matches}/16 positions matched. "
                            f"Bar 5 was: {self.target_pattern}. Episode complete."
                        )
            else:
                obs = (
                    "Unrecognized action. Use PROBE <k> <pos>, COUNT <k>, or "
                    "SUBMIT <16-char X/. pattern>."
                )

        truncated = False
        if not self.terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            obs += " Step limit reached without a submission."

        info = {
            "steps_remaining": max(0, self.MAX_STEPS - self.step_count),
        }
        return obs, reward, self.terminated, truncated, info
