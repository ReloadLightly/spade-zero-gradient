import random
import itertools


class FaceDownParityVaultEnv:
    """Identify two secretly flipped face-down cards using subset-parity queries."""

    N = 8
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.flip_a, self.flip_b = sorted(self.rng.sample(range(self.N), 2))
        self.step_count = 0
        self.history = []
        self.milestone4_awarded = False
        self.milestone1_awarded = False
        self.done = False
        obs = (
            "VAULT: 8 face-down cards lie at positions 0-7. Exactly two of them "
            "are secretly FLIPPED; the rest are not. Your goal: identify both "
            "flipped positions.\n"
            "You have 10 total actions (queries + final guess count together).\n"
            "Actions:\n"
            "  PARITY <positions...>  - e.g. 'PARITY 0 2 5' - ask whether the "
            "count of flipped cards among the listed positions (space-separated "
            "distinct integers, 0-7) is ODD or EVEN.\n"
            "  GUESS <pos1> <pos2>    - e.g. 'GUESS 3 6' - submit your final "
            "answer (two distinct integers, 0-7). This ends the episode.\n"
            "28 pairs are possible at the start."
        )
        return obs, {}

    def _consistent_pairs(self):
        result = []
        for a, b in itertools.combinations(range(self.N), 2):
            ok = True
            for subset, is_odd in self.history:
                cnt = (a in subset) + (b in subset)
                if (cnt % 2 == 1) != is_odd:
                    ok = False
                    break
            if ok:
                result.append((a, b))
        return result

    def _malformed(self, message):
        truncated = self.step_count >= self.MAX_STEPS
        if truncated:
            self.done = True
        return message, 0.0, False, truncated, {}

    def step(self, action):
        if self.done:
            return "The vault is already closed.", 0.0, True, False, {}

        self.step_count += 1
        tokens = action.strip().split()

        if not tokens:
            return self._malformed(
                "Unrecognized action. Use 'PARITY <positions...>' or "
                "'GUESS <pos1> <pos2>'."
            )

        cmd = tokens[0].upper()
        args = tokens[1:]

        if cmd == "PARITY":
            if not args:
                return self._malformed(
                    "PARITY needs at least one position, e.g. 'PARITY 0 2 5'."
                )
            try:
                ints = [int(t) for t in args]
            except ValueError:
                return self._malformed(
                    "Positions must be integers 0-7, e.g. 'PARITY 0 2 5'."
                )
            if len(set(ints)) != len(ints) or any(i < 0 or i >= self.N for i in ints):
                return self._malformed(
                    "Positions must be distinct integers in range 0-7."
                )

            subset = frozenset(ints)
            cnt = (self.flip_a in subset) + (self.flip_b in subset)
            is_odd = cnt % 2 == 1
            self.history.append((subset, is_odd))
            response = "ODD" if is_odd else "EVEN"

            candidates = self._consistent_pairs()
            reward = 0.0
            notes = []
            if len(candidates) <= 4 and not self.milestone4_awarded:
                self.milestone4_awarded = True
                reward += 0.3
                notes.append("Milestone: candidate pairs narrowed to <=4! (+0.3)")
            if len(candidates) == 1 and not self.milestone1_awarded:
                self.milestone1_awarded = True
                reward += 0.3
                notes.append("Milestone: the flipped pair is now uniquely determined! (+0.3)")

            truncated = self.step_count >= self.MAX_STEPS
            if truncated:
                self.done = True

            steps_left = self.MAX_STEPS - self.step_count
            obs_lines = [
                f"Parity of {{{', '.join(str(i) for i in sorted(subset))}}}: {response}.",
                f"Consistent candidate pairs remaining: {len(candidates)}.",
                f"Steps remaining: {steps_left}.",
            ]
            obs_lines.extend(notes)
            if truncated:
                obs_lines.append("Step budget exhausted without a final GUESS.")
            return "\n".join(obs_lines), reward, False, truncated, {
                "candidates_remaining": len(candidates)
            }

        elif cmd == "GUESS":
            if len(args) != 2:
                return self._malformed(
                    "GUESS needs exactly two positions, e.g. 'GUESS 3 6'."
                )
            try:
                ints = [int(t) for t in args]
            except ValueError:
                return self._malformed(
                    "GUESS positions must be integers, e.g. 'GUESS 3 6'."
                )
            if len(set(ints)) != 2 or any(i < 0 or i >= self.N for i in ints):
                return self._malformed(
                    "GUESS needs two distinct integers in range 0-7."
                )

            guess = sorted(ints)
            self.done = True
            correct = guess[0] == self.flip_a and guess[1] == self.flip_b
            if correct:
                obs = f"Correct! Positions {guess[0]} and {guess[1]} were the flipped cards."
                reward = 0.4
            else:
                obs = (
                    f"Incorrect guess {tuple(guess)}. The flipped cards were "
                    f"{self.flip_a} and {self.flip_b}."
                )
                reward = 0.0
            return obs, reward, True, False, {}

        else:
            return self._malformed(
                f"Unknown command '{tokens[0]}'. Use 'PARITY <positions...>' or "
                "'GUESS <pos1> <pos2>'."
            )
