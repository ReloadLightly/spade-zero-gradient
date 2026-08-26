import random


class CellularAutomatonRuleEnv:
    """A hidden totalistic 1D cellular-automaton rule must be discovered via
    self-chosen probe rows, then used to extrapolate a fixed seed row forward."""

    MAX_STEPS = 10
    CELLS = 5

    def _valid_bits(self, s):
        return len(s) == self.CELLS and all(c in '01' for c in s)

    def _apply(self, state):
        L = self.CELLS
        out = []
        for i in range(L):
            left = state[i - 1]
            mid = state[i]
            right = state[(i + 1) % L]
            s = int(left) + int(mid) + int(right)
            out.append(str(self.table[s]))
        return ''.join(out)

    def _format_error(self):
        return (
            "Malformed action. Valid forms:\n"
            f"  QUERY <{self.CELLS}-bit string of 0/1>\n"
            f"  PREDICT <k> <{self.CELLS}-bit string of 0/1>   (k an integer from 1 to {self.target_k})"
        )

    def reset(self, seed=None):
        self.rng = random.Random(seed)

        while True:
            table = [self.rng.randint(0, 1) for _ in range(4)]
            if len(set(table)) > 1:
                break
        self.table = table

        self.target_k = 3

        while True:
            s0 = ''.join(self.rng.choice('01') for _ in range(self.CELLS))
            ones = s0.count('1')
            if 0 < ones < self.CELLS:
                break
        self.s0 = s0

        self.states = [s0]
        for _ in range(self.target_k):
            self.states.append(self._apply(self.states[-1]))

        self.steps = 0
        self.claimed = set()
        self.done = False

        obs = (
            "CELLULAR AUTOMATON RULE DISCOVERY.\n"
            f"There is a hidden row of {self.CELLS} cells (0/1), cyclic (the row wraps around). "
            "Each generation, every cell's new value depends ONLY on the total count of 1s among "
            "itself and its two cyclic neighbors (a number from 0 to 3) — this count-to-bit mapping "
            "is a fixed hidden rule table with exactly 4 unknown entries (for counts 0,1,2,3).\n"
            f"Fixed seed row (generation 0): {s0}\n"
            f"GOAL: predict the exact row at generation {self.target_k} (i.e. after applying the "
            "hidden rule 3 times, starting from the seed row above).\n"
            "ACTIONS (exactly one per turn):\n"
            f"  QUERY <row>       — apply the hidden rule once to a row of your choosing (any "
            f"{self.CELLS}-bit string), see the resulting next-generation row. Use this to work "
            "out the rule table.\n"
            f"  PREDICT <k> <row> — claim that the seed row's generation-k state equals <row>, for "
            f"any k from 1 to {self.target_k}. Predicting k < {self.target_k} is an optional "
            f"checkpoint (partial reward, doesn't end the game). Predicting k={self.target_k} ends "
            "the game immediately, right or wrong.\n"
            f"You have {self.MAX_STEPS} steps total (QUERY and PREDICT both count). Wrong PREDICTs "
            "tell you how many of the cells were correct, not which ones."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already over.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        terminated = False
        tokens = action.strip().split()

        if not tokens:
            obs = self._format_error()
        else:
            cmd = tokens[0].upper()
            if cmd == "QUERY" and len(tokens) == 2 and self._valid_bits(tokens[1]):
                nxt = self._apply(tokens[1])
                obs = (
                    f"QUERY {tokens[1]} -> {nxt}\n"
                    f"Steps used: {self.steps}/{self.MAX_STEPS}."
                )
            elif cmd == "PREDICT" and len(tokens) == 3:
                k_str, guess = tokens[1], tokens[2]
                if k_str.isdigit() and 1 <= int(k_str) <= self.target_k and self._valid_bits(guess):
                    k = int(k_str)
                    true_state = self.states[k]
                    matches = sum(1 for a, b in zip(guess, true_state) if a == b)
                    correct = (matches == self.CELLS)

                    if k == self.target_k:
                        frac = matches / self.CELLS
                        reward = round(0.7 * frac, 4)
                        terminated = True
                        if correct:
                            obs = (
                                f"PREDICT k={k}: MATCH! All {self.CELLS} cells correct. "
                                "Final horizon reached — episode complete."
                            )
                        else:
                            obs = (
                                f"PREDICT k={k}: {matches}/{self.CELLS} cells correct. "
                                f"That was the final horizon (k={self.target_k}) — episode over."
                            )
                    else:
                        checkpoint_value = 0.15
                        if correct and k not in self.claimed:
                            self.claimed.add(k)
                            reward = checkpoint_value
                            obs = (
                                f"PREDICT k={k}: MATCH! Checkpoint reward earned. "
                                f"{matches}/{self.CELLS} cells correct."
                            )
                        elif correct and k in self.claimed:
                            obs = (
                                f"PREDICT k={k}: correct again, but this checkpoint was already "
                                "claimed (no additional reward)."
                            )
                        else:
                            obs = (
                                f"PREDICT k={k}: {matches}/{self.CELLS} cells correct. "
                                "Not an exact match — keep probing."
                            )
                else:
                    obs = self._format_error()
            else:
                obs = self._format_error()

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            obs += f"\nStep limit ({self.MAX_STEPS}) reached — episode ends."

        if terminated or truncated:
            self.done = True

        return obs, reward, terminated, truncated, {}
