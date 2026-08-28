import random


class EchoLedgerEnv:
    """
    A scribe repeatedly transforms a hidden starting number by adding its own
    digit sum. The solver never sees the numbers, only the 'digit echo'
    (digital root, 1-9) at indices it chooses to inspect (0-4 only). It must
    then predict the digit echoes at three far indices it cannot inspect.
    """

    def __init__(self):
        self.rng = None
        self.roots = []
        self.step_count = 0
        self.max_steps = 10
        self.max_revealable_index = 4
        self.target_indices = (6, 8, 10)
        self.revealed = set()
        self.done = False

    def _digital_root(self, n):
        if n <= 0:
            return 0
        return 9 if n % 9 == 0 else n % 9

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False
        self.revealed = set()

        root0 = self.rng.randint(1, 8)
        self.roots = [root0]
        for _ in range(max(self.target_indices)):
            prev = self.roots[-1]
            self.roots.append(self._digital_root(2 * prev))

        obs = (
            "You are auditing the Echo Ledger, a scroll of entries at indices 0 through "
            f"{max(self.target_indices)}. Each entry was produced by a scribe who repeatedly "
            "transformed a hidden starting number by adding its own digit sum, over and over, "
            "one transform per index step. You never see the actual numbers, only their "
            "'digit echo' (digital root, 1-9) at indices you choose to inspect.\n"
            "Actions (exactly one per turn):\n"
            f"  REVEAL <i>  - reveal the digit echo at index i, for i in 0..{self.max_revealable_index} only\n"
            f"  ANSWER <a> <b> <c> - submit your predicted digit echoes (1-9 each) for indices "
            f"{self.target_indices[0]}, {self.target_indices[1]}, and {self.target_indices[2]}, "
            "ending the episode\n"
            f"You have {self.max_steps} steps total. Indices beyond "
            f"{self.max_revealable_index} cannot be REVEALed directly."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        parts = text.split()
        reward = 0.0
        terminated = False

        if not parts:
            obs = "Malformed action. Use 'REVEAL <i>' or 'ANSWER <a> <b> <c>'."
        elif parts[0].upper() == "REVEAL" and len(parts) == 2:
            idx = None
            if parts[1].lstrip("-").isdigit():
                idx = int(parts[1])
            if idx is None or idx < 0 or idx > self.max_revealable_index:
                obs = f"Invalid REVEAL index. Choose an integer index from 0 to {self.max_revealable_index}."
            elif idx in self.revealed:
                obs = f"Index {idx} was already revealed: digit echo = {self.roots[idx]}."
            else:
                self.revealed.add(idx)
                obs = f"Index {idx} digit echo = {self.roots[idx]}."
        elif parts[0].upper() == "ANSWER" and len(parts) == 4:
            guesses = []
            valid = True
            for p in parts[1:]:
                if p.lstrip("-").isdigit():
                    guesses.append(int(p))
                else:
                    valid = False
                    break
            if not valid or any(g < 1 or g > 9 for g in guesses):
                obs = "Invalid ANSWER. Give three digits 1-9, one for each target index."
            else:
                truth = [self.roots[i] for i in self.target_indices]
                correct = sum(1 for g, t in zip(guesses, truth) if g == t)
                reward = correct / 3.0
                terminated = True
                self.done = True
                obs = (
                    f"Ledger closed. You scored {correct}/3 correct digit echoes. "
                    f"True echoes were index {self.target_indices[0]}={truth[0]}, "
                    f"index {self.target_indices[1]}={truth[1]}, "
                    f"index {self.target_indices[2]}={truth[2]}."
                )
        else:
            obs = "Malformed action. Use 'REVEAL <i>' or 'ANSWER <a> <b> <c>'."

        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            obs += " Step limit reached; ledger closes unscored."

        return obs, reward, terminated, truncated, {}
