import random


class DigitRootVendingEnv:
    def __init__(self):
        self.rng = None
        self.K = None
        self.tokens = {}
        self.used_tokens = set()
        self.challenge_value = None
        self.true_class = None
        self.true_answer = None
        self.step_count = 0
        self.informative_used = False
        self.class_reward_given = False
        self.done = False

    def _digital_root(self, n):
        if n <= 0:
            return 0
        return 1 + (n - 1) % 9

    def _make_value(self, target_dr):
        m = self.rng.randint(3, 15)
        return 9 * m + target_dr

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.K = self.rng.choice([2, 3, 4, 5, 6, 7, 8])
        self.step_count = 0
        self.used_tokens = set()
        self.informative_used = False
        self.class_reward_given = False
        self.done = False

        names = ["ALPHA", "BRAVO", "CIRCA"]
        drs = [3, 6, 9]
        self.rng.shuffle(drs)
        self.tokens = {name: self._make_value(d) for name, d in zip(names, drs)}

        coprime_choices = [1, 2, 4, 5, 7, 8]
        challenge_dr = self.rng.choice(coprime_choices)
        self.challenge_value = self._make_value(challenge_dr)

        self.true_class = self.K % 3
        self.true_answer = self._digital_root(self.K * self.challenge_value)

        catalog = ", ".join(f"{n}={v}" for n, v in self.tokens.items())
        obs = (
            "DIGIT-ROOT VENDING MACHINE. It holds a hidden integer multiplier K (2-8). "
            "Feed it a token of face value V; it truly computes K*V but reports ONLY the "
            "digital root (repeated digit-sum down to one digit, 1-9) of that result -- "
            "never the real number.\n"
            f"Catalog tokens (each usable once): {catalog}\n"
            f"CHALLENGE token value = {self.challenge_value}. Your job: predict the digital "
            "root the machine would report for CHALLENGE, WITHOUT ever feeding CHALLENGE "
            "to the machine.\n"
            "Actions (exactly one per turn):\n"
            "  PROBE <name>  -- feed a catalog token, see its reported digital root\n"
            "  CLASS <0|1|2> -- lock in your belief about K mod 3, for partial credit\n"
            "  COMMIT <digit 1-9> -- submit final digital-root prediction for CHALLENGE, ends episode\n"
            "You have 10 steps total. Malformed actions are corrected with no reward and no step lost."
        )
        return obs, {}

    def step(self, action):
        info = {}
        if self.done:
            return "Episode already finished.", 0.0, True, False, info

        text = (action or "").strip()
        parts = text.split()
        reward = 0.0
        terminated = False
        truncated = False

        if not parts or parts[0].upper() not in ("PROBE", "CLASS", "COMMIT"):
            obs = "Malformed action. Use 'PROBE <name>', 'CLASS <0|1|2>', or 'COMMIT <digit>'."
            return obs, 0.0, False, False, info

        cmd = parts[0].upper()

        if cmd == "PROBE":
            if len(parts) != 2 or parts[1].upper() not in self.tokens:
                obs = "Malformed PROBE. Use 'PROBE <name>' with a valid catalog name."
                return obs, 0.0, False, False, info
            name = parts[1].upper()
            if name in self.used_tokens:
                obs = f"Token {name} was already used and has no more supply."
                self.step_count += 1
            else:
                self.used_tokens.add(name)
                value = self.tokens[name]
                out_dr = self._digital_root(self.K * value)
                obs = f"Machine reports: digital root of the receipt for {name} (value {value}) is {out_dr}."
                if value % 9 != 0 and reward == 0.0 and not self.informative_used:
                    self.informative_used = True
                    reward = 0.3
                self.step_count += 1

        elif cmd == "CLASS":
            if len(parts) != 2 or parts[1] not in ("0", "1", "2"):
                obs = "Malformed CLASS. Use 'CLASS <0|1|2>'."
                return obs, 0.0, False, False, info
            guess_class = int(parts[1])
            if not self.class_reward_given and guess_class == self.true_class:
                reward = 0.2
                self.class_reward_given = True
            obs = f"Class {guess_class} recorded."
            self.step_count += 1

        else:  # COMMIT
            if len(parts) != 2 or not parts[1].isdigit() or not (1 <= int(parts[1]) <= 9):
                obs = "Malformed COMMIT. Use 'COMMIT <digit 1-9>'."
                return obs, 0.0, False, False, info
            guess = int(parts[1])
            self.step_count += 1
            if guess == self.true_answer:
                reward = 0.5
                obs = f"COMMIT accepted: {guess} matches the machine's true output. Episode complete."
            else:
                obs = f"COMMIT accepted: {guess} does not match the machine's true output. Episode complete."
            terminated = True
            self.done = True

        if not terminated and self.step_count >= 10:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode over."

        return obs, reward, terminated, truncated, info
