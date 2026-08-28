import random


class SplitStrideLedgerEnv:
    """Two-stage hidden sequence: a period-k gate selects between two linear generators."""

    def __init__(self):
        self.max_steps = 10
        self.query_max_n = 15
        self.predict_ns = [20, 21, 22, 23]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.k = self.rng.choice([2, 3])
        self.a1 = self.rng.choice([1, 2, 3])
        self.b1 = self.rng.randint(0, 5)
        while True:
            a2 = self.rng.choice([1, 2, 3])
            b2 = self.rng.randint(0, 5)
            if (a2, b2) != (self.a1, self.b1):
                self.a2, self.b2 = a2, b2
                break
        self.steps = 0
        self.done = False
        obs = (
            "Discover a hidden two-stage integer sequence S(n).\n"
            "Stage 1 (gate): a hidden period k in {2,3} decides which of two hidden linear "
            "generators produces S(n): if n % k == 0, S(n) = A(n); otherwise S(n) = B(n). "
            "Both A and B have the form a*n + b with small hidden integer constants.\n"
            "Actions:\n"
            f"  QUERY <n>   -- reveals S(n) for an integer n in [0,{self.query_max_n}]. Costs one step.\n"
            "  PREDICT <v1> <v2> <v3> <v4> -- your final guesses for S(20), S(21), S(22), S(23), "
            "in that order. Ends the episode; you earn 0.25 reward per exactly correct value.\n"
            f"You have {self.max_steps} steps total (QUERY and PREDICT both count). "
            "Gather evidence with QUERY, then submit one PREDICT."
        )
        return obs, {}

    def _S(self, n):
        if n % self.k == 0:
            return self.a1 * n + self.b1
        return self.a2 * n + self.b2

    def _malformed(self, msg):
        truncated = self.steps >= self.max_steps
        if truncated:
            self.done = True
            msg += " Step limit reached without a PREDICT; episode ends with no reward."
        return msg, 0.0, False, truncated, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.steps += 1
        action = (action or "").strip()
        parts = action.split()
        if not parts:
            return self._malformed("Malformed action. Use 'QUERY <n>' or 'PREDICT <v1> <v2> <v3> <v4>'.")

        verb = parts[0].upper()
        if verb == "QUERY":
            if len(parts) != 2:
                return self._malformed("QUERY needs exactly one integer argument.")
            try:
                n = int(parts[1])
            except ValueError:
                return self._malformed("QUERY argument must be an integer.")
            if n < 0 or n > self.query_max_n:
                return self._malformed(f"QUERY n must be between 0 and {self.query_max_n}.")
            val = self._S(n)
            obs = f"S({n}) = {val}."
            truncated = self.steps >= self.max_steps
            if truncated:
                obs += " Step limit reached without a PREDICT; episode ends with no reward."
                self.done = True
            return obs, 0.0, False, truncated, {}

        elif verb == "PREDICT":
            if len(parts) != 5:
                return self._malformed("PREDICT needs exactly 4 integer values, for S(20) S(21) S(22) S(23).")
            try:
                guesses = [int(x) for x in parts[1:]]
            except ValueError:
                return self._malformed("PREDICT values must be integers.")
            correct = 0
            details = []
            for idx, n in enumerate(self.predict_ns):
                true_v = self._S(n)
                if guesses[idx] == true_v:
                    correct += 1
                    details.append(f"S({n}) correct")
                else:
                    details.append(f"S({n}) expected {true_v}, got {guesses[idx]}")
            reward = 0.25 * correct
            self.done = True
            obs = f"PREDICT evaluated: {correct}/4 correct. " + "; ".join(details)
            return obs, reward, True, False, {}

        else:
            return self._malformed("Unknown action verb. Use QUERY or PREDICT.")
