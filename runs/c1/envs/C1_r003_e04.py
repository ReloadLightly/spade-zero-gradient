import random


class MirrorRotorEnv:
    ALPHABET = ['A', 'B', 'C', 'D']
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.rot_map = {}
        self.targets = []
        self.step_count = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.rot_map = self._gen_rot_map()
        self.targets = self._gen_targets()
        self.step_count = 0
        self.done = False
        lines = [
            "You are analyzing MIRROR-GLYPHS: strings built from glyphs A, B, C, D.",
            "A hidden pairing rule links each glyph to a rotation partner (possibly itself) -- like rotating a printed character 180 degrees, the way '6' becomes '9' while 'S' stays 'S'.",
            "A string is MIRROR-SYMMETRIC if replacing every glyph with its rotation partner and reading the result backward reproduces the original string exactly.",
            "Example of the mechanic (not the real rule): if P were paired with Q, then 'PQ' and 'QP' would each be mirror-symmetric, but 'PP' would only be mirror-symmetric if P were paired with itself.",
            "",
            "Classify these 4 target strings as SYM or ASYM under the hidden rule:",
        ]
        for i, (s, _) in enumerate(self.targets, 1):
            lines.append(f"  {i}. {s}")
        lines += [
            "",
            "You may NOT probe the target strings directly -- test other glyph strings to learn the hidden pairing instead.",
            "Actions (budget: 10 steps total):",
            "  TEST <string>   -- 1 to 4 characters from A/B/C/D, e.g. TEST AC -- reports whether that probe string is mirror-symmetric.",
            "  SUBMIT <c1> <c2> <c3> <c4> -- e.g. SUBMIT SYM ASYM SYM ASYM -- final classification of the 4 targets in order; ends the episode.",
            "Each correct classification is worth 0.25 reward (max total 1.0). A malformed action costs a step and earns no reward.",
        ]
        return "\n".join(lines), {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        parts = (action or "").strip().split()
        if not parts:
            return self._malformed("Empty action. Use 'TEST <string>' or 'SUBMIT <c1> <c2> <c3> <c4>'.")

        cmd = parts[0].upper()

        if cmd == "TEST":
            if len(parts) != 2:
                return self._malformed("TEST needs exactly one glyph string, e.g. TEST AC.")
            query = parts[1].upper()
            if not (1 <= len(query) <= 4) or any(c not in self.ALPHABET for c in query):
                return self._malformed("TEST string must be 1-4 characters using only A, B, C, D.")
            if any(query == s for s, _ in self.targets):
                return self._malformed("Cannot TEST a target string directly -- probe a different string.")
            self.step_count += 1
            result = self._is_symmetric(query)
            obs = f"TEST '{query}': {'MIRROR-SYMMETRIC' if result else 'not symmetric'} under the hidden rule."
            return self._maybe_truncate(obs)

        if cmd == "SUBMIT":
            if len(parts) != 5:
                return self._malformed("SUBMIT needs exactly 4 labels, e.g. SUBMIT SYM ASYM SYM ASYM.")
            labels = [p.upper() for p in parts[1:5]]
            if any(l not in ("SYM", "ASYM") for l in labels):
                return self._malformed("Each label must be SYM or ASYM.")
            self.step_count += 1
            correct = 0
            report = []
            for i, ((s, truth), guess) in enumerate(zip(self.targets, labels), 1):
                truth_label = "SYM" if truth else "ASYM"
                is_right = guess == truth_label
                correct += int(is_right)
                report.append(f"  {i}. {s}: guessed {guess}, actual {truth_label} -- {'correct' if is_right else 'wrong'}")
            reward = 0.25 * correct
            self.done = True
            obs = "Results:\n" + "\n".join(report) + f"\nScore: {correct}/4 correct."
            return obs, reward, True, False, {"correct": correct}

        return self._malformed("Unrecognized action. Use 'TEST <string>' or 'SUBMIT <c1> <c2> <c3> <c4>'.")

    def _malformed(self, message):
        self.step_count += 1
        return self._maybe_truncate(message, reward=0.0)

    def _maybe_truncate(self, obs, reward=0.0):
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            return obs + "\nStep limit reached without a SUBMIT -- episode over.", reward, False, True, {}
        return obs, reward, False, False, {}

    def _is_symmetric(self, s):
        n = len(s)
        for i in range(n):
            if self.rot_map[s[i]] != s[n - 1 - i]:
                return False
        return True

    def _gen_rot_map(self):
        for _ in range(30):
            letters = list(self.ALPHABET)
            self.rng.shuffle(letters)
            mapping = {}
            i = 0
            while i < len(letters):
                remaining = len(letters) - i
                if remaining >= 2 and self.rng.random() < 0.6:
                    a, b = letters[i], letters[i + 1]
                    mapping[a] = b
                    mapping[b] = a
                    i += 2
                else:
                    mapping[letters[i]] = letters[i]
                    i += 1
            if any(mapping[x] != x for x in mapping):
                return mapping
        mapping = {c: c for c in self.ALPHABET}
        mapping[self.ALPHABET[0]] = self.ALPHABET[1]
        mapping[self.ALPHABET[1]] = self.ALPHABET[0]
        return mapping

    def _gen_targets(self):
        best = None
        for _ in range(200):
            candidate = []
            for _ in range(4):
                length = self.rng.randint(2, 4)
                s = ''.join(self.rng.choice(self.ALPHABET) for _ in range(length))
                candidate.append((s, self._is_symmetric(s)))
            strings = [s for s, _ in candidate]
            labels = [t for _, t in candidate]
            if len(set(strings)) == 4 and any(labels) and not all(labels):
                return candidate
            best = candidate
        return best
