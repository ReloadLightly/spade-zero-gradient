import random


class SymbolGrammarEnv:
    def __init__(self):
        self.alphabet = ['P', 'Q', 'R', 'S']
        self.length = 5
        self.max_steps = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False

        self.allowed = {}
        for s in self.alphabet:
            nexts = self.rng.sample(self.alphabet, 2)
            self.allowed[s] = set(nexts)

        self.seed_valid = self._random_walk('P')
        self.seed_invalid = self._make_invalid_example()
        self.seed_edges = self._edges_of_word(self.seed_valid)
        self.confirmed_new_edges = set()
        self.milestone_awarded = False

        obs = (
            "SYMBOL GRAMMAR GAME.\n"
            f"Alphabet: {', '.join(self.alphabet)}. A word is a length-{self.length} "
            "string that always starts with 'P'. A word is VALID iff every "
            "consecutive pair of symbols in it belongs to a hidden set of allowed "
            "transitions (the grammar). Your goal is to construct a NEW valid word "
            "(different from the example below) within the step budget.\n"
            f"Given VALID example: {self.seed_valid}\n"
            f"Given INVALID example: {self.seed_invalid}\n"
            "Actions (one per step):\n"
            "  TEST <word>   - probe a length-5 word starting with P; you learn "
            "either 'VALID' or the index of the FIRST illegal adjacent pair (the "
            "prefix before that index is confirmed legal).\n"
            "  SUBMIT <word> - final answer: a length-5 word starting with P, "
            "different from the given valid example.\n"
            f"You have {self.max_steps} steps total (TEST and SUBMIT both count)."
        )
        return obs, {}

    def _random_walk(self, start):
        path = [start]
        cur = start
        for _ in range(self.length - 1):
            nxt = self.rng.choice(sorted(self.allowed[cur]))
            path.append(nxt)
            cur = nxt
        return ''.join(path)

    def _edges_of_word(self, word):
        return set((word[i], word[i + 1]) for i in range(len(word) - 1))

    def _make_invalid_example(self):
        idx = self.rng.randint(1, self.length - 1)
        prefix = self.seed_valid[:idx]
        prev = prefix[-1]
        forbidden = [s for s in self.alphabet if s not in self.allowed[prev]]
        wrong = self.rng.choice(forbidden)
        tail_len = self.length - idx - 1
        tail = ''.join(self.rng.choice(self.alphabet) for _ in range(tail_len))
        return prefix + wrong + tail

    def _check_word(self, word):
        edges = list(zip(word, word[1:]))
        for i, (a, b) in enumerate(edges):
            if b not in self.allowed[a]:
                return False, i, edges[:i]
        return True, None, edges

    def _malformed(self, word):
        if word is None:
            return True
        if len(word) != self.length:
            return True
        if any(c not in self.alphabet for c in word):
            return True
        if word[0] != 'P':
            return True
        return False

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        tokens = (action or "").strip().split()
        verb = tokens[0].upper() if tokens else ""
        word = tokens[1].upper() if len(tokens) > 1 else None

        if verb not in ("TEST", "SUBMIT"):
            obs = "Malformed action. Use 'TEST <word>' or 'SUBMIT <word>'."
            truncated = self.step_count >= self.max_steps
            if truncated:
                self.done = True
            return obs, 0.0, False, truncated, {}

        if self._malformed(word):
            obs = (
                f"Malformed word. It must be length {self.length}, use only "
                f"{'/'.join(self.alphabet)}, and start with P."
            )
            truncated = self.step_count >= self.max_steps
            if truncated:
                self.done = True
            return obs, 0.0, False, truncated, {}

        valid, break_idx, prefix_edges = self._check_word(word)

        if verb == "TEST":
            new_edges = set(prefix_edges) - self.seed_edges - self.confirmed_new_edges
            reward = 0.0
            if new_edges:
                self.confirmed_new_edges |= new_edges
            if not self.milestone_awarded and len(self.confirmed_new_edges) >= 2:
                reward += 0.3
                self.milestone_awarded = True
            if valid:
                obs = f"VALID: {word} satisfies the grammar in full."
            else:
                a, b = word[break_idx], word[break_idx + 1]
                obs = (
                    f"INVALID at edge {break_idx}: the pair '{a}{b}' "
                    f"(positions {break_idx}-{break_idx + 1}) is not allowed. "
                    f"Everything before that edge is confirmed legal."
                )
            truncated = self.step_count >= self.max_steps
            if truncated:
                self.done = True
            return obs, reward, False, truncated, {}

        # SUBMIT
        self.done = True
        if valid and word != self.seed_valid:
            obs = f"Correct! {word} is a newly deduced valid word."
            return obs, 0.7, True, False, {}
        if word == self.seed_valid:
            obs = "Rejected: that is just the given example, not a new deduction."
            return obs, 0.0, True, False, {}
        a, b = word[break_idx], word[break_idx + 1]
        obs = f"Rejected: invalid at edge {break_idx} (pair '{a}{b}' not allowed)."
        return obs, 0.0, True, False, {}
