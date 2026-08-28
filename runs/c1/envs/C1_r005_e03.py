import random


class MurderTimelineEnv:
    CODES = ['P', 'Q', 'R', 'S', 'T', 'U']
    TAG_POOL = [
        "the raised voices in the library",
        "the shattered vase in the hall",
        "the locked garden gate",
        "the missing silver key",
        "the telephone call from town",
        "the scream from the east wing",
    ]
    WITNESS_NAMES = [
        "the Housekeeper", "the Butler", "the Gardener",
        "the Stableboy", "the Cook", "the Constable",
    ]
    MAX_INTERVIEWS = 5
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.order = list(self.CODES)
        self.rng.shuffle(self.order)

        shuffled_tags = self.rng.sample(self.TAG_POOL, len(self.TAG_POOL))
        self.tag_map = dict(zip(self.CODES, shuffled_tags))

        self.true_adjacent = set()
        facts = []
        for i in range(len(self.order) - 1):
            a, b = self.order[i], self.order[i + 1]
            self.true_adjacent.add((a, b))
            facts.append(('link', a, b))

        anchor_kind = self.rng.choice(['first', 'last'])
        anchor_event = self.order[0] if anchor_kind == 'first' else self.order[-1]
        facts.append(('anchor', anchor_kind, anchor_event))

        self.rng.shuffle(facts)
        witness_ids = [f'W{i + 1}' for i in range(6)]
        self.witnesses = {}
        for wid, name, fact in zip(witness_ids, self.WITNESS_NAMES, facts):
            self.witnesses[wid] = {'name': name, 'fact': fact}

        self.asked = set()
        self.step_count = 0
        self.done = False

        lines = []
        lines.append(
            "A body was found at the manor. Six events of that day must be placed "
            "in chronological order. Each event is known only by a code:"
        )
        for code in self.CODES:
            lines.append(f"  {code}: {self.tag_map[code]}")
        lines.append(
            "Six witnesses (W1-W6) each recall one fact about the order, but the "
            "inquest permits interviewing at most 5 of the 6 before you must commit."
        )
        lines.append(
            "Actions: 'ASK <witness>' (e.g. 'ASK W3') to hear a witness's fact, or "
            "'ORDER <c1> <c2> <c3> <c4> <c5> <c6>' (e.g. 'ORDER P Q R S T U') to "
            "submit the full order from earliest to latest. ORDER ends the episode."
        )
        lines.append(f"You have {self.MAX_STEPS} steps total.")
        return "\n".join(lines), {}

    def _fact_text(self, fact):
        kind = fact[0]
        if kind == 'link':
            _, a, b = fact
            return (
                f"'{self.tag_map[a]}' happened immediately before "
                f"'{self.tag_map[b]}' ({a} right before {b})."
            )
        _, anchor_kind, ev = fact
        word = 'first' if anchor_kind == 'first' else 'last'
        return f"'{self.tag_map[ev]}' ({ev}) was the very {word} thing that happened that day."

    def _check_truncate(self):
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            return True
        return False

    def step(self, action):
        if self.done:
            return "The inquest has concluded.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        upper = text.upper()

        if upper.startswith("ASK"):
            rest = text[3:].strip().upper()
            if rest not in self.witnesses:
                obs = f"There is no witness '{rest}'. Valid witnesses are W1-W6."
                return obs, 0.0, False, self._check_truncate(), {}
            if rest in self.asked:
                obs = f"{self.witnesses[rest]['name']} has nothing new to add."
                return obs, 0.0, False, self._check_truncate(), {}
            if len(self.asked) >= self.MAX_INTERVIEWS:
                obs = (
                    "The coroner cuts you off: you have used all 5 permitted "
                    "interviews. Submit your ORDER."
                )
                return obs, 0.0, False, self._check_truncate(), {}
            self.asked.add(rest)
            w = self.witnesses[rest]
            obs = f"{w['name']} ({rest}) says: {self._fact_text(w['fact'])}"
            return obs, 0.0, False, self._check_truncate(), {}

        if upper.startswith("ORDER"):
            parts = text[5:].strip().upper().split()
            if len(parts) != 6 or sorted(parts) != sorted(self.CODES):
                obs = (
                    "Malformed order: give exactly the six codes P Q R S T U, each "
                    "once, e.g. 'ORDER P Q R S T U'."
                )
                return obs, 0.0, False, self._check_truncate(), {}
            correct = sum(
                1 for i in range(5) if (parts[i], parts[i + 1]) in self.true_adjacent
            )
            reward = correct * 0.2
            self.done = True
            obs = (
                f"You submit the order {' '.join(parts)}. The true order was "
                f"{' '.join(self.order)}. You correctly placed {correct} of 5 "
                f"adjacent pairs."
            )
            return obs, reward, True, False, {}

        obs = "Unrecognized action. Use 'ASK <witness>' or 'ORDER <c1> ... <c6>'."
        return obs, 0.0, False, self._check_truncate(), {}
