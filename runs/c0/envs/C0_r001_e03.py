import random


class BanquetSeatingEnv:
    """Deduce a hidden seating permutation of 5 guests across 5 seats
    using pairwise-constraint clues and a bulls-only match-count signal."""

    NAME_POOL = [
        "Aria", "Boden", "Corin", "Dahlia", "Elric",
        "Fenn", "Greta", "Halvard",
    ]
    NUM_SEATS = 5
    MAX_STEPS = 10
    THRESHOLDS = [(2, 0.1), (3, 0.2), (4, 0.3), (5, 0.4)]

    def __init__(self):
        self.rng = None
        self.guests = []
        self.true_perm = []
        self.steps = 0
        self.claimed = set()
        self.best_correct = 0
        self.terminated = False
        self.truncated = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.guests = self.rng.sample(self.NAME_POOL, self.NUM_SEATS)
        self.true_perm = self.guests[:]
        self.rng.shuffle(self.true_perm)

        self.steps = 0
        self.claimed = set()
        self.best_correct = 0
        self.terminated = False
        self.truncated = False

        clue_a, clue_b = self._starting_clues()

        obs = (
            "BANQUET SEATING PUZZLE\n"
            f"5 guests must be seated in seats 1-5 (seat 1 is nearest the "
            f"head table, seat 5 is farthest): {', '.join(self.guests)}.\n"
            "The true seat order is hidden. Your goal: submit the exact "
            "seat order (seat 1 through seat 5) within the step limit.\n"
            f"Starting clues (guaranteed true): {clue_a} {clue_b}\n"
            "Action format: 'GUESS name1 name2 name3 name4 name5' where "
            "the names are given in seat-1-to-seat-5 order and each of "
            "the 5 listed guests appears exactly once.\n"
            "After each guess you are told only how many guests are in "
            "their exact correct seat (no positional detail beyond the "
            f"count). You have {self.MAX_STEPS} steps total."
        )
        info = {"guests": self.guests[:]}
        return obs, info

    def _starting_clues(self):
        idx = {g: i for i, g in enumerate(self.true_perm)}
        g_a = self.guests[0]
        wrong_seat = self.rng.choice(
            [s for s in range(self.NUM_SEATS) if s != idx[g_a]]
        )
        clue_a = f"{g_a} is NOT seated in seat {wrong_seat + 1}."

        other_candidates = [g for g in self.guests if g != g_a]
        g_b = self.rng.choice(other_candidates)
        if idx[g_a] < idx[g_b]:
            clue_b = f"{g_a} is seated to the left of {g_b}."
        else:
            clue_b = f"{g_b} is seated to the left of {g_a}."

        return clue_a, clue_b

    def step(self, action):
        if self.terminated or self.truncated:
            return (
                "Episode already ended; call reset() to play again.",
                0.0, self.terminated, self.truncated, {}
            )

        self.steps += 1
        parsed = self._parse(action)

        if parsed is None:
            obs = (
                "Malformed action. Use exactly: 'GUESS name1 name2 name3 "
                f"name4 name5' using each of {', '.join(self.guests)} "
                "exactly once, in seat-1-to-seat-5 order."
            )
            reward = 0.0
            terminated = False
            truncated = self.steps >= self.MAX_STEPS
            self.truncated = truncated
            return obs, reward, terminated, truncated, {"steps": self.steps}

        correct_count = sum(
            1 for i in range(self.NUM_SEATS) if parsed[i] == self.true_perm[i]
        )
        if correct_count > self.best_correct:
            self.best_correct = correct_count

        reward = 0.0
        for threshold, bonus in self.THRESHOLDS:
            if correct_count >= threshold and threshold not in self.claimed:
                reward += bonus
                self.claimed.add(threshold)

        success = correct_count == self.NUM_SEATS
        terminated = success
        truncated = (not success) and self.steps >= self.MAX_STEPS
        self.terminated = terminated
        self.truncated = truncated

        if success:
            obs = (
                f"Correct! All {self.NUM_SEATS} guests are in their exact "
                "seats. Banquet seating solved."
            )
        elif truncated:
            obs = (
                f"Step limit reached. Your last guess had {correct_count} "
                f"guest(s) in the correct seat (best so far: "
                f"{self.best_correct}). Episode over."
            )
        else:
            obs = (
                f"{correct_count} of {self.NUM_SEATS} guests are in the "
                f"correct seat (best so far: {self.best_correct}). "
                f"{self.MAX_STEPS - self.steps} step(s) remain."
            )

        info = {"correct_count": correct_count, "steps": self.steps}
        return obs, reward, terminated, truncated, info

    def _parse(self, action):
        if not isinstance(action, str):
            return None
        tokens = action.strip().split()
        if len(tokens) != self.NUM_SEATS + 1:
            return None
        if tokens[0].upper() != "GUESS":
            return None
        names_lower = {g.lower(): g for g in self.guests}
        guessed = []
        seen = set()
        for tok in tokens[1:]:
            key = tok.lower()
            if key not in names_lower or key in seen:
                return None
            seen.add(key)
            guessed.append(names_lower[key])
        if len(guessed) != self.NUM_SEATS:
            return None
        return guessed
