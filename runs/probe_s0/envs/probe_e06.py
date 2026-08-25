import random


class TreasureMapEnv:
    GRID_SIZE = 5
    LANDMARK_POOL = ["WELL", "TOWER", "CAVE", "SHRINE", "OAK"]
    NUM_LANDMARKS = 3
    HARD_STEP_LIMIT = 8

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.milestone_a_awarded = False
        self.milestone_b_awarded = False

        landmark_names = self.rng.sample(self.LANDMARK_POOL, self.NUM_LANDMARKS)
        all_cells = [
            (r, c)
            for r in range(1, self.GRID_SIZE + 1)
            for c in range(1, self.GRID_SIZE + 1)
        ]

        for _ in range(2000):
            pool = list(all_cells)
            self.rng.shuffle(pool)
            chosen = pool[: 1 + self.NUM_LANDMARKS]
            treasure = chosen[0]
            landmarks = dict(zip(landmark_names, chosen[1:]))
            candidates = self._filter_candidates(
                all_cells, landmarks, list(landmark_names), treasure
            )
            if len(candidates) == 1 and candidates[0] == treasure:
                self.treasure = treasure
                self.landmarks = landmarks
                self.landmark_names = landmark_names
                break
        else:
            self.treasure = treasure
            self.landmarks = landmarks
            self.landmark_names = landmark_names

        self.asked = set()

        obs = (
            "You are hunting buried treasure on a {n}x{n} grid. Rows and columns "
            "are numbered 1-{n}. Cells are named R<row>C<col>, e.g. R3C2.\n"
            "Three landmarks are hidden somewhere on the grid: {lm}. None of them "
            "sits on the treasure cell.\n"
            "Goal: find the exact treasure cell within {steps} steps.\n"
            "Actions (send exactly one per turn):\n"
            "  ASK <LANDMARK>   -- e.g. 'ASK {ex}'. Reveals the Chebyshev "
            "('king-move') distance from that landmark to the treasure: "
            "distance = max(|row difference|, |column difference|).\n"
            "  GUESS R<row>C<col> -- e.g. 'GUESS R3C2'. Ends the episode.\n"
            "You may ask each landmark at most once for a fresh clue."
        ).format(
            n=self.GRID_SIZE,
            lm=", ".join(self.landmark_names),
            steps=self.HARD_STEP_LIMIT,
            ex=self.landmark_names[0],
        )
        return obs, {}

    def _chebyshev(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _filter_candidates(self, all_cells, landmarks, revealed_names, treasure):
        landmark_cells = set(landmarks.values())
        candidates = [c for c in all_cells if c not in landmark_cells]
        for name in revealed_names:
            lpos = landmarks[name]
            d = self._chebyshev(treasure, lpos)
            candidates = [c for c in candidates if self._chebyshev(c, lpos) == d]
        return candidates

    def _current_candidates(self):
        all_cells = [
            (r, c)
            for r in range(1, self.GRID_SIZE + 1)
            for c in range(1, self.GRID_SIZE + 1)
        ]
        return self._filter_candidates(
            all_cells, self.landmarks, list(self.asked), self.treasure
        )

    def _parse_cell(self, token):
        token = token.strip().upper()
        if len(token) < 4 or token[0] != "R" or "C" not in token[1:]:
            return None
        try:
            c_index = token.index("C", 1)
            row = int(token[1:c_index])
            col = int(token[c_index + 1 :])
        except ValueError:
            return None
        if not (1 <= row <= self.GRID_SIZE and 1 <= col <= self.GRID_SIZE):
            return None
        return (row, col)

    def step(self, action):
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, self.terminated, self.truncated, {}

        self.step_count += 1
        reward = 0.0
        action = (action or "").strip()
        parts = action.split(None, 1)
        verb = parts[0].upper() if parts else ""

        if verb == "ASK" and len(parts) == 2:
            name = parts[1].strip().upper()
            if name not in self.landmark_names:
                obs = (
                    "Unknown landmark '{n}'. Valid landmarks: {lm}."
                ).format(n=name, lm=", ".join(self.landmark_names))
            elif name in self.asked:
                obs = "You already asked {n}; no new information.".format(n=name)
            else:
                self.asked.add(name)
                d = self._chebyshev(self.treasure, self.landmarks[name])
                candidates = self._current_candidates()
                obs = (
                    "{n} reports the treasure is exactly {d} step(s) away "
                    "(king-move distance). {k} cell(s) remain consistent with "
                    "all clues so far."
                ).format(n=name, d=d, k=len(candidates))
                if len(candidates) <= 6 and not self.milestone_a_awarded:
                    reward += 0.15
                    self.milestone_a_awarded = True
                    obs += " (Progress: candidate set narrowed sharply.)"
                if len(candidates) == 1 and not self.milestone_b_awarded:
                    reward += 0.15
                    self.milestone_b_awarded = True
                    obs += " (Progress: location uniquely determined.)"
        elif verb == "GUESS" and len(parts) == 2:
            cell = self._parse_cell(parts[1])
            if cell is None:
                obs = "Malformed guess. Use GUESS R<row>C<col>, e.g. GUESS R3C2."
            elif cell == self.treasure:
                reward = 0.7
                self.terminated = True
                obs = "Correct! The treasure was at {c}.".format(
                    c=self._fmt(cell)
                )
            else:
                self.terminated = True
                obs = "Wrong. The treasure was actually at {c}.".format(
                    c=self._fmt(self.treasure)
                )
        else:
            obs = (
                "Unrecognized action. Use 'ASK <LANDMARK>' or "
                "'GUESS R<row>C<col>'."
            )

        if not self.terminated and self.step_count >= self.HARD_STEP_LIMIT:
            self.truncated = True
            obs += " Step limit reached; episode over."

        return obs, reward, self.terminated, self.truncated, {}

    def _fmt(self, cell):
        return "R{r}C{c}".format(r=cell[0], c=cell[1])
