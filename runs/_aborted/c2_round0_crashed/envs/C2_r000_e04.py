import random


class MosaicBorderEnv:
    ALPHABET = ['A', 'B', 'C', 'D']
    BORDER_LEN = 20
    WINDOW_LEN = 7
    TARGET_START = 7
    TARGET_LEN = 4
    MAX_STEPS = 10

    def _generate_border(self, motif, L, mode, length):
        border = []
        for i in range(length):
            block = i // L
            j = i % L
            if mode == 'repeat':
                symbol = motif[j]
            else:
                symbol = motif[j] if block % 2 == 0 else motif[L - 1 - j]
            border.append(symbol)
        return border

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.L = self.rng.choice([3, 4, 5])
        while True:
            motif = [self.rng.choice(self.ALPHABET) for _ in range(self.L)]
            if len(set(motif)) >= 2 and motif != motif[::-1]:
                break
        self.motif = motif
        self.mode = self.rng.choice(['repeat', 'mirror'])
        self.border = self._generate_border(motif, self.L, self.mode, self.BORDER_LEN)
        self.window = self.border[:self.WINDOW_LEN]
        self.target = self.border[self.TARGET_START:self.TARGET_START + self.TARGET_LEN]
        self.steps = 0
        self.done = False
        self.length_reward_given = False

        window_str = " ".join("{}:{}".format(i, s) for i, s in enumerate(self.window))
        target_lo = self.TARGET_START
        target_hi = self.TARGET_START + self.TARGET_LEN - 1
        obs = (
            "MOSAIC BORDER. Tiles are single letters from {A,B,C,D} arranged left to "
            "right by index. The first {wlen} tiles (indices 0-{wmax}) are:\n"
            "{window}\n"
            "A hidden rule extends this border forever. Indices {tlo}-{thi} are masked "
            "and must be predicted. You have up to {maxs} total actions.\n"
            "Actions (send exactly one per turn):\n"
            "  PEEK <index>            reveal the tile at a chosen index (0-{bmax}); "
            "the masked target range {tlo}-{thi} cannot be peeked.\n"
            "  GUESS_LENGTH <n>        guess the hidden motif length (an integer, "
            "typically small); a correct guess earns partial credit once.\n"
            "  EXTEND <s1> <s2> <s3> <s4>   submit your final 4-letter prediction for "
            "indices {tlo},{tlo1},{tlo2},{thi} in order (space-separated single "
            "letters); this ends the episode.\n"
            "Reward: 0.3 for a correct GUESS_LENGTH, plus up to 0.7 for EXTEND scaled "
            "by how many of the 4 predicted tiles are correct."
        ).format(
            wlen=self.WINDOW_LEN, wmax=self.WINDOW_LEN - 1, window=window_str,
            tlo=target_lo, thi=target_hi, tlo1=target_lo + 1, tlo2=target_lo + 2,
            maxs=self.MAX_STEPS, bmax=self.BORDER_LEN - 1,
        )
        return obs, {}

    def _corrective(self, msg, reward=0.0):
        terminated = False
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.done = True
        return msg, reward, terminated, truncated, {}

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}
        self.steps += 1
        text = (action or "").strip()
        parts = text.split()

        if not parts:
            return self._corrective(
                "Empty action. Use PEEK <index>, GUESS_LENGTH <n>, or EXTEND <4 letters>."
            )

        cmd = parts[0].upper()

        if cmd == "PEEK":
            if len(parts) != 2 or not parts[1].lstrip('-').isdigit():
                return self._corrective("Malformed PEEK. Format: PEEK <index>.")
            idx = int(parts[1])
            lo, hi = self.TARGET_START, self.TARGET_START + self.TARGET_LEN - 1
            if idx < 0 or idx >= self.BORDER_LEN:
                return self._corrective(
                    "Index out of range 0-{}.".format(self.BORDER_LEN - 1)
                )
            if lo <= idx <= hi:
                return self._corrective(
                    "Index {} is inside the masked target range {}-{}; it cannot be peeked.".format(idx, lo, hi)
                )
            return self._corrective("Tile at index {} is {}.".format(idx, self.border[idx]))

        if cmd == "GUESS_LENGTH":
            if len(parts) != 2 or not parts[1].lstrip('-').isdigit():
                return self._corrective("Malformed GUESS_LENGTH. Format: GUESS_LENGTH <n>.")
            n = int(parts[1])
            if n == self.L:
                if self.length_reward_given:
                    return self._corrective("Correct length, but credit was already given.")
                self.length_reward_given = True
                return self._corrective("Correct! The motif length is confirmed.", reward=0.3)
            return self._corrective("That is not the correct motif length.")

        if cmd == "EXTEND":
            tokens = parts[1:]
            if len(tokens) != self.TARGET_LEN or any(
                len(t) != 1 or t.upper() not in self.ALPHABET for t in tokens
            ):
                return self._corrective(
                    "Malformed EXTEND. Provide exactly {} single letters from {{A,B,C,D}}, "
                    "space-separated.".format(self.TARGET_LEN)
                )
            guess = [t.upper() for t in tokens]
            correct = sum(1 for g, t in zip(guess, self.target) if g == t)
            reward = 0.7 * correct / self.TARGET_LEN
            self.done = True
            obs = "EXTEND submitted: {}/{} tiles correct. Episode ended.".format(
                correct, self.TARGET_LEN
            )
            return obs, reward, True, False, {"correct_tiles": correct}

        return self._corrective(
            "Unrecognized command '{}'. Use PEEK, GUESS_LENGTH, or EXTEND.".format(parts[0])
        )
