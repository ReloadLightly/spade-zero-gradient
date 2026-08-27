import random


class FaceDownParityBridgeEnv:
    """Six face-down cards, each secretly marked ODD or EVEN. One PEEK
    anchors a single card's true mark; COMPARE reveals only the relation
    (SAME/DIFFERENT mark) between two cards. The solver must chain relations
    from the anchor to deduce every card's mark, then submit one GUESS."""

    CARD_COUNT = 6
    MAX_STEPS = 10

    def _intro(self):
        cards = ", ".join(str(i) for i in range(1, self.CARD_COUNT + 1))
        return (
            "Six face-down cards are laid out, labeled " + cards + ". "
            "Each card is secretly marked ODD or EVEN. Identify every "
            "card's mark within 10 total actions.\n\n"
            "Actions (exact formats):\n"
            "  PEEK <card>            - reveal one card's TRUE mark. "
            "Usable only ONCE in the whole episode.\n"
            "  COMPARE <a> <b>        - reveal whether cards a and b have "
            "the SAME mark or a DIFFERENT mark. Usable any number of "
            "times, a and b must be distinct card labels.\n"
            "  GUESS <m1> <m2> <m3> <m4> <m5> <m6> - submit your final "
            "mark (O or E) for cards 1..6 in order. This ends the episode "
            "immediately; reward is the fraction of the 6 marks you got "
            "right.\n\n"
            "You have 10 steps total, including the final GUESS."
        )

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        while True:
            self.marks = [self.rng.choice(["O", "E"]) for _ in range(self.CARD_COUNT)]
            if len(set(self.marks)) > 1:
                break
        self.peek_used = False
        self.step_count = 0
        self.done = False
        return self._intro(), {}

    def _clamp_truncate(self, terminated):
        truncated = (not terminated) and self.step_count >= self.MAX_STEPS
        if truncated:
            self.done = True
        return truncated

    def _invalid(self, message):
        truncated = self._clamp_truncate(False)
        return message, 0.0, False, truncated, {}

    def _valid_card(self, token):
        return token.isdigit() and 1 <= int(token) <= self.CARD_COUNT

    def step(self, action):
        self.step_count += 1
        parts = action.strip().split()

        if not parts:
            return self._invalid(
                "Empty action. Use PEEK <card>, COMPARE <a> <b>, or "
                "GUESS <6 marks>. Step still counted."
            )

        cmd = parts[0].upper()

        if cmd == "PEEK":
            if len(parts) != 2 or not self._valid_card(parts[1]):
                return self._invalid(
                    "Malformed PEEK. Format: PEEK <card 1-6>. Step counted, "
                    "no reward."
                )
            if self.peek_used:
                return self._invalid(
                    "PEEK already used this episode; it cannot be repeated. "
                    "Use COMPARE to keep gathering evidence. Step counted, "
                    "no reward."
                )
            cid = int(parts[1])
            self.peek_used = True
            mark = self.marks[cid - 1]
            obs = "Card " + str(cid) + " is truly marked " + mark + "."
            truncated = self._clamp_truncate(False)
            return obs, 0.0, False, truncated, {}

        if cmd == "COMPARE":
            if len(parts) != 3 or not self._valid_card(parts[1]) or not self._valid_card(parts[2]):
                return self._invalid(
                    "Malformed COMPARE. Format: COMPARE <card 1-6> "
                    "<card 1-6>. Step counted, no reward."
                )
            a, b = int(parts[1]), int(parts[2])
            if a == b:
                return self._invalid(
                    "COMPARE requires two DIFFERENT cards. Step counted, "
                    "no reward."
                )
            relation = "SAME" if self.marks[a - 1] == self.marks[b - 1] else "DIFFERENT"
            obs = "Cards " + str(a) + " and " + str(b) + " have " + relation + " marks."
            truncated = self._clamp_truncate(False)
            return obs, 0.0, False, truncated, {}

        if cmd == "GUESS":
            if len(parts) != self.CARD_COUNT + 1:
                return self._invalid(
                    "Malformed GUESS. Format: GUESS <m1> <m2> <m3> <m4> "
                    "<m5> <m6>, one O or E per card in order. Step "
                    "counted, no reward."
                )
            guesses = [p.upper() for p in parts[1:]]
            if not all(g in ("O", "E") for g in guesses):
                return self._invalid(
                    "Each guess must be O or E. Step counted, no reward."
                )
            correct = sum(1 for g, m in zip(guesses, self.marks) if g == m)
            reward = correct / float(self.CARD_COUNT)
            self.done = True
            truth = " ".join(self.marks)
            obs = (
                "Final guess scored " + str(correct) + "/" + str(self.CARD_COUNT)
                + " correct. True marks were: " + truth + "."
            )
            return obs, reward, True, False, {}

        return self._invalid(
            "Unknown command '" + parts[0] + "'. Use PEEK, COMPARE, or "
            "GUESS. Step counted, no reward."
        )
