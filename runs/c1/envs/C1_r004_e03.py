import random


class StandingsReconstructEnv:
    def __init__(self):
        self.teams = ['A', 'B', 'C', 'D']
        self.games = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        winners = None
        wins = None
        for _ in range(50):
            trial_winners = [self.rng.choice([0, 1]) for _ in self.games]
            trial_wins = [0, 0, 0, 0]
            for (i, j), w in zip(self.games, trial_winners):
                if w == 0:
                    trial_wins[i] += 1
                else:
                    trial_wins[j] += 1
            winners = trial_winners
            wins = trial_wins
            if sorted(wins) != [0, 1, 2, 3]:
                break
        self.winners = winners
        self.wins = wins
        self.peek_budget = 3
        self.peeks_used = 0
        self.peeked = [False] * 6
        self.step_count = 0
        self.done = False
        obs = self._intro()
        return obs, {}

    def _game_label(self, gi):
        i, j = self.games[gi]
        return "{} vs {}".format(self.teams[i], self.teams[j])

    def _intro(self):
        lines = []
        lines.append(
            "Hidden round-robin: 4 teams (A,B,C,D), each pair played exactly "
            "one game, no draws. Final win totals (standings): " +
            ", ".join("{}={}".format(t, w) for t, w in zip(self.teams, self.wins))
        )
        lines.append("The 6 games, fixed numbering:")
        for gi in range(6):
            lines.append("  {}) {}".format(gi + 1, self._game_label(gi)))
        lines.append(
            "You have a peek budget of {} (of 6 games): 'PEEK <n>' reveals the "
            "true winner of game n.".format(self.peek_budget)
        )
        lines.append(
            "When ready: 'SUBMIT <w1> <w2> <w3> <w4> <w5> <w6>' with each wi the "
            "letter you believe won that numbered game, e.g. SUBMIT A A D B D C."
        )
        lines.append(
            "10 steps total. Reward = (correct games)/6, awarded only when you "
            "SUBMIT, which ends the episode."
        )
        return "\n".join(lines)

    def _tally(self):
        lines = ["Tally after peeks:"]
        for t in range(4):
            confirmed_wins = 0
            peeked_involving = 0
            for gi, (i, j) in enumerate(self.games):
                if i != t and j != t:
                    continue
                if self.peeked[gi]:
                    peeked_involving += 1
                    winner_idx = i if self.winners[gi] == 0 else j
                    if winner_idx == t:
                        confirmed_wins += 1
            remaining = 3 - peeked_involving
            still_needed = self.wins[t] - confirmed_wins
            lines.append(
                "  {}: confirmed {}/{} wins, {} unresolved game(s), needs {} more "
                "win(s) among those".format(
                    self.teams[t], confirmed_wins, self.wins[t], remaining, still_needed
                )
            )
        return "\n".join(lines)

    def step(self, action):
        self.step_count += 1
        text = (action or "").strip()
        upper = text.upper()
        parts = upper.split()
        reward = 0.0
        terminated = False
        obs = ""

        if len(parts) == 2 and parts[0] == "PEEK":
            n_str = parts[1]
            if not n_str.isdigit() or not (1 <= int(n_str) <= 6):
                obs = "Invalid PEEK: give a game number from 1 to 6. " + self._tally()
            else:
                gi = int(n_str) - 1
                if self.peeked[gi]:
                    i, j = self.games[gi]
                    winner_idx = i if self.winners[gi] == 0 else j
                    obs = "Game {} ({}) already peeked: winner was {}.\n{}".format(
                        gi + 1, self._game_label(gi), self.teams[winner_idx], self._tally()
                    )
                elif self.peeks_used >= self.peek_budget:
                    obs = "Peek budget exhausted ({} used). Deduce the rest or SUBMIT.\n{}".format(
                        self.peeks_used, self._tally()
                    )
                else:
                    self.peeked[gi] = True
                    self.peeks_used += 1
                    i, j = self.games[gi]
                    winner_idx = i if self.winners[gi] == 0 else j
                    obs = "Game {} ({}) winner: {}. Peeks used: {}/{}.\n{}".format(
                        gi + 1, self._game_label(gi), self.teams[winner_idx],
                        self.peeks_used, self.peek_budget, self._tally()
                    )
        elif len(parts) == 7 and parts[0] == "SUBMIT":
            guesses = parts[1:]
            if any(g not in self.teams for g in guesses):
                obs = "Invalid SUBMIT: each of the 6 entries must be A, B, C, or D.\n" + self._tally()
            else:
                correct = 0
                detail = []
                for gi, (i, j) in enumerate(self.games):
                    winner_idx = i if self.winners[gi] == 0 else j
                    true_letter = self.teams[winner_idx]
                    ok = guesses[gi] == true_letter
                    if ok:
                        correct += 1
                    detail.append("{}) {}: guessed {}, actual {} [{}]".format(
                        gi + 1, self._game_label(gi), guesses[gi], true_letter,
                        "correct" if ok else "wrong"
                    ))
                reward = correct / 6.0
                terminated = True
                self.done = True
                obs = "SUBMIT graded: {}/6 correct.\n{}".format(correct, "\n".join(detail))
        else:
            obs = (
                "Malformed action. Use 'PEEK <n>' (n=1..6) or 'SUBMIT <6 letters>'.\n"
                + self._tally()
            )

        truncated = (not terminated) and self.step_count >= 10
        return obs, reward, terminated, truncated, {}
