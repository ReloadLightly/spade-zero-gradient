import random


class RoundRobinDeductionEnv:
    def __init__(self):
        self.teams = ['A', 'B', 'C', 'D']
        self.pairs = [('A', 'B'), ('A', 'C'), ('A', 'D'),
                       ('B', 'C'), ('B', 'D'), ('C', 'D')]

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.results = {}
        for (t1, t2) in self.pairs:
            if self.rng.random() < 0.3:
                self.results[(t1, t2)] = 'DRAW'
            else:
                self.results[(t1, t2)] = t1 if self.rng.random() < 0.5 else t2

        self.points = {t: 0 for t in self.teams}
        for (t1, t2), res in self.results.items():
            if res == 'DRAW':
                self.points[t1] += 1
                self.points[t2] += 1
            else:
                self.points[res] += 3

        self.queried = set()
        self.query_budget = 4
        self.queries_used = 0
        self.step_count = 0
        self.max_steps = 10
        self.done = False

        standings = ", ".join(f"{t}:{self.points[t]}" for t in self.teams)
        pair_order = ", ".join(f"{t1}{t2}" for t1, t2 in self.pairs)
        obs = (
            "Deduce the hidden results of a 4-team round-robin tournament "
            "(teams A, B, C, D; every pair played exactly once; win=3pts, "
            "draw=1pt each, loss=0pts).\n"
            f"Final standings (total points): {standings}.\n"
            f"The six matches, in fixed order, are: {pair_order}.\n"
            "Goal: determine the true outcome of every match.\n"
            f"You have {self.query_budget} QUERY actions to reveal a specific "
            "match's real result; the rest must be reasoned out from the "
            "standings above.\n"
            "Actions:\n"
            "  QUERY <team1> <team2>  e.g. 'QUERY A C' (spends one query)\n"
            "  SUBMIT <r1> <r2> <r3> <r4> <r5> <r6>  one result per match in "
            "the fixed order above; for match XY write 'X' if X won, 'Y' if "
            "Y won, or 'DRAW' if it was a draw\n"
            f"You have {self.max_steps} steps total. SUBMIT ends the episode "
            "and scores 1/6 credit per correct match."
        )
        return obs, {}

    def _describe(self, pair, res):
        t1, t2 = pair
        if res == 'DRAW':
            return f"{t1} vs {t2}: DRAW (1-1)."
        loser = t2 if res == t1 else t1
        return f"{t1} vs {t2}: {res} beat {loser} (3-0)."

    def step(self, action):
        if self.done:
            return "Episode already complete.", 0.0, True, False, {}

        self.step_count += 1
        trunc = self.step_count >= self.max_steps
        parts = action.strip().split()

        if not parts:
            return ("Malformed action. Use 'QUERY <team1> <team2>' or "
                    "'SUBMIT <r1>..<r6>'.", 0.0, False, trunc, {})

        cmd = parts[0].upper()

        if cmd == 'QUERY' and len(parts) == 3:
            t1, t2 = parts[1].upper(), parts[2].upper()
            if t1 not in self.teams or t2 not in self.teams or t1 == t2:
                return ("Invalid teams. Use two distinct letters among "
                        "A, B, C, D.", 0.0, False, trunc, {})
            pair = (t1, t2) if (t1, t2) in self.results else (t2, t1)
            if pair not in self.results:
                return "Invalid teams.", 0.0, False, trunc, {}
            if pair in self.queried:
                res = self.results[pair]
                return (f"(already known) {self._describe(pair, res)}",
                        0.0, False, trunc, {})
            if self.queries_used >= self.query_budget:
                return (f"No queries remaining ({self.query_budget} used). "
                         "Deduce the rest from the standings and SUBMIT.",
                        0.0, False, trunc, {})
            self.queried.add(pair)
            self.queries_used += 1
            res = self.results[pair]
            remaining = self.query_budget - self.queries_used
            obs = f"{self._describe(pair, res)} Queries remaining: {remaining}."
            return obs, 0.0, False, trunc, {}

        if cmd == 'SUBMIT' and len(parts) == 7:
            guesses = [p.upper() for p in parts[1:]]
            correct = 0
            for pair, guess in zip(self.pairs, guesses):
                if guess == self.results[pair]:
                    correct += 1
            reward = correct / 6.0
            self.done = True
            obs = f"Submission scored: {correct}/6 correct. Episode complete."
            return obs, reward, True, False, {"correct": correct}

        return ("Malformed action. Use 'QUERY <team1> <team2>' or "
                "'SUBMIT <r1> <r2> <r3> <r4> <r5> <r6>' (one of the two "
                "team letters or DRAW, in order AB AC AD BC BD CD).",
                0.0, False, trunc, {})
