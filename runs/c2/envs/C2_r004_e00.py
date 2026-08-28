import itertools
import random


class RoundRobinStandingsEnv:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.teams = ["Comets", "Badgers", "Herons", "Wolves", "Otters"]
        self.true_order = self.rng.sample(self.teams, 5)
        self.rank_of = {t: i for i, t in enumerate(self.true_order)}

        all_pairs = list(itertools.combinations(self.teams, 2))
        a, b = self.rng.choice(all_pairs)
        winner = a if self.rank_of[a] < self.rank_of[b] else b
        loser = b if winner == a else a
        self.known = {frozenset((a, b)): winner}

        self.queries_used = 0
        self.max_queries = 5
        self.steps = 0
        self.max_steps = 10
        self.m1 = self.m2 = self.m3 = self.m4 = False
        self.terminated_flag = False

        obs = (
            "ROUND-ROBIN STANDINGS DEDUCTION\n"
            f"Five teams played a full round robin (every pair met once, no draws): "
            f"{', '.join(self.teams)}.\n"
            "Each team has a fixed hidden strength rank; the stronger team always won "
            "its game.\n"
            f"Known result: {winner} defeated {loser}.\n"
            "Goal: determine the full final standings, strongest (1st) to weakest (5th).\n"
            "Actions (exact format):\n"
            "  QUERY teamA teamB   -- learn who won that game (uses one of your "
            f"{self.max_queries} queries)\n"
            "  SUBMIT t1 t2 t3 t4 t5   -- commit final standings, strongest first "
            "(ends the episode)\n"
            f"You have at most {self.max_queries} QUERY actions within a {self.max_steps}-"
            "step limit. Reward comes from provably narrowing the possibilities as you "
            "go, plus how many positions your final SUBMIT gets right."
        )
        return obs, {}

    def _match_team(self, raw):
        for t in self.teams:
            if raw.lower() == t.lower():
                return t
        return None

    def _consistent_permutations(self):
        result = []
        for perm in itertools.permutations(self.teams):
            pos = {t: i for i, t in enumerate(perm)}
            ok = True
            for pair, winner in self.known.items():
                a, b = tuple(pair)
                loser = b if winner == a else a
                if pos[winner] > pos[loser]:
                    ok = False
                    break
            if ok:
                result.append(perm)
        return result

    def _check_milestones(self):
        perms = self._consistent_permutations()
        gained = 0.0
        if not self.m1 and len(perms) <= 24:
            self.m1 = True
            gained += 0.15
        if not self.m2 and len(perms) <= 6:
            self.m2 = True
            gained += 0.15
        pinned = 0
        for t in self.teams:
            positions = {perm.index(t) for perm in perms}
            if len(positions) == 1:
                pinned += 1
        if not self.m3 and pinned >= 2:
            self.m3 = True
            gained += 0.15
        if not self.m4 and pinned >= 4:
            self.m4 = True
            gained += 0.15
        return gained

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        tokens = (action or "").strip().split()
        cmd = tokens[0].upper() if tokens else ""

        if cmd == "QUERY" and len(tokens) == 3:
            a = self._match_team(tokens[1])
            b = self._match_team(tokens[2])
            if a is None or b is None or a == b:
                obs = "Malformed QUERY: give two distinct valid team names."
            elif self.queries_used >= self.max_queries:
                obs = f"Query budget exhausted ({self.max_queries} used). Try SUBMIT."
            elif frozenset((a, b)) in self.known:
                obs = f"Already known: {self.known[frozenset((a, b))]} won that game."
            else:
                winner = a if self.rank_of[a] < self.rank_of[b] else b
                loser = b if winner == a else a
                self.known[frozenset((a, b))] = winner
                self.queries_used += 1
                reward = self._check_milestones()
                obs = (
                    f"{winner} defeated {loser}. "
                    f"({self.max_queries - self.queries_used} queries left)"
                )
        elif cmd == "SUBMIT" and len(tokens) == 6:
            subm = [self._match_team(t) for t in tokens[1:]]
            if None in subm or len(set(subm)) != 5:
                obs = "Malformed SUBMIT: list all 5 team names exactly once."
            else:
                correct = sum(1 for i in range(5) if subm[i] == self.true_order[i])
                reward = 0.08 * correct
                obs = f"Final standings recorded. {correct}/5 positions correct."
                terminated = True
                self.terminated_flag = True
        else:
            obs = (
                "Unrecognized action. Use 'QUERY teamA teamB' or "
                "'SUBMIT t1 t2 t3 t4 t5'."
            )

        if not terminated and self.steps >= self.max_steps:
            truncated = True
            obs += " Step limit reached."

        info = {
            "queries_used": self.queries_used,
            "queries_remaining": self.max_queries - self.queries_used,
        }
        return obs, reward, terminated, truncated, info
