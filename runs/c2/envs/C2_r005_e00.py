import random
import itertools


class RoundRobinCycleEnv:
    TEAMS = ["Falcons", "Wolves", "Bears", "Otters"]

    MAX_STEPS = 10
    MAX_WINS_QUERIES = 3
    MAX_MATCH_QUERIES = 2

    def __init__(self):
        self.rng = None
        self.results = {}
        self.win_counts = {}
        self.idx = {t: i for i, t in enumerate(self.TEAMS)}
        self.pairs = list(itertools.combinations(self.TEAMS, 2))
        self.step_count = 0
        self.wins_used = 0
        self.match_used = 0
        self.terminated = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.wins_used = 0
        self.match_used = 0
        self.terminated = False
        self.results = {}
        for (t1, t2) in self.pairs:
            winner = t1 if self.rng.random() < 0.5 else t2
            self.results[(t1, t2)] = winner
        self.win_counts = {t: 0 for t in self.TEAMS}
        for (t1, t2), w in self.results.items():
            self.win_counts[w] += 1
        return self._intro(), {}

    def _intro(self):
        return (
            "Four teams (Falcons, Wolves, Bears, Otters) each played every other team once "
            "(6 games, no draws). Deduce the winner of every game and submit the full result set.\n"
            "Up to 10 steps total. Actions:\n"
            "  WINS <team>              - reveals that team's total win count (0-3). Max 3 uses.\n"
            "  MATCH <teamA> <teamB>    - reveals the winner of that single game. Max 2 uses.\n"
            "  SUBMIT <p1>,<p2>,...,<p6> - one entry per game as Winner>Loser, e.g.\n"
            "    SUBMIT Falcons>Wolves,Falcons>Bears,Falcons>Otters,Wolves>Bears,Wolves>Otters,Bears>Otters\n"
            "    Must cover all 6 pairs exactly once. Ends the episode.\n"
            "Reward = (games you got right) / 6, given only on SUBMIT."
        )

    def _canon(self, t1, t2):
        if self.idx[t1] < self.idx[t2]:
            return (t1, t2)
        return (t2, t1)

    def _team_match(self, name):
        name = name.strip()
        for t in self.TEAMS:
            if t.lower() == name.lower():
                return t
        return None

    def step(self, action):
        if self.terminated:
            return "Episode already over.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()
        parts = action.split(None, 1)
        verb = parts[0].upper() if parts else ""

        if verb == "WINS" and len(parts) == 2:
            team = self._team_match(parts[1])
            if team is None:
                obs = "Unknown team name. Valid teams: " + ", ".join(self.TEAMS) + "."
                return self._footer(obs), 0.0, False, self._check_truncate(), {}
            if self.wins_used >= self.MAX_WINS_QUERIES:
                obs = "No WINS queries remaining."
                return self._footer(obs), 0.0, False, self._check_truncate(), {}
            self.wins_used += 1
            obs = f"{team} won {self.win_counts[team]} of its 3 games."
            return self._footer(obs), 0.0, False, self._check_truncate(), {}

        if verb == "MATCH" and len(parts) == 2:
            args = parts[1].split()
            if len(args) != 2:
                obs = "Usage: MATCH <teamA> <teamB>"
                return self._footer(obs), 0.0, False, self._check_truncate(), {}
            t1, t2 = self._team_match(args[0]), self._team_match(args[1])
            if t1 is None or t2 is None or t1 == t2:
                obs = "Give two distinct valid team names."
                return self._footer(obs), 0.0, False, self._check_truncate(), {}
            if self.match_used >= self.MAX_MATCH_QUERIES:
                obs = "No MATCH queries remaining."
                return self._footer(obs), 0.0, False, self._check_truncate(), {}
            self.match_used += 1
            key = self._canon(t1, t2)
            winner = self.results[key]
            loser = key[0] if winner == key[1] else key[1]
            obs = f"{winner} beat {loser}."
            return self._footer(obs), 0.0, False, self._check_truncate(), {}

        if verb == "SUBMIT" and len(parts) == 2:
            return self._handle_submit(parts[1])

        obs = "Malformed action. Use WINS <team>, MATCH <teamA> <teamB>, or SUBMIT <6 pairs>."
        return self._footer(obs), 0.0, False, self._check_truncate(), {}

    def _handle_submit(self, payload):
        entries = [e.strip() for e in payload.split(",") if e.strip()]
        parsed = {}
        malformed = len(entries) != 6
        if not malformed:
            for e in entries:
                if ">" not in e:
                    malformed = True
                    break
                a, b = e.split(">", 1)
                ta, tb = self._team_match(a), self._team_match(b)
                if ta is None or tb is None or ta == tb:
                    malformed = True
                    break
                key = self._canon(ta, tb)
                if key in parsed:
                    malformed = True
                    break
                parsed[key] = ta
            if not malformed and set(parsed.keys()) != set(self.pairs):
                malformed = True

        if malformed:
            obs = "SUBMIT must list each of the 6 pairs exactly once as Winner>Loser."
            return self._footer(obs), 0.0, False, self._check_truncate(), {}

        correct = sum(1 for key, w in parsed.items() if self.results[key] == w)
        reward = correct / 6.0
        self.terminated = True
        obs = f"Submitted. {correct}/6 games correct. Episode over."
        return obs, reward, True, False, {"correct": correct}

    def _footer(self, obs):
        remaining = self.MAX_STEPS - self.step_count
        return (
            obs
            + f" [WINS left: {self.MAX_WINS_QUERIES - self.wins_used}, "
            f"MATCH left: {self.MAX_MATCH_QUERIES - self.match_used}, "
            f"steps left: {remaining}]"
        )

    def _check_truncate(self):
        return self.step_count >= self.MAX_STEPS
