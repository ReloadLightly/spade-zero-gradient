import random
import re


class BracketSeedEnv:
    TEAMS = ['A', 'B', 'C', 'D', 'E', 'F']
    MAX_STEPS = 10
    SCOUT_BUDGET = 4

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        strengths = self.rng.sample(range(10, 91), 6)
        self.strength = dict(zip(self.TEAMS, strengths))
        ranked = sorted(self.TEAMS, key=lambda t: -self.strength[t])
        self.tier = {}
        for t in ranked[0:2]:
            self.tier[t] = 'Contender'
        for t in ranked[2:4]:
            self.tier[t] = 'Challenger'
        for t in ranked[4:6]:
            self.tier[t] = 'Underdog'

        self.matchings = self._all_matchings(self.TEAMS)
        sums = [self._sum_upset(m) for m in self.matchings]
        self.best_sum = min(sums)
        self.worst_sum = max(sums)

        self.scouted = set()
        self.scouts_used = 0
        self.step_count = 0
        self.done = False

        tier_lines = ', '.join(f'{t} ({self.tier[t]})' for t in self.TEAMS)
        obs = (
            "You are seeding a 6-team single-elimination bracket. Pair the six "
            "teams into three first-round matchups to minimize the total expected "
            "number of upsets (a weaker team beating a stronger one). Upset risk "
            "in a matchup grows as the two teams' strengths get closer together, "
            "and shrinks as the gap widens.\n"
            f"Teams and coarse tiers: {tier_lines}.\n"
            f"Actions: 'SCOUT <letter>' reveals a team's exact hidden strength "
            f"(you get {self.SCOUT_BUDGET} scouts total). 'PAIR X-Y,Z-W,Q-R' "
            "submits your final matchups using all six letters exactly once and "
            "ends the episode immediately (irreversible).\n"
            f"You have {self.MAX_STEPS} total actions available."
        )
        return obs, {}

    def step(self, action):
        self.step_count += 1
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        act = (action or '').strip().upper()
        scout_match = re.match(r'^SCOUT\s+([A-F])$', act)
        pair_match = re.match(
            r'^PAIR\s+([A-F])-([A-F])\s*,\s*([A-F])-([A-F])\s*,\s*([A-F])-([A-F])$',
            act,
        )

        reward = 0.0
        terminated = False
        info = {}

        if scout_match:
            team = scout_match.group(1)
            if self.scouts_used >= self.SCOUT_BUDGET:
                obs = f"No scouts remaining ({self.SCOUT_BUDGET}/{self.SCOUT_BUDGET} used)."
            else:
                self.scouts_used += 1
                self.scouted.add(team)
                remaining = self.SCOUT_BUDGET - self.scouts_used
                obs = (
                    f"Scouted {team}: true strength = {self.strength[team]}. "
                    f"Scouts remaining: {remaining}/{self.SCOUT_BUDGET}."
                )
        elif pair_match:
            letters = list(pair_match.groups())
            if len(set(letters)) != 6:
                obs = "Invalid PAIR: each of the six teams must appear exactly once."
            else:
                pairs = [
                    (letters[0], letters[1]),
                    (letters[2], letters[3]),
                    (letters[4], letters[5]),
                ]
                submitted_sum = self._sum_upset(pairs)
                spread = self.worst_sum - self.best_sum
                quality = 1.0 if spread < 1e-9 else (self.worst_sum - submitted_sum) / spread
                quality = max(0.0, min(1.0, quality))
                bonus = 0.1 if abs(submitted_sum - self.best_sum) < 1e-9 else 0.0
                reward = max(0.0, min(1.0, 0.9 * quality + bonus))
                terminated = True
                self.done = True
                info = {
                    'true_strengths': dict(self.strength),
                    'submitted_sum': submitted_sum,
                    'best_sum': self.best_sum,
                    'worst_sum': self.worst_sum,
                }
                obs = (
                    f"Bracket submitted: {letters[0]}-{letters[1]}, "
                    f"{letters[2]}-{letters[3]}, {letters[4]}-{letters[5]}. "
                    f"Episode ended. Reward: {reward:.3f}."
                )
        else:
            obs = (
                "Unrecognized action. Use 'SCOUT <letter>' or "
                "'PAIR X-Y,Z-W,Q-R' with all six team letters."
            )

        truncated = False
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True

        return obs, reward, terminated, truncated, info

    def _sum_upset(self, pairs):
        total = 0.0
        for a, b in pairs:
            sa, sb = self.strength[a], self.strength[b]
            weaker, stronger = (sa, sb) if sa < sb else (sb, sa)
            total += weaker / (weaker + stronger)
        return total

    @staticmethod
    def _all_matchings(elems):
        if not elems:
            return [[]]
        result = []
        first = elems[0]
        rest = elems[1:]
        for i in range(len(rest)):
            partner = rest[i]
            remaining = rest[:i] + rest[i + 1:]
            for sub in BracketSeedEnv._all_matchings(remaining):
                result.append([(first, partner)] + sub)
        return result
