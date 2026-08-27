import random
import re


class BracketSeedingEnv:
    """Seed a 4-team single-elimination bracket to minimize expected upsets."""

    TEAM_POOL = [
        "Falcons", "Wolves", "Hawks", "Bears", "Otters",
        "Lynxes", "Ravens", "Cobras", "Titans", "Sharks",
        "Panthers", "Comets",
    ]

    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.teams = []
        self.team_lookup = {}
        self.strength = {}
        self.step_count = 0
        self.scouted_pairs = set()
        self.scout_milestones_paid = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.teams = self.rng.sample(self.TEAM_POOL, 4)
        self.team_lookup = {t.lower(): t for t in self.teams}
        values = self.rng.sample(range(1, 41), 4)
        self.strength = dict(zip(self.teams, values))
        self.step_count = 0
        self.scouted_pairs = set()
        self.scout_milestones_paid = 0
        self.done = False

        obs = (
            "BRACKET SEEDING\n"
            f"Four teams are entered in a single-elimination bracket: {', '.join(self.teams)}.\n"
            "Each team has a hidden power rating. In any match the weaker team can still score "
            "an upset; the chance of an upset shrinks the larger the power gap between the two "
            "teams.\n"
            "Your job: choose which two teams meet in the first round's match A and which two "
            "meet in match B (winners of A and B meet in the final), so as to minimize the total "
            "expected number of upsets across all three matches.\n"
            "Actions (exactly one per turn):\n"
            "  SCOUT <team> <team>   -- get a qualitative read on the power gap between two teams\n"
            "  SEED <team> <team> <team> <team>  -- lock in the bracket: team1 vs team2 is match "
            "A, team3 vs team4 is match B; this ends the episode\n"
            f"You have {self.MAX_STEPS} actions total. All four team names must appear exactly "
            "once in a SEED action."
        )
        info = {"teams": list(self.teams)}
        return obs, info

    def _resolve(self, token):
        return self.team_lookup.get((token or "").strip().lower())

    def _gap_descriptor(self, diff):
        if diff < 5:
            return "even"
        if diff < 12:
            return "edge"
        if diff < 22:
            return "clear"
        return "dominant"

    def _p_upset(self, diff):
        return max(0.05, 0.5 - 0.025 * diff)

    def _expected_upsets(self, a, b, c, d):
        sa, sb, sc, sd = self.strength[a], self.strength[b], self.strength[c], self.strength[d]
        pa = self._p_upset(abs(sa - sb))
        pb = self._p_upset(abs(sc - sd))
        fav_a, dog_a = (a, b) if sa > sb else (b, a)
        fav_b, dog_b = (c, d) if sc > sd else (d, c)
        w_a = {fav_a: 1 - pa, dog_a: pa}
        w_b = {fav_b: 1 - pb, dog_b: pb}
        expected_final = 0.0
        for x, px in w_a.items():
            for y, py in w_b.items():
                diff_final = abs(self.strength[x] - self.strength[y])
                expected_final += px * py * self._p_upset(diff_final)
        return pa + pb + expected_final

    def _best_worst(self):
        a, b, c, d = self.teams
        partitions = [(a, b, c, d), (a, c, b, d), (a, d, b, c)]
        totals = [self._expected_upsets(*p) for p in partitions]
        return min(totals), max(totals)

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        truncated = self.step_count >= self.MAX_STEPS
        if truncated:
            self.done = True
        text = (action or "").strip()

        m = re.match(r"^SCOUT\s+(\S+)\s+(\S+)$", text, re.IGNORECASE)
        if m:
            t1, t2 = self._resolve(m.group(1)), self._resolve(m.group(2))
            if not t1 or not t2 or t1 == t2:
                obs = f"Invalid SCOUT: name two distinct teams from {', '.join(self.teams)}."
                return obs, 0.0, False, truncated, {}
            diff = abs(self.strength[t1] - self.strength[t2])
            fav = t1 if self.strength[t1] > self.strength[t2] else t2
            desc = self._gap_descriptor(diff)
            pair_key = frozenset((t1, t2))
            reward = 0.0
            if pair_key not in self.scouted_pairs and self.scout_milestones_paid < 2:
                self.scout_milestones_paid += 1
                reward = 0.1
            self.scouted_pairs.add(pair_key)
            obs = f"Scouting {t1} vs {t2}: gap is '{desc}', {fav} looks favored."
            if truncated:
                obs += " Step limit reached without a SEED submission; episode ends."
            return obs, reward, False, truncated, {}

        m = re.match(r"^SEED\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", text, re.IGNORECASE)
        if m:
            names = [self._resolve(m.group(i)) for i in range(1, 5)]
            if None in names or sorted(names) != sorted(self.teams):
                obs = f"Invalid SEED: must list each of {', '.join(self.teams)} exactly once."
                return obs, 0.0, False, truncated, {}
            self.done = True
            submitted = self._expected_upsets(*names)
            best, worst = self._best_worst()
            if worst - best < 1e-9:
                seed_reward = 0.8
            else:
                frac = (submitted - best) / (worst - best)
                seed_reward = max(0.0, 1.0 - frac) * 0.8
            obs = (
                f"Bracket locked: {names[0]} vs {names[1]}, {names[2]} vs {names[3]}.\n"
                f"Expected upsets for this seeding: {submitted:.3f} "
                f"(best possible: {best:.3f}, worst possible: {worst:.3f})."
            )
            return obs, seed_reward, True, False, {
                "submitted": submitted, "best": best, "worst": worst,
            }

        obs = (
            "Unrecognized action. Use 'SCOUT <team> <team>' or "
            "'SEED <team> <team> <team> <team>'."
        )
        return obs, 0.0, False, truncated, {}
