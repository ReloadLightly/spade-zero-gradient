import random


def _all_pairings(items):
    if not items:
        yield []
        return
    a = items[0]
    rest = items[1:]
    for i in range(len(rest)):
        b = rest[i]
        remaining = rest[:i] + rest[i + 1:]
        for sub in _all_pairings(remaining):
            yield [(a, b)] + sub


def _pair_risk(a_strength, b_strength):
    return min(a_strength, b_strength) / (a_strength + b_strength)


def _total_risk(pairing, strengths):
    return sum(_pair_risk(strengths[a], strengths[b]) for a, b in pairing)


def _classify_ratio(ratio):
    if ratio < 1.15:
        return "a virtual coin flip"
    if ratio < 1.4:
        return "a moderate favorite"
    if ratio < 1.8:
        return "a strong favorite"
    return "an overwhelming favorite"


class BracketSeedEnv:
    TEAM_IDS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    TEAM_NAMES = ["Ospreys", "Kestrels", "Foxes", "Wolves",
                  "Bears", "Lynxes", "Tigers", "Hawks"]
    MAX_STEPS = 10
    MAX_INFO_SCOUTS = 3

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        values = self.rng.sample(range(30, 100), 8)
        self.strengths = dict(zip(self.TEAM_IDS, values))
        self.graph = {tid: [] for tid in self.TEAM_IDS}
        self.scouted_pairs = set()
        self.info_scouts_used = 0
        self.step_count = 0
        self.done = False

        roster = "\n".join(
            f"  {tid} ({name})" for tid, name in zip(self.TEAM_IDS, self.TEAM_NAMES)
        )
        obs = (
            "Bracket Seeding Challenge. 8 teams, hidden strength values.\n"
            "Goal: submit 4 first-round pairs (a full partition of all 8 teams) "
            "that minimizes the total expected-upset risk across the round.\n"
            f"Teams:\n{roster}\n\n"
            "Actions (exactly one per turn):\n"
            "  SCOUT <team_id> <team_id>  - compare two teams; learn who is "
            "favored and a rough margin label. Costs a step.\n"
            "  SEED <a>-<b> <c>-<d> <e>-<f> <g>-<h>  - submit your final pairing "
            "using all 8 team ids exactly once. Ends the episode.\n"
            f"You have {self.MAX_STEPS} steps total (scouting + the final SEED). "
            "Exact strength numbers are never revealed."
        )
        return obs, {}

    def _reachable(self, src, dst):
        visited = set()
        stack = [src]
        while stack:
            node = stack.pop()
            if node == dst:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.graph.get(node, []))
        return False

    def _finish(self, obs, reward, terminated, truncated):
        self.done = True
        return obs, reward, terminated, truncated, {"step": self.step_count}

    def step(self, action):
        if self.done:
            return self._finish("Episode already finished.", 0.0, True, False)

        self.step_count += 1
        budget_left = self.MAX_STEPS - self.step_count
        tokens = (action or "").strip().split()

        if not tokens:
            obs = "Empty action. Use SCOUT <id> <id> or SEED <a>-<b> ... (4 pairs)."
            return self._advance(obs, 0.0, budget_left)

        cmd = tokens[0].upper()

        if cmd == "SCOUT":
            if len(tokens) != 3:
                obs = "Malformed SCOUT. Format: SCOUT <team_id> <team_id>."
                return self._advance(obs, 0.0, budget_left)
            a, b = tokens[1].upper(), tokens[2].upper()
            if a not in self.strengths or b not in self.strengths or a == b:
                obs = f"Unknown or duplicate team id(s) in '{a} {b}'. Valid ids: {', '.join(self.TEAM_IDS)}."
                return self._advance(obs, 0.0, budget_left)

            pair_key = frozenset((a, b))
            winner, loser = (a, b) if self.strengths[a] > self.strengths[b] else (b, a)
            ratio = self.strengths[winner] / self.strengths[loser]
            label = _classify_ratio(ratio)

            reward = 0.0
            if pair_key in self.scouted_pairs:
                obs = f"{winner} beats {loser} ({label}). No new insight (already scouted)."
            else:
                self.scouted_pairs.add(pair_key)
                already_known = self._reachable(winner, loser)
                if not already_known and self.info_scouts_used < self.MAX_INFO_SCOUTS:
                    reward = 0.1
                    self.info_scouts_used += 1
                    obs = f"{winner} beats {loser} ({label}). New information gained."
                elif not already_known:
                    obs = f"{winner} beats {loser} ({label}). New information (progress budget already used)."
                else:
                    obs = f"{winner} beats {loser} ({label}). Confirms what you could already infer."
                self.graph[winner].append(loser)
            return self._advance(obs, reward, budget_left)

        if cmd == "SEED":
            pair_tokens = tokens[1:]
            if len(pair_tokens) != 4:
                obs = "Malformed SEED. Give exactly 4 pairs like SEED T1-T2 T3-T4 T5-T6 T7-T8."
                return self._advance(obs, 0.0, budget_left)
            pairs = []
            seen = set()
            valid = True
            for pt in pair_tokens:
                parts = pt.upper().split("-")
                if len(parts) != 2:
                    valid = False
                    break
                x, y = parts
                if x not in self.strengths or y not in self.strengths or x == y:
                    valid = False
                    break
                if x in seen or y in seen:
                    valid = False
                    break
                seen.add(x)
                seen.add(y)
                pairs.append((x, y))
            if not valid or len(seen) != 8:
                obs = ("Invalid SEED: must use all 8 team ids exactly once across 4 "
                       "disjoint pairs, e.g. SEED T1-T2 T3-T4 T5-T6 T7-T8.")
                return self._advance(obs, 0.0, budget_left)

            submitted_risk = _total_risk(pairs, self.strengths)
            all_pairings = list(_all_pairings(self.TEAM_IDS))
            risks = [_total_risk(p, self.strengths) for p in all_pairings]
            best, worst = min(risks), max(risks)

            if worst > best:
                quality = 0.7 * (worst - submitted_risk) / (worst - best)
            else:
                quality = 0.7
            quality = max(0.0, min(0.7, quality))

            detail = ", ".join(
                f"{x}-{y}: {_pair_risk(self.strengths[x], self.strengths[y]):.2f} upset risk"
                for x, y in pairs
            )
            obs = (
                f"Final seeding accepted. {detail}. "
                f"Total risk {submitted_risk:.2f} (best possible {best:.2f}, worst possible {worst:.2f})."
            )
            return self._finish(obs, quality, True, False)

        obs = "Unrecognized command. Use SCOUT <id> <id> or SEED <a>-<b> <c>-<d> <e>-<f> <g>-<h>."
        return self._advance(obs, 0.0, budget_left)

    def _advance(self, obs, reward, budget_left):
        if budget_left <= 0:
            obs = obs + " Step budget exhausted without a valid SEED submission."
            return self._finish(obs, reward, False, True)
        return obs, reward, False, False, {"step": self.step_count, "steps_left": budget_left}
