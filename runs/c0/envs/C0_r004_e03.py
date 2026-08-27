import random
import itertools
import re

class RoundRobinDeductionEnv:
    TEAMS = ['A', 'B', 'C', 'D']
    NAME_POOL = ["Falcons", "Wolves", "Bears", "Hawks", "Comets",
                 "Titans", "Ravens", "Lions", "Sharks", "Eagles"]
    MAX_STEPS = 10
    MAX_ASKS = 2

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.match_order = list(itertools.combinations(self.TEAMS, 2))
        names = self.rng.sample(self.NAME_POOL, 4)
        self.names = dict(zip(self.TEAMS, names))

        self.true_winner = []
        for pair in self.match_order:
            self.true_winner.append(self.rng.choice(pair))

        self.wins = {t: 0 for t in self.TEAMS}
        for w in self.true_winner:
            self.wins[w] += 1

        self.step_count = 0
        self.asks_used = 0
        self.credited = [False] * 6
        self.total_awarded = 0.0
        self.terminated = False

        obs = self._render_intro()
        return obs, {}

    def _render_intro(self):
        lines = []
        lines.append("ROUND-ROBIN DEDUCTION: Four teams each played every other "
                      "team once (no draws, every match has a winner). Determine "
                      "the winner of all 6 matches.")
        lines.append("Teams: " + ", ".join(f"{t}={self.names[t]}" for t in self.TEAMS))
        standings = ", ".join(f"{t}={self.wins[t]} win(s)" for t in self.TEAMS)
        lines.append(f"Final win totals (season standings): {standings}")
        match_list = ", ".join(f"{i+1}) {a} vs {b}" for i, (a, b) in enumerate(self.match_order))
        lines.append(f"Matches in fixed order: {match_list}")
        lines.append(f"You have {self.MAX_STEPS} total steps. Actions:")
        lines.append(f"  ASK <X> <Y>  - directly reveal the winner of the match between "
                      f"team X and team Y (only {self.MAX_ASKS} ASKs allowed for the whole game)")
        lines.append("  SUBMIT <r1> <r2> <r3> <r4> <r5> <r6>  - guess the winner letter "
                      "for each of the 6 matches, in the order listed above")
        lines.append("You may SUBMIT more than once. Each submission reports how many of "
                      "the 6 slots you got right this time, and how many distinct matches "
                      "you have permanently earned credit for (credit is never taken away). "
                      "It does NOT say which slots are right or wrong. "
                      "All 6 credited ends the episode with full reward. "
                      "Malformed actions cost a step and earn nothing.")
        return "\n".join(lines)

    def step(self, action):
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()
        upper = action.upper()

        if upper.startswith("ASK"):
            obs, reward = self._handle_ask(upper)
        elif upper.startswith("SUBMIT"):
            obs, reward = self._handle_submit(upper)
        else:
            obs = ("Unrecognized action. Use 'ASK <X> <Y>' or "
                   "'SUBMIT <r1> <r2> <r3> <r4> <r5> <r6>'.")
            reward = 0.0

        terminated = self.terminated
        truncated = (not terminated) and self.step_count >= self.MAX_STEPS
        if truncated:
            obs += (f"\nStep limit reached. Final tally: "
                     f"{sum(self.credited)}/6 matches credited.")

        return obs, reward, terminated, truncated, {}

    def _handle_ask(self, upper):
        m = re.match(r'^ASK\s+([ABCD])\s+([ABCD])$', upper)
        if not m:
            return ("Malformed ASK. Format: ASK <X> <Y> using two different "
                    "letters from A, B, C, D."), 0.0
        x, y = m.group(1), m.group(2)
        if x == y:
            return "ASK requires two distinct teams.", 0.0
        if self.asks_used >= self.MAX_ASKS:
            return (f"No ASKs remaining ({self.MAX_ASKS} already used). "
                    "Use SUBMIT instead."), 0.0
        pair = tuple(sorted((x, y)))
        if pair not in self.match_order:
            return "That match does not exist.", 0.0
        idx = self.match_order.index(pair)
        winner = self.true_winner[idx]
        self.asks_used += 1
        remaining = self.MAX_ASKS - self.asks_used
        return (f"ASK result: in {pair[0]} vs {pair[1]}, the winner was {winner} "
                f"({self.names[winner]}). ASKs remaining: {remaining}."), 0.0

    def _handle_submit(self, upper):
        rest = upper[len("SUBMIT"):].split()
        if len(rest) != 6:
            return (f"Malformed SUBMIT: expected exactly 6 results, got {len(rest)}. "
                    "Format: SUBMIT <r1> <r2> <r3> <r4> <r5> <r6>."), 0.0

        for i, (tok, pair) in enumerate(zip(rest, self.match_order)):
            if tok not in pair:
                return (f"Malformed SUBMIT: slot {i+1} ({pair[0]} vs {pair[1]}) "
                         f"must be '{pair[0]}' or '{pair[1]}', got '{tok}'."), 0.0

        raw_correct = sum(1 for tok, truth in zip(rest, self.true_winner) if tok == truth)

        newly = 0
        for i, (tok, truth) in enumerate(zip(rest, self.true_winner)):
            if tok == truth and not self.credited[i]:
                self.credited[i] = True
                newly += 1

        total_credited = sum(self.credited)

        if newly > 0:
            if total_credited == 6:
                reward = 1.0 - self.total_awarded
            else:
                reward = newly / 6.0
            self.total_awarded += reward
        else:
            reward = 0.0

        if total_credited == 6:
            self.terminated = True
            obs = (f"Submission matched {raw_correct}/6 this attempt. "
                    "All 6 matches correctly deduced! Episode complete.")
        else:
            obs = (f"Submission matched {raw_correct}/6 slots this attempt "
                    f"(order not disclosed). Permanently credited so far: "
                    f"{total_credited}/6. ASKs remaining: "
                    f"{self.MAX_ASKS - self.asks_used}. "
                    f"Steps remaining: {self.MAX_STEPS - self.step_count}.")

        return obs, reward
