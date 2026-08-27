import random


def _rot_right(s, k):
    k %= len(s)
    return s if k == 0 else s[-k:] + s[:-k]


def _rot_left(s, k):
    k %= len(s)
    return s if k == 0 else s[k:] + s[:k]


def _invert(s):
    return ''.join('.' if c == 'X' else 'X' for c in s)


def _mirror(s):
    return s[::-1]


RULES = {
    'ROTR1': lambda b: _rot_right(b, 1),
    'ROTR2': lambda b: _rot_right(b, 2),
    'ROTL1': lambda b: _rot_left(b, 1),
    'INVERT': _invert,
    'MIRROR': _mirror,
}

RULE_DOC = (
    " ROTR1  - shift every step one position to the right (wrap around)\n"
    " ROTR2  - shift every step two positions to the right (wrap around)\n"
    " ROTL1  - shift every step one position to the left (wrap around)\n"
    " INVERT - flip every hit to a rest and every rest to a hit\n"
    " MIRROR - reverse the order of the 8 steps"
)

L = 8
MASK_SIZE = 3
MAX_STEPS = 10


class DrumPatternContinuationEnv:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.steps = 0
        self.bar3_solved = False
        self.bar4_solved = False
        self.rule_solved = False

        hits = self.rng.randint(3, 5)
        positions = set(self.rng.sample(range(L), hits))
        self.bar1 = ''.join('X' if i in positions else '.' for i in range(L))

        self.rule_name = self.rng.choice(list(RULES.keys()))
        rule_fn = RULES[self.rule_name]
        self.bar2 = rule_fn(self.bar1)
        self.bar3 = rule_fn(self.bar2)
        self.bar4 = rule_fn(self.bar3)

        best_mask = None
        for _ in range(20):
            candidate_mask = set(self.rng.sample(range(L), MASK_SIZE))
            revealed = [i for i in range(L) if i not in candidate_mask]
            consistent_count = 0
            for name, fn in RULES.items():
                pred = fn(self.bar1)
                if all(pred[i] == self.bar2[i] for i in revealed):
                    consistent_count += 1
            if best_mask is None:
                best_mask = candidate_mask
            if consistent_count >= 2:
                best_mask = candidate_mask
                break
        self.masked = set(best_mask)

        obs = self._render_intro()
        return obs, {'true_rule': self.rule_name, 'steps': self.steps}

    def _bar2_display(self):
        return ''.join('?' if i in self.masked else self.bar2[i] for i in range(L))

    def _render_intro(self):
        return (
            "You are analyzing a drum rhythm notated in 8 sixteenth-note steps per bar "
            "('X'=hit, '.'=rest; positions 1-8 left to right).\n\n"
            f"Bar 1 (given): {self.bar1}\n"
            f"Bar 2 (partially given): {self._bar2_display()}   ('?' = hidden step)\n\n"
            "A single hidden TRANSFORMATION RULE, drawn from the list below, was applied to "
            "Bar 1 to produce Bar 2, applied again to Bar 2 to produce Bar 3, and again to "
            "Bar 3 to produce Bar 4. Your job is to identify the rule and predict Bars 3 and 4 exactly.\n\n"
            "Candidate rules:\n" + RULE_DOC + "\n\n"
            f"You have up to {MAX_STEPS} actions total. Send exactly one action per turn:\n"
            " TEST <RULE_NAME>          - check how many currently visible Bar 2 steps a candidate rule matches (no reward, information only)\n"
            " REVEAL <position 1-8>     - reveal one currently hidden ('?') step of Bar 2\n"
            " SUBMIT_RULE <RULE_NAME>   - claim the true rule (reward 0.3 the first time you are correct)\n"
            " GUESS <bar3> <bar4>       - state your full 8-character prediction for Bar 3 and Bar 4, "
            "e.g. GUESS X..X.X.. .X..X..X (reward 0.3 for an exact Bar 3, 0.4 for an exact Bar 4, first "
            "time each; episode succeeds once both are exact)\n\n"
            "Malformed actions cost a step and get a corrective message but no reward."
        )

    def step(self, action):
        self.steps += 1
        info = {'steps': self.steps}
        reward = 0.0
        terminated = False

        tokens = (action or '').strip().split()
        if not tokens:
            obs = "Empty action. Use TEST, REVEAL, SUBMIT_RULE, or GUESS as described."
            return obs, reward, terminated, self._check_truncate(), info

        cmd = tokens[0].upper()

        if cmd == 'TEST' and len(tokens) == 2:
            name = tokens[1].upper()
            if name not in RULES:
                obs = f"Unknown rule name '{tokens[1]}'. Valid names: {', '.join(RULES)}."
            else:
                pred = RULES[name](self.bar1)
                revealed = [i for i in range(L) if i not in self.masked]
                matches = sum(1 for i in revealed if pred[i] == self.bar2[i])
                total = len(revealed)
                consistent = "consistent with all visible steps" if matches == total else "NOT fully consistent"
                obs = (f"TEST {name}: matches {matches}/{total} visible Bar 2 steps ({consistent})."
                       f"\nBar 2 so far: {self._bar2_display()}")

        elif cmd == 'REVEAL' and len(tokens) == 2:
            try:
                pos = int(tokens[1])
            except ValueError:
                pos = None
            if pos is None or pos < 1 or pos > L:
                obs = f"Invalid position '{tokens[1]}'. Use REVEAL <position 1-8>."
            elif (pos - 1) not in self.masked:
                obs = f"Step {pos} is already visible: {self._bar2_display()}"
            else:
                self.masked.discard(pos - 1)
                obs = f"Revealed step {pos} of Bar 2. Bar 2 so far: {self._bar2_display()}"

        elif cmd == 'SUBMIT_RULE' and len(tokens) == 2:
            name = tokens[1].upper()
            if name not in RULES:
                obs = f"Unknown rule name '{tokens[1]}'. Valid names: {', '.join(RULES)}."
            elif name == self.rule_name:
                if not self.rule_solved:
                    self.rule_solved = True
                    reward += 0.3
                    obs = f"Correct! {name} is the hidden rule. (+0.3)"
                else:
                    obs = f"Correct, {name} is the hidden rule (already credited)."
            else:
                obs = f"{name} is not the hidden rule."

        elif cmd == 'GUESS' and len(tokens) == 3:
            g3, g4 = tokens[1].upper(), tokens[2].upper()
            valid3 = len(g3) == L and set(g3) <= {'X', '.'}
            valid4 = len(g4) == L and set(g4) <= {'X', '.'}
            if not (valid3 and valid4):
                obs = "Each guess must be an 8-character string of only 'X' and '.'. Format: GUESS <bar3> <bar4>."
            else:
                match3 = sum(1 for i in range(L) if g3[i] == self.bar3[i])
                match4 = sum(1 for i in range(L) if g4[i] == self.bar4[i])
                lines = [f"Bar 3 guess matched {match3}/{L} steps.", f"Bar 4 guess matched {match4}/{L} steps."]
                if match3 == L and not self.bar3_solved:
                    self.bar3_solved = True
                    reward += 0.3
                    lines.append("Bar 3 is exactly correct! (+0.3)")
                if match4 == L and not self.bar4_solved:
                    self.bar4_solved = True
                    reward += 0.4
                    lines.append("Bar 4 is exactly correct! (+0.4)")
                if self.bar3_solved and self.bar4_solved:
                    terminated = True
                    lines.append("Both bars confirmed exact. Episode complete.")
                obs = "\n".join(lines)

        else:
            obs = ("Unrecognized action. Use one of:\n"
                   " TEST <RULE_NAME>\n REVEAL <position 1-8>\n SUBMIT_RULE <RULE_NAME>\n GUESS <bar3> <bar4>")

        truncated = self._check_truncate() and not terminated
        reward = max(0.0, min(1.0, reward))
        return obs, reward, terminated, truncated, info

    def _check_truncate(self):
        return self.steps >= MAX_STEPS
