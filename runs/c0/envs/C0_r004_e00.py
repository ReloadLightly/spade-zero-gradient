import random
import re


class PasswordRiddleEnv:
    """Recover a 4-digit distinct-digit password using overheard riddle
    fragments plus Mastermind-style feedback on guesses."""

    def __init__(self):
        self.max_steps = 10
        self.secret = []
        self.clues = []
        self.clue_index = 0
        self.step_count = 0
        self.best_matches = 0

    def reset(self, seed=None):
        rng = random.Random(seed)
        self.secret = rng.sample(range(1, 10), 4)

        clue_parity = (
            "The first digit whispered is "
            + ("even" if self.secret[0] % 2 == 0 else "odd")
            + "."
        )
        clue_order = (
            "The second digit is "
            + ("greater than" if self.secret[1] > self.secret[0] else "less than")
            + " the first digit."
        )
        clue_sum = f"All four digits sum to {sum(self.secret)}."
        max_idx = self.secret.index(max(self.secret))
        clue_max_pos = f"The single largest digit sits in position {max_idx + 1} of the four."

        self.clues = [clue_parity, clue_order, clue_sum, clue_max_pos]
        rng.shuffle(self.clues)

        self.clue_index = 0
        self.step_count = 0
        self.best_matches = 0

        obs = (
            "You overheard four riddle-fragments about a guarded door's password: "
            "a 4-digit code using DISTINCT digits from 1-9.\n"
            "GOAL: recover the exact password within 10 total actions.\n\n"
            "ACTIONS (send exactly one per turn):\n"
            "  ASK            - recall the next unheard riddle-fragment (4 total).\n"
            "  GUESS wxyz     - guess the password, e.g. 'GUESS 3172' "
            "(4 distinct digits, 1-9).\n"
            "                   Feedback: 'exact' = right digit in the right "
            "position; 'present' = right digit, wrong position.\n\n"
            "You have 10 actions total. Fragment 1/4 is available now."
        )
        return obs, {}

    def step(self, action):
        self.step_count += 1
        act = (action or "").strip()
        parts = act.split(None, 1)
        cmd = parts[0].upper() if parts else ""
        remainder = parts[1] if len(parts) > 1 else ""

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if cmd == "ASK":
            if self.clue_index < len(self.clues):
                clue_text = self.clues[self.clue_index]
                self.clue_index += 1
                obs = f"Fragment {self.clue_index}/{len(self.clues)}: {clue_text}"
                if self.clue_index < len(self.clues):
                    obs += (
                        f"\n({len(self.clues) - self.clue_index} fragment(s) still "
                        "unheard - ASK again, or GUESS.)"
                    )
                else:
                    obs += "\n(That was the last fragment. GUESS when ready.)"
            else:
                obs = (
                    "No more fragments remain in memory. GUESS the password "
                    "(e.g. 'GUESS 1234')."
                )
        elif cmd == "GUESS":
            found = re.findall(r"\d", remainder)
            if len(found) != 4 or any(d == "0" for d in found) or len(set(found)) != 4:
                obs = (
                    "Malformed guess. Send exactly 4 distinct digits from 1-9, "
                    "e.g. 'GUESS 3172'."
                )
            else:
                guess = [int(d) for d in found]
                exact = sum(1 for g, s in zip(guess, self.secret) if g == s)
                present = len(set(guess) & set(self.secret)) - exact
                info = {"exact": exact, "present": present}
                if exact > self.best_matches:
                    reward += 0.2 * (exact - self.best_matches)
                    self.best_matches = exact
                if exact == 4:
                    reward += 0.2
                    terminated = True
                    obs = f"Correct! The password was {''.join(found)}. The door unlocks."
                else:
                    obs = (
                        f"Guess {''.join(found)}: {exact} exact (right digit, right "
                        f"spot), {present} present (right digit, wrong spot). "
                        f"Best exact so far: {self.best_matches}/4."
                    )
        else:
            obs = (
                "Unrecognized action. Use 'ASK' to recall a fragment, or "
                "'GUESS wxyz' with 4 distinct digits 1-9."
            )

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            obs += f"\nOut of actions. The password was {''.join(str(d) for d in self.secret)}."

        return obs, reward, terminated, truncated, info
