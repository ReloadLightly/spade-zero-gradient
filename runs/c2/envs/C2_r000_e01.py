import re
import random


class PopulationRegimeEnv:
    """Alternating scarce/abundant growth regimes; solver must infer the
    regime schedule from partial reveals and guesses, then predict ahead."""

    def __init__(self):
        self.rng = None
        self.seq = []
        self.k = None
        self.d = None
        self.r = None
        self.revealed = set()
        self.guess_reward_count = 0
        self.step_count = 0
        self.max_steps = 10
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.k = self.rng.choice([2, 3])
        self.d = self.rng.randint(2, 6)
        self.r = self.rng.choice([2, 3])
        p0 = self.rng.randint(5, 20)

        seq = [p0]
        for s in range(1, 11):
            block = (s - 1) // self.k
            if block % 2 == 0:
                seq.append(seq[-1] + self.d)
            else:
                seq.append(seq[-1] * self.r)
        self.seq = seq

        self.revealed = {0, 1, 2}
        self.guess_reward_count = 0
        self.step_count = 0
        self.done = False

        obs = (
            "POPULATION GROWTH PATTERN GAME\n"
            "A population is tracked over 10 seasons (indices 1-10) starting "
            "from index 0. Growth alternates between two hidden regimes: a "
            "SCARCE regime (adds a fixed amount each season) and an ABUNDANT "
            "regime (multiplies by a fixed factor each season). The regime "
            "switches after a fixed, hidden number of seasons and keeps "
            "alternating between the two.\n"
            f"Known values: index0={seq[0]}, index1={seq[1]}, index2={seq[2]}.\n"
            "Indices 3-9 are hidden. Index 10 is the value you must predict.\n"
            "Actions (exactly one per turn):\n"
            "  REVEAL <index>          - reveal the true value at a hidden "
            "index (3-9). Costs a turn, no reward.\n"
            "  GUESS <index> <value>   - guess the value at a still-hidden "
            "index (3-9). A correct guess on an index you have NOT revealed "
            "earns reward (up to 3 such correct guesses count; feedback says "
            "too low/too high/correct).\n"
            "  PREDICT <value>         - final guess for index 10; ends the "
            "episode; reward if exact.\n"
            f"You have {self.max_steps} turns total."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        text = (action or "").strip()
        m = re.match(r'^(REVEAL|GUESS|PREDICT)\s*(.*)$', text, re.IGNORECASE)
        if not m:
            obs = ("Malformed action. Use REVEAL <index>, "
                   "GUESS <index> <value>, or PREDICT <value>.")
        else:
            cmd = m.group(1).upper()
            rest = m.group(2).strip().split()

            if cmd == 'REVEAL':
                if len(rest) != 1 or not rest[0].lstrip('-').isdigit():
                    obs = "Malformed REVEAL. Use REVEAL <index> with index 3-9."
                else:
                    idx = int(rest[0])
                    if idx < 3 or idx > 9:
                        obs = f"Index {idx} is not a valid hidden index (must be 3-9)."
                    else:
                        self.revealed.add(idx)
                        obs = f"Index {idx} value is {self.seq[idx]}."

            elif cmd == 'GUESS':
                if (len(rest) != 2 or not rest[0].lstrip('-').isdigit()
                        or not rest[1].lstrip('-').isdigit()):
                    obs = "Malformed GUESS. Use GUESS <index> <value>, both integers."
                else:
                    idx = int(rest[0])
                    val = int(rest[1])
                    if idx < 3 or idx > 9:
                        obs = f"Index {idx} is not a valid hidden index (must be 3-9)."
                    else:
                        true_val = self.seq[idx]
                        if val == true_val:
                            if idx not in self.revealed and self.guess_reward_count < 3:
                                reward = 0.2
                                self.guess_reward_count += 1
                                obs = (f"Correct! Index {idx} = {val}. "
                                       f"({self.guess_reward_count}/3 scored guesses)")
                            else:
                                obs = (f"Correct, index {idx} = {val}, but no "
                                       "further reward available for this guess.")
                            self.revealed.add(idx)
                        elif val < true_val:
                            obs = f"Too low for index {idx}."
                        else:
                            obs = f"Too high for index {idx}."

            elif cmd == 'PREDICT':
                if len(rest) != 1 or not rest[0].lstrip('-').isdigit():
                    obs = "Malformed PREDICT. Use PREDICT <value>."
                else:
                    val = int(rest[0])
                    true_val = self.seq[10]
                    terminated = True
                    self.done = True
                    if val == true_val:
                        reward = 0.4
                        obs = f"Correct! Index 10 = {val}. Episode complete."
                    else:
                        obs = f"Incorrect. Index 10 was {true_val}. Episode complete."
            else:
                obs = ("Malformed action. Use REVEAL <index>, "
                       "GUESS <index> <value>, or PREDICT <value>.")

        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            obs += f" Step limit reached ({self.max_steps})."

        return obs, reward, terminated, truncated, {}
