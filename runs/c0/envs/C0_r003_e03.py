import random
import re


class ParityCardsEnv:
    """Five face-down cards, each secretly RED or BLACK. The solver asks
    parity questions about chosen groups of cards and must deduce the full
    hidden pattern before submitting one final answer."""

    NUM_CARDS = 5
    STEP_LIMIT = 10
    INFO_REWARD_UNIT = 0.12
    FINAL_REWARD = 0.4

    def __init__(self):
        self.rng = None
        self.bits = []
        self.step_count = 0
        self.rank = 0
        self.basis = [0] * self.NUM_CARDS
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.bits = [self.rng.randint(0, 1) for _ in range(self.NUM_CARDS)]
        self.step_count = 0
        self.rank = 0
        self.basis = [0] * self.NUM_CARDS
        self.done = False
        observation = (
            "Five cards lie face down, numbered 1-5. Each hides a color, "
            "RED or BLACK, fixed for the whole game. Find the exact color "
            "of every card within 10 steps.\n"
            "Actions:\n"
            "  PARITY <two or more distinct card numbers, space-separated>\n"
            "    e.g. 'PARITY 1 3 4' -> tells you EVEN or ODD, the parity "
            "of how many RED cards are among those you named.\n"
            "  GUESS <5 characters of 0/1, card 1 to 5 in order, 1=RED>\n"
            "    e.g. 'GUESS 10110' -> your one final answer; ends the game.\n"
            "You earn credit as your PARITY questions genuinely narrow down "
            "the possibilities, and a final reward if your GUESS is exactly "
            "right. You get exactly one GUESS attempt, so query first."
        )
        info = {"steps_used": 0}
        return observation, info

    def _mask(self, indices):
        mask = 0
        for i in indices:
            mask |= 1 << (i - 1)
        return mask

    def _add_to_basis(self, vec):
        for bitpos in range(self.NUM_CARDS - 1, -1, -1):
            if not (vec >> bitpos) & 1:
                continue
            if self.basis[bitpos] == 0:
                self.basis[bitpos] = vec
                return True
            vec ^= self.basis[bitpos]
        return False

    def step(self, action):
        if self.done:
            return "The game has already ended.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False
        text = (action or "").strip()
        upper = text.upper()

        if upper.startswith("PARITY"):
            rest = upper[len("PARITY"):]
            tokens = [t for t in re.split(r"[\s,]+", rest.strip()) if t]
            valid = True
            indices = []
            if len(tokens) < 2:
                valid = False
            else:
                for t in tokens:
                    if not re.fullmatch(r"[1-5]", t):
                        valid = False
                        break
                    indices.append(int(t))
                if valid and len(set(indices)) != len(indices):
                    valid = False
            if not valid:
                observation = (
                    "Malformed PARITY command. Name two or more distinct "
                    "card numbers from 1-5, separated by spaces, e.g. "
                    "'PARITY 2 5'."
                )
            else:
                count_red = sum(self.bits[i - 1] for i in indices)
                answer = "ODD" if count_red % 2 == 1 else "EVEN"
                vec = self._mask(indices)
                gained_rank = self._add_to_basis(vec)
                if gained_rank:
                    self.rank += 1
                    reward = self.INFO_REWARD_UNIT
                    observation = (
                        f"Cards {indices}: parity is {answer}. That question "
                        f"gave you new information (independent fact "
                        f"{self.rank}/{self.NUM_CARDS})."
                    )
                else:
                    observation = (
                        f"Cards {indices}: parity is {answer}. That follows "
                        f"from what you already know -- no new information."
                    )
        elif upper.startswith("GUESS"):
            rest = text[len("GUESS"):].strip()
            if not re.fullmatch(r"[01]{5}", rest):
                observation = (
                    "Malformed GUESS command. Give exactly 5 characters of "
                    "0/1 for cards 1-5 in order, e.g. 'GUESS 01101'."
                )
            else:
                terminated = True
                guessed = [int(c) for c in rest]
                if guessed == self.bits:
                    reward = self.FINAL_REWARD
                    observation = (
                        f"Correct! The hidden pattern was "
                        f"{''.join(map(str, self.bits))}."
                    )
                else:
                    observation = (
                        f"Incorrect. That was your one guess; the game ends. "
                        f"The hidden pattern was {''.join(map(str, self.bits))}."
                    )
        else:
            observation = (
                "Unrecognized action. Use 'PARITY <indices>' to ask a "
                "question or 'GUESS <5 bits>' to submit your final answer."
            )

        truncated = False
        if not terminated and self.step_count >= self.STEP_LIMIT:
            truncated = True
            observation += " Step limit reached; the game ends."

        self.done = terminated or truncated
        info = {"steps_used": self.step_count, "rank": self.rank}
        return observation, reward, terminated, truncated, info
