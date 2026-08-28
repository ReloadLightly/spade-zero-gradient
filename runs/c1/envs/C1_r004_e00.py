import random
import re


class RiddleWardenGateEnv:
    SYMBOLS = ['F', 'K', 'N', 'V']
    SYMBOL_NAMES = {'F': 'Flame', 'K': 'Key', 'N': 'Night', 'V': 'Vow'}
    WARD_NAMES = {1: 'First Ward', 2: 'Second Ward', 3: 'Third Ward', 4: 'Fourth Ward'}
    MAX_STEPS = 10

    ASK_RE = re.compile(r'^\s*ASK\s+([1-4])\s+([FKNVfknv])\s*$')
    SUBMIT_RE = re.compile(r'^\s*SUBMIT\s+([FKNVfknv]{4})\s*$')

    def _opening(self):
        return (
            "You loiter beneath the gatehouse of a sealed door, overhearing four "
            "riddle-wardens (Wards 1-4, left to right) who each guard exactly one "
            "of four sealing-words -- FLAME(F), KEY(K), NIGHT(N), VOW(V) -- with "
            "no word repeated across wards. Your goal: recover the true password, "
            "the sequence of words at Ward 1, Ward 2, Ward 3, Ward 4.\n"
            "Actions (one per step, budget " + str(self.MAX_STEPS) + " steps total):\n"
            "  ASK <ward 1-4> <letter F/K/N/V>  -- a warden truthfully answers "
            "whether that word seals their ward (TRUE/FALSE).\n"
            "  SUBMIT <4 letters, e.g. FKNV>    -- your final password guess; ends "
            "the episode immediately. You get 0.25 reward per ward you place "
            "correctly (up to 1.0 for all four).\n"
            "You may ASK as many times as your budget allows before you SUBMIT. "
            "There is no reward for asking; SUBMIT is the only way to score, and "
            "it is final."
        )

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        symbols = list(self.SYMBOLS)
        self.rng.shuffle(symbols)
        self.solution = {ward: sym for ward, sym in zip((1, 2, 3, 4), symbols)}
        self.steps_used = 0
        self.asks_used = 0
        self.terminated = False
        self.truncated = False
        info = {'steps_used': 0}
        return self._opening(), info

    def step(self, action):
        if self.terminated or self.truncated:
            info = {'steps_used': self.steps_used}
            return "The gate is already resolved; no further actions are possible.", 0.0, True, True, info

        self.steps_used += 1
        remaining = self.MAX_STEPS - self.steps_used

        m = self.ASK_RE.match(action or '')
        if m:
            ward = int(m.group(1))
            letter = m.group(2).upper()
            self.asks_used += 1
            actual = self.solution[ward]
            correct = (actual == letter)
            ward_name = self.WARD_NAMES[ward]
            sym_name = self.SYMBOL_NAMES[letter]
            if correct:
                obs = (f"The {ward_name} warden mutters: 'Aye, the {sym_name} "
                       f"seals this ward.' (TRUE) [{remaining} steps left]")
            else:
                obs = (f"The {ward_name} warden shakes his head: 'Nay, not the "
                       f"{sym_name} here.' (FALSE) [{remaining} steps left]")
            truncated = remaining <= 0
            self.truncated = truncated
            info = {'steps_used': self.steps_used, 'asks_used': self.asks_used}
            return obs, 0.0, False, truncated, info

        m = self.SUBMIT_RE.match(action or '')
        if m:
            guess = m.group(1).upper()
            matches = sum(1 for ward in (1, 2, 3, 4) if guess[ward - 1] == self.solution[ward])
            reward = matches / 4.0
            self.terminated = True
            solved = matches == 4
            if solved:
                obs = f"You speak '{guess}' -- the gate groans and swings open. The password is confirmed!"
            else:
                reveal = ''.join(self.solution[w] for w in (1, 2, 3, 4))
                obs = (f"You speak '{guess}' -- the gate shudders but only {matches}/4 "
                       f"wards release. The true password was {reveal}.")
            info = {'steps_used': self.steps_used, 'asks_used': self.asks_used,
                    'matches': matches, 'solved': solved}
            return obs, reward, True, False, info

        obs = ("Malformed action. Use 'ASK <ward 1-4> <letter F/K/N/V>' or "
               f"'SUBMIT <4 letters>'. [{remaining} steps left]")
        truncated = remaining <= 0
        self.truncated = truncated
        info = {'steps_used': self.steps_used}
        return obs, 0.0, False, truncated, info
