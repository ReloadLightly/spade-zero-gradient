import random
import re


class CellularAutomatonRuleEnv:
    def __init__(self):
        self.W = 6
        self.step_limit = 10
        self.rng = None
        self.rule_bits = None
        self.challenge = None
        self.next1 = None
        self.next2 = None
        self.steps = 0
        self.done = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.rule_bits = self._sample_rule()
        self.challenge = self._sample_challenge()
        self.next1 = self._apply(self.challenge)
        self.next2 = self._apply(self.next1)
        self.steps = 0
        self.done = False
        obs = self._intro()
        info = {"width": self.W, "step_limit": self.step_limit, "challenge": self.challenge}
        return obs, info

    def _sample_rule(self):
        bits = [0] * 8
        for _ in range(200):
            bits = [self.rng.randint(0, 1) for _ in range(8)]
            if len(set(bits)) == 1:
                continue
            if all(bits[idx] == ((idx >> 1) & 1) for idx in range(8)):
                continue
            if all(bits[idx] == 1 - ((idx >> 1) & 1) for idx in range(8)):
                continue
            return bits
        return bits

    def _sample_challenge(self):
        while True:
            s = ''.join(self.rng.choice('01') for _ in range(self.W))
            if '0' in s and '1' in s:
                return s

    def _apply(self, s):
        out = []
        for i in range(self.W):
            l = int(s[(i - 1) % self.W])
            c = int(s[i])
            r = int(s[(i + 1) % self.W])
            idx = (l << 2) | (c << 1) | r
            out.append(str(self.rule_bits[idx]))
        return ''.join(out)

    def _intro(self):
        return (
            f"Hidden 1D cellular automaton: radius-1, circular boundary, binary strings of length {self.W}.\n"
            f"Each cell's next value depends only on itself and its two neighbors (wrapping around) via a "
            f"fixed, unknown local rule.\n\n"
            f"GOAL: predict the CHALLENGE state's next state (1 step) and the state after that (2 steps).\n"
            f"CHALLENGE STATE: {self.challenge}\n\n"
            f"ACTIONS (send exactly one per turn):\n"
            f"  PROBE <bits>   -- reveals the automaton's next state for any length-{self.W} binary string "
            f"you choose (you may NOT probe the exact challenge state).\n"
            f"  PREDICT <bits1> <bits2>  -- submit your final answer (bits1 = 1 step ahead, bits2 = 2 steps "
            f"ahead) of the CHALLENGE state; this ends the episode.\n\n"
            f"You have {self.step_limit} actions total. Reward: 0.5 for a correct 1-step prediction + 0.5 "
            f"for a correct 2-step prediction."
        )

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}
        self.steps += 1
        text = (action or "").strip()
        parts = text.split()
        bitpat = re.compile(r'^[01]+$')

        if not parts:
            obs = "Empty action. Use 'PROBE <bits>' or 'PREDICT <bits1> <bits2>'."
            return self._maybe_truncate(obs, 0.0)

        verb = parts[0].upper()

        if verb == "PROBE":
            if len(parts) != 2 or not bitpat.match(parts[1]) or len(parts[1]) != self.W:
                obs = f"Malformed PROBE. Usage: PROBE <bits> with exactly {self.W} characters, each '0' or '1'."
                return self._maybe_truncate(obs, 0.0)
            bits = parts[1]
            if bits == self.challenge:
                obs = "You may not probe the exact challenge state. Choose a different string to reveal the rule."
                return self._maybe_truncate(obs, 0.0)
            nxt = self._apply(bits)
            obs = f"PROBE {bits} -> {nxt}"
            return self._maybe_truncate(obs, 0.0)

        if verb == "PREDICT":
            if len(parts) != 3 or not all(bitpat.match(p) and len(p) == self.W for p in parts[1:]):
                obs = f"Malformed PREDICT. Usage: PREDICT <bits1> <bits2>, each exactly {self.W} characters of '0'/'1'."
                return self._maybe_truncate(obs, 0.0)
            g1, g2 = parts[1], parts[2]
            reward = 0.0
            if g1 == self.next1:
                reward += 0.5
            if g2 == self.next2:
                reward += 0.5
            self.done = True
            correct1 = "correct" if g1 == self.next1 else f"incorrect (actual {self.next1})"
            correct2 = "correct" if g2 == self.next2 else f"incorrect (actual {self.next2})"
            obs = f"PREDICT submitted. 1-step: {correct1}. 2-step: {correct2}. Reward earned: {reward:.1f}."
            return obs, reward, True, False, {"reward": reward}

        obs = "Unknown action. Use 'PROBE <bits>' or 'PREDICT <bits1> <bits2>'."
        return self._maybe_truncate(obs, 0.0)

    def _maybe_truncate(self, obs, reward):
        if self.steps >= self.step_limit:
            self.done = True
            obs = obs + f"\nStep limit ({self.step_limit}) reached without a PREDICT. Episode truncated."
            return obs, reward, False, True, {}
        return obs, reward, False, False, {}
