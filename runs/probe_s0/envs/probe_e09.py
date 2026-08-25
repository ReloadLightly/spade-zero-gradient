import random


class SealedRoomSwitchesEnv:
    """Deduce which of 4 switches controls which of 4 bulbs in 2 sealed rooms."""

    SWITCHES = ["W", "X", "Y", "Z"]
    ROOMS = {"NORTH": ["N1", "N2"], "SOUTH": ["S1", "S2"]}
    BULBS = ["N1", "N2", "S1", "S2"]
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        shuffled = self.BULBS[:]
        self.rng.shuffle(shuffled)
        self.mapping = dict(zip(self.SWITCHES, shuffled))  # switch -> bulb
        self.switch_state = {s: False for s in self.SWITCHES}
        self.step_count = 0
        self.terminated = False
        obs = (
            "You control 4 switches (W, X, Y, Z) from a control room. Each switch "
            "is wired to exactly one bulb among 4 bulbs, split between two sealed "
            "rooms: NORTH (bulbs N1, N2) and SOUTH (bulbs S1, S2). The wiring is a "
            "fixed, unknown one-to-one mapping. You cannot see a room's bulbs "
            "without entering it.\n"
            "Goal: determine the full switch-to-bulb mapping.\n"
            "Actions (one per turn, 10 turns max):\n"
            "  FLIP <letters>   - toggle one or more switches, e.g. FLIP W Y\n"
            "  LOOK <room>      - enter NORTH or SOUTH and read that room's bulb states\n"
            "  GUESS <w=?> <x=?> <y=?> <z=?> - final answer, e.g. GUESS W=N1 X=N2 Y=S1 Z=S2\n"
            "GUESS ends the episode immediately (reward = fraction of switches you got right).\n"
            f"Switch panel: {self._panel_str()}"
        )
        info = {"seed": seed, "steps_remaining": self.MAX_STEPS}
        return obs, info

    def _panel_str(self):
        return " ".join(f"{s}={'ON' if self.switch_state[s] else 'OFF'}" for s in self.SWITCHES)

    def _room_str(self, room):
        parts = []
        for bulb in self.ROOMS[room]:
            switch = next(s for s, b in self.mapping.items() if b == bulb)
            state = "ON" if self.switch_state[switch] else "OFF"
            parts.append(f"{bulb}={state}")
        return " ".join(parts)

    def step(self, action):
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        text = (action or "").strip()
        upper = text.upper()
        tokens = upper.split()

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if not tokens:
            obs = "Empty action. Use FLIP <letters>, LOOK <room>, or GUESS <w=?> <x=?> <y=?> <z=?>."
        elif tokens[0] == "FLIP" and len(tokens) >= 2:
            letters = tokens[1:]
            if all(t in self.SWITCHES for t in letters):
                for t in letters:
                    self.switch_state[t] = not self.switch_state[t]
                obs = f"Switches toggled. Switch panel: {self._panel_str()}"
            else:
                obs = "Malformed FLIP: letters must be from W, X, Y, Z, e.g. FLIP W Y."
        elif tokens[0] == "LOOK" and len(tokens) == 2 and tokens[1] in self.ROOMS:
            room = tokens[1]
            obs = f"{room} room: {self._room_str(room)}"
        elif tokens[0] == "GUESS" and len(tokens) == 5:
            pairs = tokens[1:]
            guess_map = {}
            valid = True
            for p in pairs:
                if "=" not in p:
                    valid = False
                    break
                sw, bulb = p.split("=", 1)
                if sw not in self.SWITCHES or bulb not in self.BULBS or sw in guess_map:
                    valid = False
                    break
                guess_map[sw] = bulb
            if valid and len(guess_map) == 4:
                correct = sum(1 for s in self.SWITCHES if guess_map[s] == self.mapping[s])
                reward = correct / 4.0
                terminated = True
                info["success"] = correct == 4
                info["correct_pairs"] = correct
                obs = (
                    f"Guess resolved: {correct}/4 switch-bulb pairs correct. "
                    f"True mapping: {', '.join(f'{s}={b}' for s, b in self.mapping.items())}."
                )
            else:
                obs = (
                    "Malformed GUESS: give all four switches exactly once, "
                    "e.g. GUESS W=N1 X=N2 Y=S1 Z=S2."
                )
        else:
            obs = (
                "Unrecognized action. Use FLIP <letters>, LOOK <room>, "
                "or GUESS <w=?> <x=?> <y=?> <z=?>."
            )

        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            obs += " Step limit reached without a final guess."

        self.terminated = terminated
        info["steps_remaining"] = max(0, self.MAX_STEPS - self.step_count)
        return obs, reward, terminated, truncated, info
