import re
import random


class SealedRoomWiringEnv:
    ROOMS = ['A', 'B', 'C', 'D']
    N = 4
    MAX_STEPS = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        rooms_shuffled = self.ROOMS[:]
        self.rng.shuffle(rooms_shuffled)
        self.perm = {i + 1: rooms_shuffled[i] for i in range(self.N)}
        self.switch_state = {i: self.rng.choice([True, False]) for i in range(1, self.N + 1)}
        self.steps = 0
        self.solved = False

        obs = (
            f"{self.N} switches (numbered 1-{self.N}) are wired one-to-one to {self.N} bulbs "
            f"sealed in rooms {'-'.join([self.ROOMS[0], self.ROOMS[-1]])}, via a hidden fixed "
            "wiring you must deduce. Switches begin in an unknown mix of ON/OFF states.\n"
            "Actions (exactly one per turn):\n"
            "  'flip <n>' - toggles switch n; gives no direct feedback (rooms are sealed).\n"
            "  'check' - reports the current ON/OFF status of all rooms.\n"
            f"  'solve <pairs>' - e.g. 'solve 1-A,2-B,3-C,4-D', submits your final mapping for "
            "all switches and ends the episode.\n"
            f"You have {self.MAX_STEPS} steps total. Reward is 1.0/{self.N} for each "
            "switch-room pair you get right in your solve submission."
        )
        info = {"n_switches": self.N, "rooms": list(self.ROOMS)}
        return obs, info

    def _bulb_on(self, room):
        for sw, r in self.perm.items():
            if r == room:
                return self.switch_state[sw]
        return False

    def _parse_mapping(self, s):
        parts = [p.strip() for p in s.split(',') if p.strip()]
        if len(parts) != self.N:
            return None, f"expected {self.N} pairs, got {len(parts)}"
        mapping = {}
        seen_rooms = set()
        for p in parts:
            m = re.fullmatch(r'(\d+)\s*-\s*([A-Da-d])', p)
            if not m:
                return None, f"bad pair '{p}', use format 'n-R'"
            sw = int(m.group(1))
            room = m.group(2).upper()
            if sw < 1 or sw > self.N:
                return None, f"switch {sw} out of range 1-{self.N}"
            if room not in self.ROOMS:
                return None, f"room {room} is not valid"
            if sw in mapping:
                return None, f"switch {sw} specified twice"
            if room in seen_rooms:
                return None, f"room {room} specified twice"
            mapping[sw] = room
            seen_rooms.add(room)
        return mapping, None

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        action = (action or "").strip()

        m_flip = re.fullmatch(r'flip\s+(\d+)', action, re.IGNORECASE)
        m_check = re.fullmatch(r'check', action, re.IGNORECASE)
        m_solve = re.fullmatch(r'solve\s+(.+)', action, re.IGNORECASE)

        if m_flip:
            n = int(m_flip.group(1))
            if n < 1 or n > self.N:
                obs = f"Invalid switch number '{n}'. Valid switches are 1-{self.N}."
            else:
                self.switch_state[n] = not self.switch_state[n]
                obs = f"You flipped switch {n}. Its bulb's new state is hidden — use 'check' to observe it."
        elif m_check:
            status = ", ".join(
                f"Room {r}: {'ON' if self._bulb_on(r) else 'OFF'}" for r in self.ROOMS
            )
            obs = f"Bulb status - {status}."
        elif m_solve:
            mapping, err = self._parse_mapping(m_solve.group(1))
            if err:
                obs = f"Malformed solve command ({err}). Format: 'solve 1-A,2-B,3-C,4-D'."
            else:
                correct = sum(1 for sw, room in mapping.items() if self.perm.get(sw) == room)
                reward = correct / self.N
                terminated = True
                self.solved = True
                actual = ", ".join(f"{sw}-{self.perm[sw]}" for sw in sorted(self.perm))
                obs = (
                    f"Solution submitted: {correct}/{self.N} pairs correct. "
                    f"Actual wiring was: {actual}."
                )
                info['correct'] = correct
        else:
            obs = (
                "Unrecognized action. Use 'flip <n>', 'check', or "
                "'solve <pairs>' (e.g. 'solve 1-A,2-B,3-C,4-D')."
            )

        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            obs += " Step limit reached without a solve submission — episode ends with no reward."

        obs += f" [Steps used: {self.steps}/{self.MAX_STEPS}]"
        return obs, reward, terminated, truncated, info
