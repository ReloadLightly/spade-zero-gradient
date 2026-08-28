import random


class GreenWaveAvenueEnv:
    def __init__(self):
        self.num_signals = 5
        self.cycle = 12
        self.green_len = 3
        self.max_steps = 10

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.done = False

        gaps = [self.rng.randint(2, 5) for _ in range(self.num_signals)]
        self.travel_time = []
        total = 0
        for g in gaps:
            total += g
            self.travel_time.append(total)

        self.offsets = [self.rng.randrange(self.cycle) for _ in range(self.num_signals)]

        obs = (
            f"GREEN WAVE AVENUE: {self.num_signals} traffic signals lie along an avenue. "
            f"Each signal has a hidden repeating light cycle of {self.cycle} minutes, green "
            f"for {self.green_len} consecutive minutes each cycle, then red the rest of the "
            f"cycle. You must choose ONE departure time t0 (an integer minute, taken mod "
            f"{self.cycle}) for a single trip down the avenue, trying to hit as many green "
            f"lights as possible.\n"
            f"Known travel times from the avenue start to each signal (minutes): "
            f"{self.travel_time}\n"
            "Actions:\n"
            "  PROBE <i>  - ask when signal i (0-4) next turns green, measured from minute 0 "
            "(costs a step)\n"
            "  SUBMIT <t0> - commit to departure time t0 and end the episode (irreversible)\n"
            f"You have {self.max_steps} steps total. Use exactly these action formats."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        parts = action.strip().split()

        if len(parts) == 2 and parts[0].upper() == "PROBE":
            idx_str = parts[1]
            if not idx_str.lstrip("-").isdigit():
                obs = "Malformed PROBE: give a signal index 0-4, e.g. 'PROBE 2'."
            else:
                i = int(idx_str)
                if i < 0 or i >= self.num_signals:
                    obs = f"Malformed PROBE: signal index must be 0-{self.num_signals - 1}."
                else:
                    d = self.offsets[i] % self.cycle
                    if d == 0:
                        obs = f"Signal {i}: it is green right at minute 0 (green now)."
                    else:
                        obs = (
                            f"Signal {i}: departing at minute 0, it would next turn green "
                            f"in {d} minute(s)."
                        )
        elif len(parts) == 2 and parts[0].upper() == "SUBMIT":
            t_str = parts[1]
            if not t_str.lstrip("-").isdigit():
                obs = "Malformed SUBMIT: give an integer minute, e.g. 'SUBMIT 5'."
            else:
                t0 = int(t_str) % self.cycle
                hits = self._count_hits(t0)
                best = self._best_possible()
                reward = hits / best
                terminated = True
                self.done = True
                obs = (
                    f"Trip complete: departing at minute {t0} hit {hits} of "
                    f"{self.num_signals} green lights (best possible was {best}). "
                    "Episode over."
                )
        else:
            obs = "Malformed action. Use 'PROBE <i>' or 'SUBMIT <t0>'."

        if not self.done and self.step_count >= self.max_steps:
            truncated = True
            self.done = True
            obs += " Step limit reached; episode truncated with no submission."

        return obs, reward, terminated, truncated, {}

    def _count_hits(self, t0):
        hits = 0
        for i in range(self.num_signals):
            arrival = (t0 + self.travel_time[i]) % self.cycle
            rel = (arrival - self.offsets[i]) % self.cycle
            if rel < self.green_len:
                hits += 1
        return hits

    def _best_possible(self):
        return max(self._count_hits(t0) for t0 in range(self.cycle))
