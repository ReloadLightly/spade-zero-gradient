import random
import re


class MissingTrainEnv:
    """Isolate the one station-log entry that no known train's timetable
    can explain, revealing a train missing from the schedule."""

    STATIONS = ['A', 'B', 'C', 'D']
    POS = {'A': 0, 'B': 8, 'C': 20, 'D': 35}
    TRAIN_IDS = ['T1', 'T2']
    MAX_STEPS = 10
    MAX_DEP = 30

    def __init__(self):
        self.rng = None
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.trains = {}
        self.missing = {}
        self.station_logs = {}
        self.explained = {}
        self.anomalous = set()
        self.anomalous_stations = set()
        self.partial_awarded = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.partial_awarded = False

        self.trains = {}
        explained = {}
        for tid in self.TRAIN_IDS:
            origin, direction, dep, visits = self._make_train()
            self.trains[tid] = {'origin': origin, 'dir': direction,
                                 'dep': dep, 'visits': visits}
            for s, t in visits.items():
                explained[(s, t)] = tid
        self.explained = explained

        origin, direction, dep, visits = None, None, None, None
        for _ in range(500):
            origin, direction, dep, visits = self._make_train()
            if len(visits) < 2:
                continue
            if any((s, t) in explained for s, t in visits.items()):
                continue
            break
        self.missing = {'origin': origin, 'dir': direction, 'dep': dep,
                         'visits': visits}

        self.anomalous = set(self.missing['visits'].items())
        self.anomalous_stations = {s for s, _ in self.anomalous}

        logs = {s: [] for s in self.STATIONS}
        for tr in self.trains.values():
            for s, t in tr['visits'].items():
                logs[s].append(t)
        for s, t in self.missing['visits'].items():
            logs[s].append(t)
        self.station_logs = {s: sorted(set(v)) for s, v in logs.items()}

        return self._intro(), {}

    def _make_train(self):
        oi = self.rng.randrange(len(self.STATIONS))
        if oi == 0:
            direction = 1
        elif oi == len(self.STATIONS) - 1:
            direction = -1
        else:
            direction = self.rng.choice([1, -1])
        dep = self.rng.randint(0, self.MAX_DEP)
        origin = self.STATIONS[oi]
        idxs = range(oi, len(self.STATIONS)) if direction == 1 else range(0, oi + 1)
        visits = {}
        for i in idxs:
            s = self.STATIONS[i]
            visits[s] = dep + abs(self.POS[s] - self.POS[origin])
        return origin, direction, dep, visits

    def _intro(self):
        lines = [
            "You are a dispatch inspector. Station logs record the arrival "
            "time of every train that passed, but entries carry no train ID. "
            "Two scheduled trains, T1 and T2, are on file. One log entry "
            "belongs to neither -- a train missing from the schedule. Find "
            "one (station, time) entry that cannot be produced by T1 or T2.",
            "All trains move at a constant 1 position-unit per minute. "
            "Station positions: " +
            ", ".join(f"{s}={self.POS[s]}" for s in self.STATIONS) + ".",
            f"Known train IDs: {', '.join(self.TRAIN_IDS)}.",
            "Actions (one per turn): 'SCHEDULE <id>' reveals a known train's "
            "origin, direction and departure time; 'LOG <station>' lists "
            "every recorded arrival time at that station (unlabeled); "
            "'ANSWER <station> <time>' submits the entry you believe cannot "
            "belong to T1 or T2.",
            f"You have {self.MAX_STEPS} steps total.",
        ]
        return "\n".join(lines)

    def step(self, action):
        if self.terminated or self.truncated:
            return "Episode already finished.", 0.0, self.terminated, self.truncated, {}

        self.step_count += 1
        reward = 0.0
        parts = (action or "").strip().split()

        if not parts:
            obs = "Empty action. Use SCHEDULE <id>, LOG <station>, or ANSWER <station> <time>."
        else:
            cmd = parts[0].upper()
            if cmd == 'SCHEDULE' and len(parts) == 2:
                tid = parts[1].upper()
                if tid in self.trains:
                    tr = self.trains[tid]
                    d = 'toward D (increasing position)' if tr['dir'] == 1 else 'toward A (decreasing position)'
                    obs = f"{tid}: departs {tr['origin']} at t={tr['dep']}, heading {d}."
                else:
                    obs = f"No such train '{tid}'. Known IDs: {', '.join(self.TRAIN_IDS)}."
            elif cmd == 'LOG' and len(parts) == 2:
                st = parts[1].upper()
                if st in self.station_logs:
                    obs = f"Station {st} log (unlabeled arrival times): {self.station_logs[st]}"
                else:
                    obs = f"No such station '{st}'. Stations: {', '.join(self.STATIONS)}."
            elif cmd == 'ANSWER' and len(parts) == 3:
                st = parts[1].upper()
                tstr = parts[2]
                if st not in self.station_logs:
                    obs = f"No such station '{st}'. Stations: {', '.join(self.STATIONS)}."
                elif not re.fullmatch(r'-?\d+', tstr):
                    obs = "Time must be an integer, e.g. ANSWER C 27."
                else:
                    t = int(tstr)
                    if (st, t) in self.anomalous:
                        reward = 1.0 - (0.4 if self.partial_awarded else 0.0)
                        self.terminated = True
                        obs = (f"Confirmed: no scheduled train explains a sighting at "
                               f"{st} at t={t}. You have identified the missing train's "
                               f"footprint.")
                    elif (st, t) in self.explained:
                        obs = f"That entry is explained by {self.explained[(st, t)]}'s schedule -- not the anomaly."
                    elif st in self.anomalous_stations:
                        if not self.partial_awarded:
                            reward = 0.4
                            self.partial_awarded = True
                            obs = f"Station {st} does hold the unexplained entry, but t={t} is not it. Check the LOG again."
                        else:
                            obs = f"Station {st} is right, but t={t} is still wrong."
                    else:
                        obs = f"No unexplained entry at {st}. Try a different station."
            else:
                obs = ("Unrecognized action. Use 'SCHEDULE <id>', 'LOG <station>', "
                       "or 'ANSWER <station> <time>'.")

        if not self.terminated and self.step_count >= self.MAX_STEPS:
            self.truncated = True
            obs += f"\nStep limit ({self.MAX_STEPS}) reached."

        return obs, reward, self.terminated, self.truncated, {}
