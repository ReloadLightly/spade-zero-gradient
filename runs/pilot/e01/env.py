import random
import string


class MissingTrainEnv:
    STATIONS = ["Ashcombe", "Bridgeworth", "Calderfield", "Dunraven", "Emberholt"]
    MAX_STEPS = 10
    TOTAL_TRAINS = 5

    def __init__(self):
        self.rng = None
        self.trains = []
        self.printed_trains = []
        self.station_logs = {}
        self.missing_train = None
        self.steps = 0
        self.terminated = False
        self.origin_awarded = False
        self.direction_awarded = False
        self.time_awarded = False

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self._generate_world()
        self.steps = 0
        self.terminated = False
        self.origin_awarded = False
        self.direction_awarded = False
        self.time_awarded = False
        obs = self._initial_observation()
        info = {"stations": list(self.STATIONS)}
        return obs, info

    def _generate_world(self):
        n = len(self.STATIONS)
        trains = None
        for _ in range(300):
            candidate = []
            time_map = {i: {} for i in range(n)}
            collision = False
            for _t in range(self.TOTAL_TRAINS):
                direction = self.rng.choice(["EAST", "WEST"])
                if direction == "EAST":
                    origin_idx = self.rng.randint(0, n - 2)
                    passed = list(range(origin_idx, n))
                else:
                    origin_idx = self.rng.randint(1, n - 1)
                    passed = list(range(origin_idx, -1, -1))
                start_time = self.rng.randint(2, 15)
                times = {}
                for idx in passed:
                    tm = start_time + abs(idx - origin_idx)
                    if tm in time_map[idx]:
                        collision = True
                        break
                    times[idx] = tm
                if collision:
                    break
                for idx, tm in times.items():
                    time_map[idx][tm] = _t
                candidate.append({
                    "direction": direction,
                    "origin_idx": origin_idx,
                    "start_time": start_time,
                    "times": times,
                })
            if not collision and len(candidate) == self.TOTAL_TRAINS:
                trains = candidate
                break
        if trains is None:
            trains = []
            for t in range(self.TOTAL_TRAINS):
                direction = "EAST" if t % 2 == 0 else "WEST"
                if direction == "EAST":
                    origin_idx = t % (n - 1)
                    passed = list(range(origin_idx, n))
                else:
                    origin_idx = 1 + (t % (n - 1))
                    passed = list(range(origin_idx, -1, -1))
                start_time = 2 + t * 20
                times = {idx: start_time + abs(idx - origin_idx) for idx in passed}
                trains.append({
                    "direction": direction,
                    "origin_idx": origin_idx,
                    "start_time": start_time,
                    "times": times,
                })

        missing_index = self.rng.randrange(self.TOTAL_TRAINS)
        self.trains = trains
        self.missing_train = trains[missing_index]
        self.station_logs = {
            i: sorted(tr["times"][i] for tr in trains if i in tr["times"])
            for i in range(n)
        }
        self.printed_trains = [tr for j, tr in enumerate(trains) if j != missing_index]

    def _initial_observation(self):
        lines = []
        lines.append("MISSING TRAIN INVESTIGATION")
        idx_names = ", ".join(f"{i}:{name}" for i, name in enumerate(self.STATIONS))
        lines.append(f"Stations in line order (index shown, 1 time unit per adjacent gap): {idx_names}")
        lines.append("A train travels EAST (increasing index) or WEST (decreasing index) from its origin to the end of the line, at 1 station per time unit.")
        lines.append("PRINTED TIMETABLE (every KNOWN train's origin, direction, departure time):")
        for i, tr in enumerate(self.printed_trains):
            letter = string.ascii_uppercase[i]
            lines.append(
                f"  Train {letter}: origin={self.STATIONS[tr['origin_idx']]}, "
                f"direction={tr['direction']}, departure_time={tr['start_time']}"
            )
        lines.append(
            "Station sensors independently log the arrival TIME (no train ID) of every train that "
            "actually passes, including any train missing from the printed timetable. Exactly one "
            "real train is missing from the printed timetable above."
        )
        lines.append("GOAL: determine the missing train's origin station, direction, and departure time.")
        lines.append("ACTIONS (exact formats):")
        lines.append("  LOG <station_name>")
        lines.append("  GUESS <station_name> <EAST|WEST> <departure_time>")
        lines.append(
            f"You have {self.MAX_STEPS} steps total. Correct guess components (origin, direction, "
            "time) each earn credit the first time you get them right; a fully correct GUESS ends the episode."
        )
        return "\n".join(lines)

    def _station_index(self, name):
        name_low = name.strip().lower()
        for i, s in enumerate(self.STATIONS):
            if s.lower() == name_low:
                return i
        return None

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        info = {}

        if self.terminated:
            return "Episode already finished.", 0.0, True, False, info

        parts = (action or "").strip().split()
        if not parts:
            obs = "Empty action. Use 'LOG <station_name>' or 'GUESS <station_name> <EAST|WEST> <time>'."
            return self._finish(obs, 0.0, False)

        verb = parts[0].upper()

        if verb == "LOG":
            if len(parts) != 2:
                obs = "Malformed LOG action. Format: LOG <station_name>"
                return self._finish(obs, 0.0, False)
            idx = self._station_index(parts[1])
            if idx is None:
                obs = f"Unknown station '{parts[1]}'. Valid stations: {', '.join(self.STATIONS)}"
                return self._finish(obs, 0.0, False)
            times = self.station_logs[idx]
            obs = f"LOG[{self.STATIONS[idx]}]: recorded arrival times = {times}"
            return self._finish(obs, 0.0, False)

        if verb == "GUESS":
            if len(parts) != 4:
                obs = "Malformed GUESS action. Format: GUESS <station_name> <EAST|WEST> <departure_time>"
                return self._finish(obs, 0.0, False)
            idx = self._station_index(parts[1])
            direction = parts[2].upper()
            if idx is None:
                obs = f"Unknown station '{parts[1]}'. Valid stations: {', '.join(self.STATIONS)}"
                return self._finish(obs, 0.0, False)
            if direction not in ("EAST", "WEST"):
                obs = "Direction must be EAST or WEST."
                return self._finish(obs, 0.0, False)
            try:
                guess_time = int(parts[3])
            except ValueError:
                obs = "Departure time must be an integer."
                return self._finish(obs, 0.0, False)

            correct_origin = idx == self.missing_train["origin_idx"]
            correct_dir = direction == self.missing_train["direction"]
            correct_time = guess_time == self.missing_train["start_time"]

            gained = 0.0
            if correct_origin and not self.origin_awarded:
                gained += 0.3
                self.origin_awarded = True
            if correct_dir and not self.direction_awarded:
                gained += 0.3
                self.direction_awarded = True
            if correct_time and not self.time_awarded:
                gained += 0.4
                self.time_awarded = True

            reward = gained
            if correct_origin and correct_dir and correct_time:
                terminated = True
                obs = (
                    f"CORRECT. The missing train departed {self.STATIONS[idx]} heading {direction} "
                    f"at time {guess_time}. Investigation closed."
                )
            else:
                fb = []
                fb.append("origin " + ("correct" if correct_origin else "incorrect"))
                fb.append("direction " + ("correct" if correct_dir else "incorrect"))
                fb.append("time " + ("correct" if correct_time else "incorrect"))
                obs = "Guess evaluated: " + ", ".join(fb) + ". Not fully solved yet."
            return self._finish(obs, reward, terminated)

        obs = "Unrecognized action verb. Use 'LOG <station_name>' or 'GUESS <station_name> <EAST|WEST> <time>'."
        return self._finish(obs, 0.0, False)

    def _finish(self, obs, reward, terminated):
        self.terminated = terminated
        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
        info = {"step": self.steps}
        return obs, reward, terminated, truncated, info
