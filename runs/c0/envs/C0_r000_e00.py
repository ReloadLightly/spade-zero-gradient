import random


class MissingTrainDeductionEnv:
    NUM_TRAINS = 4
    MAX_STEPS = 10

    def __init__(self):
        self.rng = None
        self.step_count = 0
        self.terminated = False
        self.trains = {}
        self.phantom_arrival = None
        self.log = []
        self.revealed_ids = set()
        self.ratio = None
        self.station_names = None

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.terminated = False
        self.revealed_ids = set()

        stories = [
            ("Ashcombe", "Brindle Junction", "Corvane Terminus"),
            ("Millhaven", "Ostercroft", "Fenwick End"),
            ("Ravensdale", "Quillmoor", "Thornstead"),
            ("Selbury", "Harmond Cross", "Kettlewick"),
        ]
        s1, s2, s3 = self.rng.choice(stories)
        self.station_names = (s1, s2, s3)
        self.ratio = self.rng.choice([2, 3, 4])

        base_depart = self.rng.randint(360, 480)
        departs = []
        d = base_depart
        for _ in range(self.NUM_TRAINS):
            d += self.rng.randint(6, 14)
            departs.append(d)

        self.trains = {}
        arrivals_s3 = []
        for i, dep in enumerate(departs):
            tid = "T%d" % (i + 1)
            t1 = self.rng.randint(8, 22)
            arrival_s3 = dep + self.ratio * t1
            self.trains[tid] = {
                "depart": dep,
                "arrival_s2": dep + t1,
                "arrival_s3": arrival_s3,
            }
            arrivals_s3.append(arrival_s3)

        for _ in range(200):
            phantom_dep = self.rng.randint(base_depart - 10, departs[-1] + 10)
            phantom_t1 = self.rng.randint(8, 22)
            phantom_arrival = phantom_dep + self.ratio * phantom_t1
            if phantom_arrival not in arrivals_s3:
                break
        self.phantom_arrival = phantom_arrival
        self.log = sorted(arrivals_s3 + [phantom_arrival])

        roster_lines = "\n".join(
            "  %s -- departs %s at %d" % (tid, s1, self.trains[tid]["depart"])
            for tid in self.trains
        )
        log_line = ", ".join(str(t) for t in self.log)
        seg_word = "segment" if self.ratio - 1 == 1 else "segments"

        obs = (
            "You are auditing the timetable for the %s-%s-%s line.\n"
            "Distances (in equal track segments): %s to %s is 1 segment; "
            "%s to %s is %d %s. Every train on this line runs at its own "
            "constant speed for the whole journey.\n"
            "Published roster (departure time from %s, minutes after midnight):\n"
            "%s\n"
            "The %s arrivals board recorded these times today: %s\n"
            "That is %d recorded arrivals for %d rostered trains -- exactly one "
            "arrival belongs to a train that is not on the roster.\n"
            "Actions (one per turn):\n"
            "  REVEAL <train_id> -- learn that train's arrival time at %s "
            "(costs a step)\n"
            "  SUBMIT <time> -- declare the arrival time you believe belongs "
            "to the unlisted train\n"
            "You have %d steps total."
            % (
                s1, s2, s3,
                s1, s2, s2, s3, self.ratio - 1, seg_word,
                s1, roster_lines,
                s3, log_line,
                len(self.log), self.NUM_TRAINS,
                s2,
                self.MAX_STEPS,
            )
        )
        return obs, {"log": list(self.log), "num_trains": self.NUM_TRAINS}

    def step(self, action):
        if self.terminated:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        action = (action or "").strip()
        parts = action.split()
        s1, s2, s3 = self.station_names
        reward = 0.0
        terminated = False

        if len(parts) == 2 and parts[0].upper() == "REVEAL":
            tid = parts[1].upper()
            if tid not in self.trains:
                obs = "Unknown train id '%s'. Known ids: %s." % (
                    parts[1], ", ".join(sorted(self.trains))
                )
            elif tid in self.revealed_ids:
                obs = "%s was already revealed (arrival at %s = %d)." % (
                    tid, s2, self.trains[tid]["arrival_s2"]
                )
            else:
                self.revealed_ids.add(tid)
                reward = 0.4 / self.NUM_TRAINS
                obs = (
                    "%s departs %s at %d and arrives %s at %d. (%d/%d trains revealed)"
                    % (
                        tid, s1, self.trains[tid]["depart"], s2,
                        self.trains[tid]["arrival_s2"],
                        len(self.revealed_ids), self.NUM_TRAINS,
                    )
                )
        elif len(parts) == 2 and parts[0].upper() == "SUBMIT":
            try:
                guess = int(parts[1])
            except ValueError:
                obs = "SUBMIT needs an integer time, e.g. SUBMIT 512."
            else:
                if guess == self.phantom_arrival:
                    reward = 0.6
                    terminated = True
                    obs = (
                        "Correct. %d at %s matches no rostered train's true speed "
                        "-- that is the missing service." % (guess, s3)
                    )
                elif guess not in self.log:
                    obs = "%d never appears on the %s arrivals board. Check the log again." % (
                        guess, s3
                    )
                else:
                    obs = (
                        "%d is on the board, but once you work out its speed it "
                        "matches a rostered train. Not the answer." % guess
                    )
        else:
            obs = "Malformed action. Use 'REVEAL <train_id>' or 'SUBMIT <time>'."

        self.terminated = terminated
        truncated = (not terminated) and self.step_count >= self.MAX_STEPS
        info = {"revealed": sorted(self.revealed_ids), "step": self.step_count}
        return obs, reward, terminated, truncated, info
