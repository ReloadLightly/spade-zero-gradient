import random
import heapq


class TaskDeadlineSchedulingEnv:
    """Single-machine scheduling: minimize late tasks under hidden durations."""

    N_TASKS = 5
    MAX_STEPS = 10
    DUR_MIN, DUR_MAX = 3, 9

    def __init__(self):
        self.rng = None
        self.durations = []
        self.deadlines = []
        self.probed = []
        self.steps = 0
        self.best_ratio = 0.0
        self.optimal_count = 0
        self.optimal_ontime_ids = set()
        self.done = False

    def _hodgson_optimal(self, durations, deadlines):
        order = sorted(range(len(durations)), key=lambda i: deadlines[i])
        heap = []
        t = 0
        for i in order:
            d = durations[i]
            heapq.heappush(heap, (-d, i))
            t += d
            if t > deadlines[i]:
                largest_neg, _dropped_id = heapq.heappop(heap)
                t += largest_neg
        ontime_ids = {i for (_, i) in heap}
        return len(ontime_ids), ontime_ids

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        durations, deadlines, opt_count, opt_ids = None, None, None, None
        for _ in range(200):
            cand_durations = [self.rng.randint(self.DUR_MIN, self.DUR_MAX) for _ in range(self.N_TASKS)]
            total = sum(cand_durations)
            cand_deadlines = []
            for d in cand_durations:
                lo = max(d, total // 4)
                cand_deadlines.append(self.rng.randint(lo, total))
            count, ids = self._hodgson_optimal(cand_durations, cand_deadlines)
            durations, deadlines, opt_count, opt_ids = cand_durations, cand_deadlines, count, ids
            if 1 <= count <= self.N_TASKS - 1:
                break

        self.durations = durations
        self.deadlines = deadlines
        self.optimal_count = opt_count
        self.optimal_ontime_ids = opt_ids
        self.probed = [False] * self.N_TASKS
        self.steps = 0
        self.best_ratio = 0.0
        self.done = False

        n = self.N_TASKS
        deadline_str = ", ".join(f"T{i}=deadline {self.deadlines[i]}" for i in range(n))
        obs = (
            f"You are scheduling {n} tasks (T0..T{n - 1}) on a single machine, one at a time "
            "from time 0, back-to-back with no gaps. Each task's PUBLIC deadline is listed "
            "below; its processing DURATION is hidden until probed. A task counts as 'on time' "
            "if the running total of durations up to and including it (in your chosen order) "
            "is <= its deadline. Goal: choose a full processing order maximizing the number of "
            "on-time tasks.\n"
            f"Deadlines: {deadline_str}\n"
            "Actions:\n"
            "  PROBE <id>            reveal the duration of task <id>, e.g. PROBE 2\n"
            f"  SCHEDULE <id,id,...>  submit a full order of all {n} task ids, comma-separated, "
            "e.g. SCHEDULE 0,1,2,3,4; may be resubmitted to try again\n"
            f"You have {self.MAX_STEPS} steps total. The episode ends when you match the best "
            "achievable on-time count or when steps run out."
        )
        info = {"deadlines": list(self.deadlines)}
        return obs, info

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        action = (action or "").strip()
        terminated = False
        reward = 0.0

        parts = action.split(None, 1)
        verb = parts[0].upper() if parts else ""

        if verb == "PROBE" and len(parts) == 2:
            arg = parts[1].strip()
            if arg.isdigit() and 0 <= int(arg) < self.N_TASKS:
                tid = int(arg)
                self.probed[tid] = True
                obs = f"Task T{tid} duration = {self.durations[tid]}."
            else:
                obs = f"Malformed PROBE: give a task id from 0 to {self.N_TASKS - 1}."
        elif verb == "SCHEDULE" and len(parts) == 2:
            tokens = parts[1].replace(" ", "").split(",")
            valid = (
                len(tokens) == self.N_TASKS
                and all(t.isdigit() for t in tokens)
                and sorted(int(t) for t in tokens) == list(range(self.N_TASKS))
            )
            if valid:
                order = [int(t) for t in tokens]
                t = 0
                ontime = 0
                for tid in order:
                    t += self.durations[tid]
                    if t <= self.deadlines[tid]:
                        ontime += 1
                ratio = min(ontime / self.optimal_count, 1.0) if self.optimal_count else 1.0
                reward = max(0.0, ratio - self.best_ratio)
                self.best_ratio = max(self.best_ratio, ratio)
                obs = (
                    f"Schedule {order}: {ontime}/{self.N_TASKS} tasks on time "
                    f"(best possible is {self.optimal_count}). Best ratio so far: "
                    f"{self.best_ratio:.2f}."
                )
                if ontime >= self.optimal_count:
                    terminated = True
                    self.done = True
            else:
                obs = (
                    f"Malformed SCHEDULE: list all {self.N_TASKS} task ids "
                    f"0-{self.N_TASKS - 1} exactly once, comma-separated."
                )
        else:
            obs = "Malformed action. Use 'PROBE <id>' or 'SCHEDULE <id,id,...>'."

        truncated = False
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True
            self.done = True

        info = {"steps": self.steps, "best_ratio": self.best_ratio}
        return obs, reward, terminated, truncated, info
