import re
import itertools
import random


class FarmPlotAllocationEnv:
    def __init__(self):
        self.rng = None
        self.plots = []
        self.crops = []
        self.yields = {}
        self.scouted = set()
        self.steps = 0
        self.step_limit = 8
        self.done = False
        self.optimal_total = 0
        self.optimal_assignment = None

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.plots = ["P1", "P2", "P3"]
        self.crops = ["WHEAT", "CORN", "SOY"]
        self.yields = {
            p: {c: self.rng.randint(3, 9) for c in self.crops} for p in self.plots
        }
        self.scouted = set()
        self.steps = 0
        self.done = False

        best_total = -1
        best_assignment = None
        for perm in itertools.permutations(self.crops):
            total = sum(self.yields[self.plots[i]][perm[i]] for i in range(3))
            if total > best_total:
                best_total = total
                best_assignment = dict(zip(self.plots, perm))
        self.optimal_total = best_total
        self.optimal_assignment = best_assignment

        obs = (
            "You are allocating 3 farm plots (P1, P2, P3) among 3 crops "
            "(WHEAT, CORN, SOY). Each plot-crop pair has a hidden integer "
            "yield (3-9) revealed only by scouting. Each crop must end up "
            "on exactly one plot (a one-to-one assignment) — your goal is "
            "to maximize the total yield summed across all three plots.\n"
            "Actions (send exactly one per turn):\n"
            "  SCOUT <plot> <crop>   e.g. 'SCOUT P1 WHEAT' — reveals the "
            "exact yield for that pair.\n"
            "  PLANT P1=<crop>,P2=<crop>,P3=<crop>  — commits your final "
            "assignment and ends the episode (each crop must appear "
            "exactly once).\n"
            f"You have {self.step_limit} total actions (scouting plus "
            "planting combined) — not enough to scout every one of the 9 "
            "pairs, so choose what to scout carefully. PLANT ends the "
            "episode immediately, so submit it only once you're ready."
        )
        return obs, {}

    def step(self, action):
        if self.done:
            return "Episode already ended.", 0.0, True, False, {}

        self.steps += 1
        text = re.sub(r"\s+", " ", str(action).strip().upper())
        reward = 0.0
        terminated = False

        scout_match = re.match(r"^SCOUT (P[123]) (WHEAT|CORN|SOY)$", text)
        if scout_match:
            plot, crop = scout_match.group(1), scout_match.group(2)
            value = self.yields[plot][crop]
            self.scouted.add((plot, crop))
            remaining = self.step_limit - self.steps
            obs = (
                f"Scouted {plot}/{crop}: yield = {value}. "
                f"({len(self.scouted)} of 9 pairs known; "
                f"{remaining} actions remaining.)"
            )
        elif text.startswith("PLANT"):
            rest = text[len("PLANT"):].strip()
            raw_parts = [p.replace(" ", "") for p in rest.split(",")]
            assignment = {}
            valid_format = len(raw_parts) == 3
            if valid_format:
                for part in raw_parts:
                    pm = re.match(r"^(P[123])=(WHEAT|CORN|SOY)$", part)
                    if not pm:
                        valid_format = False
                        break
                    assignment[pm.group(1)] = pm.group(2)
            is_bijection = (
                valid_format
                and set(assignment.keys()) == set(self.plots)
                and sorted(assignment.values()) == sorted(self.crops)
            )
            if not is_bijection:
                obs = (
                    "Invalid PLANT command. You must assign all three "
                    "plots, each to a different crop, in the form "
                    "'PLANT P1=<crop>,P2=<crop>,P3=<crop>'."
                )
            else:
                total = sum(self.yields[p][assignment[p]] for p in self.plots)
                ratio = total / self.optimal_total
                reward = round(min(0.2 + 0.8 * ratio, 1.0), 4)
                terminated = True
                self.done = True
                obs = (
                    f"Planted: {assignment}. Total yield = {total} "
                    f"(best possible = {self.optimal_total}). "
                    f"Reward = {reward:.3f}."
                )
        else:
            obs = (
                "Unrecognized action. Use 'SCOUT <plot> <crop>' or "
                "'PLANT P1=<crop>,P2=<crop>,P3=<crop>'."
            )

        truncated = False
        if not terminated and self.steps >= self.step_limit:
            truncated = True
            self.done = True
            obs += " Step limit reached without a valid PLANT — episode ends with no yield."

        return obs, reward, terminated, truncated, {}
