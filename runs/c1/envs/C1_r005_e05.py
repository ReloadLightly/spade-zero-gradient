import random
import re


class RowReachSprinklerEnv:
    """Water every garden plot on a line using as few hose repositions as possible.

    The hose's reach R is a hidden constant integer for the episode. Each
    MOVE both explores (reveals which plots newly get wet) and permanently
    advances progress (already-wet plots stay wet), so the solver must plan
    reach-discovery and final coverage together within a shared step budget.
    """

    MIN_COORD = 0
    MAX_COORD = 30
    NUM_PLOTS = 6
    MAX_STEPS = 10
    R_MIN = 3
    R_MAX = 6

    ACTION_RE = re.compile(r"^\s*MOVE\s+(-?\d+)\s*$", re.IGNORECASE)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.plots = sorted(
            self.rng.sample(range(self.MIN_COORD, self.MAX_COORD + 1), self.NUM_PLOTS)
        )
        self.reach = self.rng.randint(self.R_MIN, self.R_MAX)
        self.watered = set()
        self.step_count = 0
        self.reposition_count = 0
        self.optimal_count = self._compute_optimal(self.plots, self.reach)
        self.done = False

        obs = (
            f"GARDEN ROW: plots sit at positions {self.plots} on a line from "
            f"{self.MIN_COORD} to {self.MAX_COORD}.\n"
            f"Wherever you place the hose, it waters every plot within its "
            f"reach R (an integer between {self.R_MIN} and {self.R_MAX}, fixed "
            f"but unknown to you) of that position. Watered plots stay watered.\n"
            f"ACTION FORMAT: 'MOVE <integer position>' repositions the hose "
            f"and sprays. You have {self.MAX_STEPS} total actions to water "
            f"every plot, using as few repositions as possible.\n"
            f"Watered so far: none. Unwatered: {self.plots}."
        )
        info = {"plots": list(self.plots), "max_steps": self.MAX_STEPS}
        return obs, info

    def step(self, action: str):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.step_count += 1
        match = self.ACTION_RE.match(action or "")

        if not match:
            terminated, truncated = False, self.step_count >= self.MAX_STEPS
            self.done = truncated
            obs = (
                "Malformed action. Use exactly: MOVE <integer position>, e.g. "
                f"a whole number between {self.MIN_COORD} and {self.MAX_COORD}. "
                f"Steps used: {self.step_count}/{self.MAX_STEPS}."
            )
            return obs, 0.0, terminated, truncated, {"invalid": True}

        pos = int(match.group(1))
        if pos < self.MIN_COORD or pos > self.MAX_COORD:
            terminated, truncated = False, self.step_count >= self.MAX_STEPS
            self.done = truncated
            obs = (
                f"Position {pos} is out of bounds "
                f"[{self.MIN_COORD}, {self.MAX_COORD}]. Try again. "
                f"Steps used: {self.step_count}/{self.MAX_STEPS}."
            )
            return obs, 0.0, terminated, truncated, {"invalid": True}

        self.reposition_count += 1
        newly = []
        for i, plot in enumerate(self.plots):
            if i not in self.watered and abs(plot - pos) <= self.reach:
                self.watered.add(i)
                newly.append(plot)

        progress_reward = 0.4 * (len(newly) / self.NUM_PLOTS)
        unwatered = [p for i, p in enumerate(self.plots) if i not in self.watered]

        if len(self.watered) == self.NUM_PLOTS:
            completion_reward = 0.3
            if self.reposition_count <= self.optimal_count:
                efficiency_reward = 0.3
            else:
                efficiency_reward = max(
                    0.0, 0.3 - 0.1 * (self.reposition_count - self.optimal_count)
                )
            reward = progress_reward + completion_reward + efficiency_reward
            self.done = True
            obs = (
                f"MOVE to {pos}: newly watered {newly if newly else 'none'}. "
                f"All {self.NUM_PLOTS} plots are now watered in "
                f"{self.reposition_count} repositions. Garden complete!"
            )
            return obs, reward, True, False, {"reposition_count": self.reposition_count}

        truncated = self.step_count >= self.MAX_STEPS
        self.done = truncated
        obs = (
            f"MOVE to {pos}: newly watered {newly if newly else 'none'}. "
            f"Watered {len(self.watered)}/{self.NUM_PLOTS}. Unwatered: {unwatered}. "
            f"Repositions used: {self.reposition_count}. "
            f"Steps used: {self.step_count}/{self.MAX_STEPS}."
        )
        if truncated:
            obs += " Step budget exhausted; garden left unfinished."
        return obs, progress_reward, False, truncated, {}

    @staticmethod
    def _compute_optimal(plots, reach):
        n = len(plots)
        count = 0
        i = 0
        span = 2 * reach
        while i < n:
            limit = plots[i] + span
            count += 1
            j = i
            while j < n and plots[j] <= limit:
                j += 1
            i = j
        return count
