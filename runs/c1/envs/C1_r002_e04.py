import re


class GroovePeriodEnv:
    VOICES = ('K', 'S', 'H')
    PERIOD_CHOICES = (5, 6, 7, 8)
    BAR_LEN = 8
    TARGET_BAR = 3
    MAX_STEPS = 10
    PROBE_MIN = 8
    PROBE_MAX = 39

    PROBE_RE = re.compile(r'^PROBE\s+([KSHksh])\s+(\d+)$')
    SUBMIT_RE = re.compile(
        r'^SUBMIT\s+K:([Xx.]{8})\s+S:([Xx.]{8})\s+H:([Xx.]{8})$'
    )

    def __init__(self):
        self.rng = None
        self.periods = {}
        self.phases = {}
        self.steps_used = 0
        self.done = False

    def _hit(self, voice, step):
        return (step % self.periods[voice]) == self.phases[voice]

    def _bar_pattern(self, voice, bar_index):
        start = bar_index * self.BAR_LEN
        return ''.join(
            'X' if self._hit(voice, start + i) else '.'
            for i in range(self.BAR_LEN)
        )

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.periods = {v: self.rng.choice(self.PERIOD_CHOICES) for v in self.VOICES}
        self.phases = {v: self.rng.randrange(self.periods[v]) for v in self.VOICES}
        self.steps_used = 0
        self.done = False

        bar0 = {v: self._bar_pattern(v, 0) for v in self.VOICES}
        t_start = self.TARGET_BAR * self.BAR_LEN
        t_end = t_start + self.BAR_LEN - 1

        lines = [
            "Three drum voices loop forever: Kick (K), Snare (S), Hi-Hat (H).",
            "Each voice hits at global step s exactly when (s mod period) == phase, "
            "for that voice's own hidden fixed period and phase (never shown directly).",
            "Bar 0 (steps 0-7) is notated below for all three voices ('X'=hit, '.'=rest):",
            f"K: {bar0['K']}",
            f"S: {bar0['S']}",
            f"H: {bar0['H']}",
            f"Goal: predict bar {self.TARGET_BAR} (steps {t_start}-{t_end}) exactly, for all three voices.",
            f"Action 'PROBE <K|S|H> <step>' reveals hit/no-hit for one voice at one step "
            f"between {self.PROBE_MIN} and {self.PROBE_MAX} (steps 0-7 are already shown above).",
            "Action 'SUBMIT K:xxxxxxxx S:xxxxxxxx H:xxxxxxxx' (8 chars of X/. per voice, in step "
            f"order for bar {self.TARGET_BAR}) grades your prediction and ends the episode.",
            f"You have {self.MAX_STEPS} actions total (probes and the submission both count). "
            "Reward is 1/3 per voice predicted exactly right in the target bar.",
        ]
        return "\n".join(lines), {}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps_used += 1
        remaining = self.MAX_STEPS - self.steps_used
        text = (action or "").strip()

        m_submit = self.SUBMIT_RE.match(text)
        if m_submit:
            pred = {
                'K': m_submit.group(1).upper(),
                'S': m_submit.group(2).upper(),
                'H': m_submit.group(3).upper(),
            }
            actual = {v: self._bar_pattern(v, self.TARGET_BAR) for v in self.VOICES}
            correct_count = sum(1 for v in self.VOICES if pred[v] == actual[v])
            reward = correct_count / 3.0
            self.done = True
            report = [
                f"{v}: {'MATCH' if pred[v] == actual[v] else 'MISS'} "
                f"(yours={pred[v]} actual={actual[v]})"
                for v in self.VOICES
            ]
            obs = "Submission graded.\n" + "\n".join(report) + f"\nTotal reward: {reward:.3f}"
            return obs, reward, True, False, {}

        m_probe = self.PROBE_RE.match(text)
        if m_probe:
            voice = m_probe.group(1).upper()
            pos = int(m_probe.group(2))
            if pos < self.PROBE_MIN or pos > self.PROBE_MAX:
                obs = (
                    f"Invalid probe step {pos}: steps 0-7 are already shown above; "
                    f"choose a step between {self.PROBE_MIN} and {self.PROBE_MAX}. "
                    f"{max(remaining, 0)} actions left."
                )
                return obs, 0.0, False, remaining <= 0, {}
            hit = self._hit(voice, pos)
            obs = (
                f"Probe {voice} at step {pos}: {'HIT' if hit else 'no hit'}. "
                f"{max(remaining, 0)} actions left."
            )
            return obs, 0.0, False, remaining <= 0, {}

        obs = (
            "Could not parse that action. Use 'PROBE <K|S|H> <step>' with step in "
            f"[{self.PROBE_MIN},{self.PROBE_MAX}], or 'SUBMIT K:xxxxxxxx S:xxxxxxxx H:xxxxxxxx' "
            f"(8 chars of X/. per voice). {max(remaining, 0)} actions left."
        )
        return obs, 0.0, False, remaining <= 0, {}
