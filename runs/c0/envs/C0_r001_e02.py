import random
import re
import string


class CargoManifestEnv:
    """Bin-packing under partial information: minimize bins used within a step budget."""

    N_BOXES = 5
    STEP_LIMIT = 10
    PACK_UNIT_REWARD = 0.1  # 0.1 * 5 boxes = 0.5
    SHIP_OPTIMAL_REWARD = 0.5
    SHIP_NEAR_REWARD = 0.2

    ACTION_RE = re.compile(r'^\s*(INSPECT|PACK|SHIP)\b\s*(.*)$', re.IGNORECASE)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.box_ids = list(string.ascii_uppercase[: self.N_BOXES])

        # Search for a weight/capacity draw whose optimal bin count is 2 or 3
        # (avoids trivial 1-bin or maximally-fragmented instances).
        for _ in range(200):
            capacity = self.rng.randint(30, 45)
            weights = [self.rng.randint(8, 32) for _ in range(self.N_BOXES)]
            optimal = self._optimal_bins(weights, capacity)
            if optimal is not None and 2 <= optimal <= 3:
                break
        self.capacity = capacity
        self.true_weights = dict(zip(self.box_ids, weights))
        self.optimal_bins = optimal

        self.ranges = {}
        for b in self.box_ids:
            w = self.true_weights[b]
            delta = self.rng.randint(4, 8)
            self.ranges[b] = (max(1, w - delta), w + delta)

        self.inspected = set()
        self.bins = {}  # bin_id (int) -> list of box letters
        self.box_bin = {}  # box letter -> bin_id
        self.steps = 0
        self.done = False

        header = (
            f"CARGO MANIFEST\n"
            f"Ship all {self.N_BOXES} boxes ({', '.join(self.box_ids)}) using as FEW bins as "
            f"possible. Each bin has capacity {self.capacity} kg. Exact box weights are secret; "
            f"you only know rough ranges below. You have {self.STEP_LIMIT} steps total.\n"
            f"Actions (exactly one per turn):\n"
            f"  INSPECT <box>       - reveal a box's exact weight (uses a step)\n"
            f"  PACK <box> <bin#>   - place a box into a numbered bin (bin# is any positive integer)\n"
            f"  SHIP                - finalize once all boxes are packed; ends the episode\n"
        )
        return header + self._status_text(), {"optimal_bins": self.optimal_bins}

    def step(self, action):
        if self.done:
            return "Episode already finished.", 0.0, True, False, {}

        self.steps += 1
        reward = 0.0
        terminated = False
        msg = ""

        m = self.ACTION_RE.match(action or "")
        if not m:
            msg = "Malformed action. Use 'INSPECT <box>', 'PACK <box> <bin#>', or 'SHIP'."
        else:
            verb = m.group(1).upper()
            rest = m.group(2).strip()

            if verb == "INSPECT":
                box = rest.strip().upper()
                if box not in self.box_ids:
                    msg = f"Unknown box '{rest}'. Valid boxes: {', '.join(self.box_ids)}."
                elif box in self.inspected:
                    msg = f"Box {box} is already inspected: exactly {self.true_weights[box]} kg."
                else:
                    self.inspected.add(box)
                    msg = f"Inspected box {box}: exactly {self.true_weights[box]} kg."

            elif verb == "PACK":
                parts = rest.split()
                if len(parts) != 2 or not parts[1].lstrip('-').isdigit() or int(parts[1]) <= 0:
                    msg = "Use 'PACK <box> <bin#>' with a positive integer bin number."
                else:
                    box = parts[0].strip().upper()
                    bin_id = int(parts[1])
                    if box not in self.box_ids:
                        msg = f"Unknown box '{parts[0]}'. Valid boxes: {', '.join(self.box_ids)}."
                    elif box in self.box_bin:
                        msg = f"Box {box} is already packed in Bin {self.box_bin[box]}."
                    else:
                        load = sum(self.true_weights[x] for x in self.bins.get(bin_id, []))
                        remaining = self.capacity - load
                        w = self.true_weights[box]
                        if w > remaining:
                            msg = (
                                f"Box {box} does NOT fit in Bin {bin_id}: it needs more than the "
                                f"{remaining} kg of space left there (load {load}/{self.capacity})."
                            )
                        else:
                            self.bins.setdefault(bin_id, []).append(box)
                            self.box_bin[box] = bin_id
                            reward = self.PACK_UNIT_REWARD
                            new_load = load + w
                            msg = f"Packed box {box} into Bin {bin_id} (load now {new_load}/{self.capacity})."

            elif verb == "SHIP":
                if len(self.box_bin) < self.N_BOXES:
                    missing = [b for b in self.box_ids if b not in self.box_bin]
                    msg = f"Cannot ship yet: unpacked boxes remain: {', '.join(missing)}."
                else:
                    bins_used = len([bid for bid, items in self.bins.items() if items])
                    terminated = True
                    self.done = True
                    if bins_used == self.optimal_bins:
                        reward = self.SHIP_OPTIMAL_REWARD
                        msg = f"Shipped using {bins_used} bins — optimal! Full ship bonus earned."
                    elif bins_used == self.optimal_bins + 1:
                        reward = self.SHIP_NEAR_REWARD
                        msg = (
                            f"Shipped using {bins_used} bins. A better packing existed "
                            f"({self.optimal_bins} bins was possible). Partial ship bonus earned."
                        )
                    else:
                        msg = (
                            f"Shipped using {bins_used} bins, well above the "
                            f"{self.optimal_bins}-bin optimum. No ship bonus earned."
                        )

        truncated = False
        if not terminated and self.steps >= self.STEP_LIMIT:
            truncated = True
            self.done = True
            msg += " Step limit reached — episode over."

        obs = msg + "\n" + self._status_text()
        return obs, reward, terminated, truncated, {"steps": self.steps}

    def _status_text(self):
        lines = [f"Steps used: {self.steps}/{self.STEP_LIMIT}"]
        if self.bins:
            for bid in sorted(self.bins):
                items = self.bins[bid]
                load = sum(self.true_weights[x] for x in items)
                lines.append(f"  Bin {bid}: [{', '.join(items) if items else '-'}] load {load}/{self.capacity}")
        else:
            lines.append("  No bins opened yet.")
        remaining = [b for b in self.box_ids if b not in self.box_bin]
        if remaining:
            parts = []
            for b in remaining:
                if b in self.inspected:
                    parts.append(f"{b}={self.true_weights[b]}kg")
                else:
                    lo, hi = self.ranges[b]
                    parts.append(f"{b}~{lo}-{hi}kg")
            lines.append("Unpacked: " + ", ".join(parts))
        else:
            lines.append("Unpacked: none — ready to SHIP.")
        return "\n".join(lines)

    @staticmethod
    def _optimal_bins(weights, capacity):
        n = len(weights)
        if any(w > capacity for w in weights):
            return None
        full = (1 << n) - 1
        feasible = [False] * (1 << n)
        for mask in range(1 << n):
            total = sum(weights[i] for i in range(n) if mask & (1 << i))
            feasible[mask] = total <= capacity
        dp = [n + 1] * (1 << n)
        dp[0] = 0
        for mask in range(1, 1 << n):
            sub = mask
            while sub > 0:
                if feasible[sub] and dp[mask ^ sub] + 1 < dp[mask]:
                    dp[mask] = dp[mask ^ sub] + 1
                sub = (sub - 1) & mask
        return dp[full]
