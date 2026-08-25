import random
import re
import itertools


class CourierRouteEnv:
    """Courier must deliver to 5 stops, choosing an order that minimizes total travel."""

    LABELS = ('A', 'B', 'C', 'D', 'E')
    STEP_LIMIT = 10
    ACTION_RE = re.compile(r'^\s*GOTO\s+([A-Za-z])\s*$', re.IGNORECASE)

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        cells = [(x, y) for x in range(1, 10) for y in range(1, 10)]
        chosen = self.rng.sample(cells, len(self.LABELS))
        self.positions = dict(zip(self.LABELS, chosen))
        self.depot = (0, 0)
        self.current_pos = self.depot
        self.current_label = 'DEPOT'
        self.visited = set()
        self.steps = 0
        self.total_distance = 0
        self.done = False
        self.truncated_flag = False
        self.optimal_distance = self._brute_force_optimal()

        dist_lines = ', '.join(
            '%s=%d' % (lbl, self._dist(self.depot, self.positions[lbl]))
            for lbl in self.LABELS
        )
        obs = (
            "You are a courier at the DEPOT. Deliver packages to all 5 stops "
            "(%s) choosing an order that minimizes total travel distance "
            "(Manhattan distance between consecutive locations).\n"
            "Action format: 'GOTO <letter>' e.g. 'GOTO C'. You have %d steps total.\n"
            "Distances from DEPOT to each stop: %s\n"
            "Once you arrive at a stop, you will learn the distances from that stop "
            "to the still-undelivered stops. Plan your route to minimize the total "
            "distance travelled across all 5 deliveries."
            % (', '.join(self.LABELS), self.STEP_LIMIT, dist_lines)
        )
        info = {'optimal_distance': self.optimal_distance}
        return obs, info

    def _dist(self, p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    def _brute_force_optimal(self):
        best = None
        for perm in itertools.permutations(self.LABELS):
            total = 0
            prev = self.depot
            for lbl in perm:
                total += self._dist(prev, self.positions[lbl])
                prev = self.positions[lbl]
            if best is None or total < best:
                best = total
        return best

    def step(self, action):
        if self.done or self.truncated_flag:
            return "Episode already finished.", 0.0, self.done, self.truncated_flag, {}

        self.steps += 1
        match = self.ACTION_RE.match(action or '')

        if not match:
            obs = ("Malformed action. Use exactly: 'GOTO <letter>' where <letter> "
                   "is one of %s." % ', '.join(self.LABELS))
            return self._finish_step(obs, 0.0)

        label = match.group(1).upper()

        if label not in self.LABELS:
            obs = "Unknown stop '%s'. Valid stops are: %s." % (label, ', '.join(self.LABELS))
            return self._finish_step(obs, 0.0)

        if label in self.visited:
            obs = "Stop %s has already been delivered. Choose an undelivered stop." % label
            return self._finish_step(obs, 0.0)

        dest = self.positions[label]
        leg = self._dist(self.current_pos, dest)
        self.total_distance += leg
        self.current_pos = dest
        self.current_label = label
        self.visited.add(label)
        reward = 0.5 / len(self.LABELS)

        remaining = [l for l in self.LABELS if l not in self.visited]

        if not remaining:
            self.done = True
            excess_ratio = (self.total_distance - self.optimal_distance) / self.optimal_distance
            bonus = 0.5 * max(0.0, 1.0 - excess_ratio)
            reward += bonus
            obs = (
                "Arrived at %s (leg distance %d). All deliveries complete! "
                "Total distance travelled: %d (optimal was %d)."
                % (label, leg, self.total_distance, self.optimal_distance)
            )
            return obs, reward, True, False, {
                'total_distance': self.total_distance,
                'optimal_distance': self.optimal_distance,
            }

        dist_lines = ', '.join(
            '%s=%d' % (l, self._dist(dest, self.positions[l])) for l in remaining
        )
        obs = (
            "Arrived at %s (leg distance %d, running total %d). Remaining stops: %s.\n"
            "Distances from %s to remaining stops: %s"
            % (label, leg, self.total_distance, ', '.join(remaining), label, dist_lines)
        )
        return self._finish_step(obs, reward)

    def _finish_step(self, obs, reward):
        if self.steps >= self.STEP_LIMIT and not self.done:
            self.truncated_flag = True
            obs = obs + (" Step limit (%d) reached -- deliveries incomplete." % self.STEP_LIMIT)
            return obs, reward, False, True, {
                'total_distance': self.total_distance,
                'delivered': sorted(self.visited),
            }
        return obs, reward, False, False, {
            'total_distance': self.total_distance,
            'delivered': sorted(self.visited),
        }
