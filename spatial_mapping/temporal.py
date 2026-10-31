"""Bounded temporal alignment and scheduling, independent of ROS."""
from bisect import bisect_left
from collections import deque
from threading import Lock
import numpy as np


class PoseHistory:
    """Interpolate coherent poses in one clock domain; never extrapolate."""
    def __init__(self, capacity=1024, max_gap_s=.15):
        self.samples = deque(maxlen=capacity)
        self.max_gap_s = float(max_gap_s)
        self.lock = Lock()

    def add(self, stamp_s, position, quaternion):
        p, q = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
        if (not np.isfinite(stamp_s) or stamp_s <= 0 or p.shape != (3,)
                or q.shape != (4,) or not np.isfinite(p).all()
                or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9):
            return False
        with self.lock:
            if self.samples and stamp_s < self.samples[-1][0]:
                self.samples.clear()
            if self.samples and stamp_s == self.samples[-1][0]:
                self.samples.pop()
            self.samples.append((float(stamp_s), p.copy(), q / np.linalg.norm(q)))
        return True

    def at(self, stamp_s):
        with self.lock:
            samples = list(self.samples)
        if not samples or not np.isfinite(stamp_s):
            return None
        index = bisect_left([s[0] for s in samples], stamp_s)
        if index < len(samples) and abs(samples[index][0] - stamp_s) < 1e-9:
            return samples[index][1].copy(), samples[index][2].copy()
        if index == 0 or index == len(samples):
            return None
        a, b = samples[index - 1], samples[index]
        if b[0] - a[0] > self.max_gap_s:
            return None
        f = (stamp_s - a[0]) / (b[0] - a[0])
        qa, qb = a[2], b[2]
        dot = float(np.dot(qa, qb))
        if dot < 0:
            qb, dot = -qb, -dot
        dot = np.clip(dot, -1, 1)
        if dot > .9995:
            q = qa + f * (qb - qa)
        else:
            theta = np.arccos(dot)
            q = (np.sin((1 - f) * theta) * qa + np.sin(f * theta) * qb) / np.sin(theta)
        return a[1] + f * (b[1] - a[1]), q / np.linalg.norm(q)


class LatestObservation:
    """Capacity-one mailbox: consume each strictly increasing source ID once."""
    def __init__(self):
        self.lock = Lock()
        self.pending = None
        self.last_submitted = None
        self.replaced = 0
        self.duplicates = 0

    def submit(self, source_id, payload):
        with self.lock:
            if self.last_submitted is not None and source_id <= self.last_submitted:
                self.duplicates += 1
                return False
            self.last_submitted = source_id
            self.replaced += self.pending is not None
            self.pending = source_id, payload
            return True

    def take(self):
        with self.lock:
            item, self.pending = self.pending, None
            return item
