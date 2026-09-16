"""Anytime backward grid search with explicit scene-snapshot replanning.

AD* consistency updates refine an inflated-heuristic solution to epsilon=1.
Scene changes trigger a fresh graph search; obstacle motion is not predicted.
"""

import heapq
import threading
import numpy as np

from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.planners.clearance import ClearanceScene
from smallsat_sim.planners.validation import positive_finite


class NoPathError(RuntimeError):
    """No collision-free reference is available; callers must stop the vehicle."""


class ADStarPlanner(BasePlanner):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.env_cfg.planner
        self.resolution = float(cfg.resolution)
        positive_finite("resolution", self.resolution)
        self.initial_epsilon = float(cfg.epsilon)
        if not np.isfinite(self.initial_epsilon) or self.initial_epsilon < 1:
            raise ValueError("epsilon must be finite and at least one")
        self.epsilon_decrement = float(cfg.epsilon_decrement)
        positive_finite("epsilon_decrement", self.epsilon_decrement)
        self.bounds = np.asarray(cfg.bounds, dtype=float)
        if self.bounds.shape != (2, 3) or not np.all(np.isfinite(self.bounds)) or np.any(self.bounds[0] > self.bounds[1]):
            raise ValueError("bounds must be finite ordered 3D corners")
        self.clearance = float(getattr(cfg, "clearance", 0.02))
        positive_finite("clearance", self.clearance)
        self.directions = tuple(tuple(np.asarray(d, dtype=float) * self.resolution)
                                for d in cfg.unit_directions if np.linalg.norm(d) > 0)
        if not self.directions:
            raise ValueError("planner requires nonzero neighbor directions")
        if any(tuple(-np.asarray(d)) not in self.directions for d in self.directions):
            raise ValueError("AD* requires symmetric grid directions")
        self.start = self._endpoint(cfg.start_pos)
        self.goal = self._endpoint(cfg.goal_pos)
        self.path = []
        self.idx_reference_point = 0
        self._stop = threading.Event()
        self.planning_thread = None
        self._error = None
        self.replan()
        if not hasattr(env, "_managed_planners"):
            env._managed_planners = []
        env._managed_planners.append(self)

    def _endpoint(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value < self.bounds[0]) or np.any(value > self.bounds[1]):
            raise ValueError("Planner endpoints must lie inside bounds")
        if not np.allclose(value / self.resolution, np.round(value / self.resolution)):
            raise ValueError("Planner endpoints must lie on the resolution grid")
        return tuple(value)

    def _key(self, state):
        g, rhs = self.g.get(state, np.inf), self.rhs.get(state, np.inf)
        heuristic = np.linalg.norm(np.asarray(state) - self.start)
        return (rhs + self.epsilon * heuristic, rhs) if g > rhs else (g + heuristic, g)

    def _push(self, state):
        key = self._key(state)
        self._open_keys[state] = key
        heapq.heappush(self.OPEN, (key, state))

    def _peek(self):
        while self.OPEN and self._open_keys.get(self.OPEN[0][1]) != self.OPEN[0][0]:
            heapq.heappop(self.OPEN)
        return self.OPEN[0][0] if self.OPEN else (np.inf, np.inf)

    def _get_neighbors(self, state):
        result = []
        for direction in self.directions:
            point = np.asarray(state) + direction
            point = np.round(point / self.resolution) * self.resolution
            if np.any(point < self.bounds[0]) or np.any(point > self.bounds[1]):
                continue
            neighbor = tuple(point)
            edge = tuple(sorted((state, neighbor)))
            if edge not in self._edges:
                self._edges[edge] = self.scene.segment_is_clear(state, neighbor, clearance=self.clearance)
            if self._edges[edge]:
                result.append(neighbor)
        return result

    def _update_state(self, state):
        if state != self.goal:
            self.rhs[state] = min((np.linalg.norm(np.asarray(state) - other) + self.g.get(other, np.inf)
                                   for other in self._get_neighbors(state)), default=np.inf)
        self._open_keys.pop(state, None)
        if self.g.get(state, np.inf) != self.rhs.get(state, np.inf):
            if state in self.CLOSED:
                self.INCONS.add(state)
            else:
                self._push(state)

    def compute_shortest_path(self):
        while (self._peek() < self._key(self.start)
               or self.rhs.get(self.start, np.inf) != self.g.get(self.start, np.inf)):
            if self._stop.is_set():
                raise NoPathError("Planning cancelled")
            if not self.OPEN:
                break
            _, state = heapq.heappop(self.OPEN)
            self._open_keys.pop(state, None)
            if self.g.get(state, np.inf) > self.rhs.get(state, np.inf):
                self.g[state] = self.rhs[state]
                self.CLOSED.add(state)
            else:
                self.g[state] = np.inf
                self._update_state(state)
            for neighbor in self._get_neighbors(state):
                self._update_state(neighbor)

    def generate_path(self):
        if not np.isfinite(self.g.get(self.start, np.inf)):
            raise NoPathError("No collision-free path to goal")
        path, visited = [self.start], {self.start}
        while path[-1] != self.goal:
            state = path[-1]
            candidates = [n for n in self._get_neighbors(state) if n not in visited
                          and np.isfinite(self.g.get(n, np.inf))]
            if not candidates:
                raise NoPathError("Path reconstruction failed")
            next_state = min(candidates, key=lambda n: np.linalg.norm(np.asarray(state) - n) + self.g[n])
            path.append(next_state)
            visited.add(next_state)
        return path

    def replan(self, start=None, goal=None):
        """Snapshot current obstacles and recompute; never retain a stale failed path."""
        self.stop_planning_thread()
        if threading.current_thread() is not self.planning_thread:
            self._stop.clear()
        if start is not None:
            self.start = self._endpoint(start)
        if goal is not None:
            self.goal = self._endpoint(goal)
        self.path, self._error = [], None
        self.scene = ClearanceScene(self.env)
        self.quaternion = self.scene.quaternion.copy()
        self._edges = {}
        self.epsilon = self.initial_epsilon
        self.g, self.rhs = {}, {self.goal: 0.}
        self.OPEN, self._open_keys = [], {}
        self.CLOSED, self.INCONS = set(), set()
        if any(self.scene.distance(p) <= self.clearance for p in (self.start, self.goal)):
            raise NoPathError("Start or goal violates spacecraft clearance")
        self._push(self.goal)
        while True:
            self.compute_shortest_path()
            path = self.generate_path()
            if self.epsilon == 1:
                break
            self.epsilon = max(1., self.epsilon - self.epsilon_decrement)
            states = set(self._open_keys) | self.INCONS
            self.OPEN, self._open_keys = [], {}
            self.CLOSED, self.INCONS = set(), set()
            for state in states:
                self._push(state)
        self.path = path
        self.idx_reference_point = min(1, len(path) - 1)
        self.visualize([np.asarray(p) for p in path])
        return path

    def start_planning_thread(self):
        self.stop_planning_thread()
        self._stop.clear()
        def run():
            try:
                self.replan()
            except Exception as error:
                self.path, self._error = [], error
        self.planning_thread = threading.Thread(target=run, daemon=True)
        self.planning_thread.start()

    def stop_planning_thread(self):
        worker = self.planning_thread
        if worker is not None and worker is not threading.current_thread():
            self._stop.set()
            worker.join()
            self.planning_thread = None

    close = stop_planning_thread

    def reset(self):
        return self.replan()

    def get_reference(self, obs):
        if not self.path:
            raise NoPathError("No completed path is available") from self._error
        position = np.asarray(obs[:3])
        index = self.idx_reference_point
        if np.linalg.norm(position - self.path[index]) < min(.2, self.resolution / 2):
            index = min(index + 1, len(self.path) - 1)
        target = np.asarray(self.path[index])
        # Recheck actual pose-to-reference against current obstacles, including
        # the vehicle's orientation, before returning a commandable waypoint.
        scene = ClearanceScene(self.env)
        if not scene.segment_is_clear(position, target, clearance=self.clearance,
                                      end_quaternion=self.quaternion):
            self.path = []
            raise NoPathError("Current route is obstructed; stop and replan")
        self.idx_reference_point = index
        return target.reshape(3, 1), self.quaternion.reshape(4, 1)

    def closest_point_on_trajectory(self, point):
        if not self.path:
            raise NoPathError("No completed path is available")
        point = np.asarray(point)
        closest, best, arc, accumulated = np.asarray(self.path[0]), np.inf, 0., 0.
        for a, b in zip(self.path[:-1], self.path[1:]):
            a, b = np.asarray(a), np.asarray(b)
            length = np.linalg.norm(b - a)
            t = np.clip(np.dot(point - a, b - a) / length**2, 0., 1.)
            candidate = a + t * (b - a)
            distance = np.linalg.norm(candidate - point)
            if distance < best:
                closest, best, arc = candidate, distance, accumulated + t * length
            accumulated += length
        return closest, arc
