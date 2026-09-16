"""Conservative swept-pose checks against a snapshot of MuJoCo collision geometry."""

import mujoco
import numpy as np


class ClearanceScene:
    """Query actual vehicle geoms without modifying the running simulation.

    Checks use MuJoCo's collision representation (including convex mesh hulls),
    not rendered meshes. Obstacles are stationary within this snapshot.
    """

    def __init__(self, env, body_name="body0"):
        self.model = env.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = env.data.qpos
        self.data.mocap_pos[:] = env.data.mocap_pos
        self.data.mocap_quat[:] = env.data.mocap_quat
        self.body = int(self.model.body(body_name).id)
        joint = int(self.model.body_jntadr[self.body])
        if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Clearance planning requires a free-joint spacecraft")
        self.qadr = int(self.model.jnt_qposadr[joint])
        self.quaternion = self.data.qpos[self.qadr + 3:self.qadr + 7].copy()
        descendants = {self.body}
        for body in range(self.body + 1, self.model.nbody):
            if int(self.model.body_parentid[body]) in descendants:
                descendants.add(body)
        self.vehicle_geoms = [i for i in range(self.model.ngeom)
                              if int(self.model.geom_bodyid[i]) in descendants]
        others = [i for i in range(self.model.ngeom) if i not in self.vehicle_geoms]
        self.pairs = [(a, b) for a in self.vehicle_geoms for b in others
                      if (self.model.geom_contype[a] & self.model.geom_conaffinity[b])
                      or (self.model.geom_contype[b] & self.model.geom_conaffinity[a])]
        mujoco.mj_forward(self.model, self.data)
        self.radius = max((np.linalg.norm(self.data.geom_xpos[i] - self.data.xpos[self.body])
                           + self.model.geom_rbound[i] for i in self.vehicle_geoms), default=0.)

    def distance(self, position, quaternion=None, limit=1e3):
        self.data.qpos[self.qadr:self.qadr + 3] = position
        self.data.qpos[self.qadr + 3:self.qadr + 7] = (
            self.quaternion if quaternion is None else quaternion)
        mujoco.mj_forward(self.model, self.data)
        return min((mujoco.mj_geomDistance(self.model, self.data, a, b, limit, None)
                    for a, b in self.pairs), default=float(limit))

    def segment_is_clear(self, start, end, *, clearance=0.02, quaternion=None,
                         end_quaternion=None, tolerance=1e-4):
        """Certify the entire translation/shortest rotation; reject unresolved gaps.

        Surface distance is Lipschitz in rigid-body displacement. A midpoint
        clearance larger than the maximum displacement to either endpoint proves
        that interval clear; otherwise subdivide. This catches diagonal edges and
        thin obstacles without relying on a fixed sampling interval.
        """
        q0 = np.asarray(self.quaternion if quaternion is None else quaternion, dtype=float)
        q1 = np.asarray(q0 if end_quaternion is None else end_quaternion, dtype=float)
        if (not np.all(np.isfinite(np.r_[start, end, q0, q1, clearance, tolerance]))
                or np.linalg.norm(q0) < 1e-12 or np.linalg.norm(q1) < 1e-12
                or clearance < 0 or tolerance <= 0):
            raise ValueError("Swept poses and clearance must be finite and valid")
        q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        pending = [(np.asarray(start), np.asarray(end), q0, q1)]
        while pending:
            a, b, qa, qb = pending.pop()
            angle = 2 * np.arccos(np.clip(np.dot(qa, qb), -1., 1.))
            displacement = np.linalg.norm(b - a) / 2 + self.radius * angle / 2
            mid, qm = (a + b) / 2, qa + qb
            qm /= np.linalg.norm(qm)
            distance = self.distance(mid, qm, limit=clearance + displacement + 1.)
            if distance > clearance + displacement:
                continue
            if distance <= clearance or displacement <= tolerance:
                return False
            pending.extend(((a, mid, qa, qm), (mid, b, qm, qb)))
        return True
