"""Own cameras, overlays, browser/native viewing and offscreen recording.

Call all methods from the same thread. RL uses its dedicated rendering worker;
classical simulation calls directly from its simulation thread.
"""
import mujoco
import mujoco.viewer
import numpy as np

from .browser import BrowserViewer
from .video import VideoRecorder


class SceneOutput:
    def __init__(self, model, data, *, width, height, browser=False, port=8080,
                 native=False, key_callback=None, distance=4., azimuth=-65., elevation=-35.):
        self.model, self.data = model, data
        self.width, self.height = width, height
        self.renderer = self.browser = self.native = None
        self.recorder = VideoRecorder()
        self.overlays = {}
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance, self.camera.azimuth = distance, azimuth
        self.camera.elevation = elevation
        self.camera.trackbodyid = int(model.jnt_bodyid[0]) if model.njnt else 0
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        try:
            if browser:
                self.browser = BrowserViewer(model, data, port=port)
            if native:
                self.native = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
                for field in ("distance", "azimuth", "elevation", "trackbodyid", "type"):
                    setattr(self.native.cam, field, getattr(self.camera, field))
        except BaseException:
            self.close()
            raise

    def set_overlay(self, name, points, *, color=(1, 0, 0, 1), radius=.05, capsules=()):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        rgba = np.clip(np.asarray(color, dtype=float), 0, 1)
        capsules = tuple(
            (np.array(start, dtype=float, copy=True), np.array(end, dtype=float, copy=True),
             float(capsule_radius), np.clip(np.asarray(capsule_color, dtype=float), 0, 1))
            for start, end, capsule_radius, capsule_color in capsules
        )
        previous = self.overlays.get(name)
        if previous is not None:
            old_points, old_color, old_radius, old_capsules = previous
            same_capsules = len(old_capsules) == len(capsules) and all(
                all(np.array_equal(a, b) for a, b in zip(old, new, strict=True))
                for old, new in zip(old_capsules, capsules, strict=True)
            )
            if (np.array_equal(old_points, points) and np.array_equal(old_color, rgba)
                    and old_radius == radius and same_capsules):
                return
        self.overlays[name] = points.copy(), rgba, radius, capsules
        if self.browser is not None:
            self.browser.set_overlay(name, points, rgba, radius, capsules)

    def _draw_overlays(self, scene):
        for points, color, radius, capsules in self.overlays.values():
            for point in points:
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                   size=[radius, 0, 0], pos=point, mat=np.eye(3).ravel(), rgba=color)
                scene.ngeom += 1
            for start, end, radius, color in capsules:
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                   size=np.zeros(3), pos=np.zeros(3), mat=np.eye(3).ravel(), rgba=color)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
                scene.ngeom += 1

    def update_viewer(self):
        if self.browser is not None:
            self.browser.update(self.data)
        if self.native is not None:
            with self.native.lock():
                self.native.user_scn.ngeom = 0
                self._draw_overlays(self.native.user_scn)
            self.native.sync()

    def start_video(self, *, fps, path=None):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.recorder.start(width=self.width, height=self.height, fps=fps, path=path)

    def record_frame(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        if self.overlays:
            self._draw_overlays(self.renderer.scene)
        self.recorder.write(self.renderer.render())

    def close(self):
        try:
            self.recorder.close()
        finally:
            try:
                if self.renderer is not None:
                    self.renderer.close()
            finally:
                self.renderer = None
                try:
                    if self.browser is not None:
                        self.browser.close()
                finally:
                    self.browser = None
                    if self.native is not None:
                        self.native.close()
                        self.native = None
