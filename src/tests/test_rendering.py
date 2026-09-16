"""Shared rendering owns output resources and leaves callers with only overlays."""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
import pytest

from smallsat_sim.envs.rendering.classical import ClassicalRendering
from smallsat_sim.envs.rendering.scene import SceneOutput
from smallsat_sim.envs.rendering.video import VideoRecorder
from smallsat_sim.envs.rendering import scene as scene_module


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string('''
        <mujoco><option timestep="0.01"/>
        <worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>
    ''')


@pytest.fixture
def fake_video(monkeypatch):
    writers = []

    class Writer:
        def __init__(self, path, codec, fps, size):
            Path(path).touch()
            self.count, self.released, self.fps = 0, False, fps
            writers.append(self)

        def isOpened(self):
            return True

        def write(self, frame):
            self.count += 1
            self.pixel = frame[0, 0].copy()

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, 'VideoWriter', Writer)
    return writers


def test_streaming_recorder_moves_clip_and_can_record_again(tmp_path, fake_video):
    recorder = VideoRecorder()
    recorder.start(width=16, height=16, fps=30)
    temporary = recorder.path
    frame = np.full((16, 16, 3), [255, 0, 0], dtype=np.uint8)
    for _ in range(2000):
        recorder.write(frame)
    destination = tmp_path / 'first.mp4'
    assert recorder.finish(destination) == destination
    assert destination.exists() and not temporary.exists()
    assert fake_video[0].count == 2000 and fake_video[0].released
    np.testing.assert_array_equal(fake_video[0].pixel, [0, 0, 255])
    recorder.start(width=16, height=16, fps=30, path=tmp_path / 'second.mp4')
    recorder.write(frame)
    recorder.close()
    recorder.close()
    assert fake_video[1].count == 1 and fake_video[1].released


def test_failed_writer_is_released_and_temporary_removed(monkeypatch):
    class FailedWriter:
        def __init__(self, path, *args):
            self.path = Path(path)
        def isOpened(self):
            return False
        def release(self):
            self.released = True
    writers = []
    def create(*args):
        writer = FailedWriter(*args)
        writers.append(writer)
        return writer
    monkeypatch.setattr(cv2, 'VideoWriter', create)
    recorder = VideoRecorder()
    with pytest.raises(RuntimeError, match='Cannot open video writer'):
        recorder.start(width=16, height=16, fps=30)
    assert writers[0].released
    assert not writers[0].path.parent.exists()
    recorder.close()


def make_env(model, *, video=False, browser=False):
    env = ClassicalRendering()
    env.model, env.data = model, mujoco.MjData(model)
    env._key_callback = lambda key: None
    env.args = Namespace(headless=not browser, video=video)
    env.env_cfg = SimpleNamespace(
        control=SimpleNamespace(control_decimation=1),
        viewer=SimpleNamespace(use_viser=browser, viser_port=8080),
        renderer=SimpleNamespace(width=16, height=16, fps=30,
                                 start_recording=0., end_recording=10.),
    )
    env._setup_rendering(env.args)
    return env


def test_classical_browser_uses_shared_scene_without_opengl(model, monkeypatch):
    browsers = []
    class Browser:
        def __init__(self, *args, **kwargs):
            self.updates = 0
            browsers.append(self)
        def update(self, data):
            self.updates += 1
        def set_overlay(self, *args):
            self.overlay = args
        def close(self):
            self.closed = True
    monkeypatch.setattr(scene_module, 'BrowserViewer', Browser)
    def forbidden(*args, **kwargs):
        raise AssertionError('Browser-only viewing must not allocate OpenGL')
    monkeypatch.setattr(mujoco, 'Renderer', forbidden)
    env = make_env(model, browser=True)
    env.visualization.set_overlay('reference', [[1, 2, 3]])
    env._update_viewer()
    env._update_renderer()
    assert browsers[0].updates == 1
    assert browsers[0].overlay[0] == 'reference'
    env.close()
    env.close()
    assert browsers[0].closed


def test_classical_sampling_does_not_drift_and_finalization_allows_next_clip(
        model, monkeypatch, tmp_path, fake_video):
    class Renderer:
        def __init__(self, *args, **kwargs):
            pass
        def update_scene(self, *args, **kwargs):
            pass
        def render(self):
            return np.zeros((16, 16, 3), dtype=np.uint8)
        def close(self):
            self.closed = True
    monkeypatch.setattr(mujoco, 'Renderer', Renderer)
    env = make_env(model, video=True)
    for timestamp in np.arange(0., 1., .01):
        env.data.time = timestamp
        env._update_renderer()
        env._update_renderer()  # A second caller at the same time must not duplicate a frame.
    assert fake_video[0].count == 30
    assert fake_video[0].fps == 30
    destination = env.get_sim_rendering('run', str(tmp_path))
    assert destination.exists() and fake_video[0].released
    env.data.time = 0.
    env._update_renderer()
    assert fake_video[1].count == 1
    env.close()
    assert fake_video[1].released


def test_overlay_layers_preserve_scene_geometry_and_bound_capacity(model):
    output = SceneOutput(model, mujoco.MjData(model), width=16, height=16)
    scene = mujoco.MjvScene(model, maxgeom=3)
    scene.ngeom = 1  # One simulated geometry already occupies the scene.
    output.set_overlay('reference', [[1, 2, 3]], radius=.02)
    output.set_overlay('prediction', [[4, 5, 6], [7, 8, 9]], radius=.03)
    output._draw_overlays(scene)
    assert scene.ngeom == 3
    np.testing.assert_array_equal(scene.geoms[1].pos, [1, 2, 3])
    np.testing.assert_array_equal(scene.geoms[2].pos, [4, 5, 6])
    output.close()
