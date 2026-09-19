"""RL visualization output and rendering resource tests."""

from types import SimpleNamespace

import cv2
import test_functional_rollout as fixtures
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from smallsat_sim.envs.rendering import browser as browser_rendering
from smallsat_sim.envs.rendering.rollout import RLVisualization


def test_callback_selects_one_env_and_includes_terminal_frame():
    vis = object.__new__(RLVisualization)
    vis.browser = True
    vis.index, vis.stride = 1, 3
    received = []

    def receive(qpos, done):
        received.append((np.array(qpos), bool(done)))
        return np.int32(0)

    vis._receive = receive

    def step(carry, i):
        state = SimpleNamespace(qpos=jnp.array([[99.0], [i.astype(float)]]))
        vis.observe(i, state, jnp.array([False, i == 2]))
        return carry, i

    jax.lax.scan(step, 0, jnp.arange(5))
    jax.effects_barrier()
    assert [float(x[0][0]) for x in received] == [0, 2, 3]
    assert [x[1] for x in received] == [False, True, False]
    assert all(x[0].shape == (1,) for x in received)


def test_video_is_bounded_and_training_does_not_repeat(tmp_path, monkeypatch):
    frames, writers = [], []

    class Renderer:
        def __init__(self, *a, **kw):
            pass

        def update_scene(self, *a, **kw):
            pass

        def render(self):
            return np.zeros((16, 16, 3), np.uint8)

        def close(self):
            pass

    class Writer:
        def __init__(self, *a):
            self.released = False
            writers.append(self)

        def isOpened(self):
            return True

        def write(self, frame):
            frames.append(frame)

        def release(self):
            self.released = True

    monkeypatch.setattr(mujoco, "Renderer", Renderer)
    monkeypatch.setattr(cv2, "VideoWriter", Writer)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    args = SimpleNamespace(
        video=True, viewer=False, video_duration=0.06, video_dir=str(tmp_path)
    )
    config = SimpleNamespace(num_envs=2, model_dt=0.01, control_decimation=1)
    vis = RLVisualization(model, args, SimpleNamespace(width=16, height=16))
    try:
        vis.begin("training", config)
        for _ in range(10):
            vis._receive(model.qpos0, False)
        vis.finish()
        assert len(frames) == 2
        assert writers[0].released
        assert vis.begin("training", config) is None
        vis.begin("evaluation", config)
        vis._receive(model.qpos0, False)
        vis._receive(model.qpos0, True)
        vis._receive(model.qpos0, False)
        vis.finish()
        assert len(frames) == 4
        assert writers[1].released
    finally:
        vis.close()
        vis.close()


def test_viewer_only_does_not_initialize_renderer_or_write(tmp_path, monkeypatch):
    class Scene:
        def __init__(self, *a, **kw):
            pass

        def update_from_mjdata(self, data):
            pass

    class Server:
        def __init__(self, **kw):
            pass

        def on_client_connect(self, fn):
            pass

        def get_port(self):
            return 8080

        def get_clients(self):
            return {}

        def stop(self):
            pass

    monkeypatch.setattr(browser_rendering, "ViserMujocoScene", Scene)
    monkeypatch.setattr(browser_rendering, "viser", SimpleNamespace(ViserServer=Server))

    def forbidden(*a, **kw):
        raise AssertionError("Viewer must not initialize OpenGL")

    monkeypatch.setattr(mujoco, "Renderer", forbidden)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    vis = RLVisualization(
        model,
        SimpleNamespace(video=False, viewer=True, video_dir=str(tmp_path)),
        SimpleNamespace(width=16, height=16),
    )
    try:
        vis.begin(
            "training", SimpleNamespace(num_envs=1, model_dt=0.01, control_decimation=1)
        )
        vis._receive(model.qpos0, False)
        vis.finish()
        assert not list(tmp_path.iterdir())
    finally:
        vis.close()


def test_visualized_rollout_preserves_results_and_captures_before_reset(monkeypatch):
    original = fixtures.ru.run_functional_rollout
    vis = object.__new__(RLVisualization)
    vis.browser = True
    vis.index, vis.stride = 0, 1
    received = []

    def receive(qpos, done):
        received.append(float(qpos[0]))
        return np.int32(0)

    vis._receive = receive

    def instrumented(**kwargs):
        return original(**kwargs, visualization=vis)

    monkeypatch.setattr(fixtures.ru, "run_functional_rollout", instrumented)
    # Reuse the numerical assertions on returns, resets and final state.
    fixtures.test_run_functional_rollout_resets_and_reports_returns(single_episode=False)
    jax.effects_barrier()
    assert received == [1, 0, 1]
