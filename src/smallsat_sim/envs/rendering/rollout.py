"""Selected-environment outputs for functional RL rollouts.

All OpenGL work stays on one thread. Ordered IO callbacks transfer only one
pose at sampled simulation times; no callbacks are traced when disabled.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import copy
import math
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax.experimental import io_callback

from .scene import SceneOutput


def add_visualization_args(parser):
    parser.add_argument(
        "--viewer", action="store_true", help="Watch RL in a browser; saves nothing"
    )
    parser.add_argument(
        "--view-env",
        type=int,
        default=0,
        help="RL environment to watch/record (default: 0)",
    )
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument(
        "--video-duration",
        type=float,
        default=10.0,
        help="Maximum seconds in the single training clip (default: 10); evaluation records an episode",
    )
    parser.add_argument(
        "--video-dir",
        default="videos",
        help="Directory for explicitly requested videos",
    )


class RLVisualization:
    def __init__(self, model, args, renderer_config):
        self.index = getattr(args, "view_env", 0)
        self.browser = bool(getattr(args, "viewer", False))
        self.video = bool(args.video)
        self.duration = float(getattr(args, "video_duration", 10.0))
        if self.index < 0 or not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError(
                "view-env must be nonnegative and video-duration must be finite and positive"
            )
        self.model = copy.copy(model)
        self.args = args
        self.width, self.height = renderer_config.width, renderer_config.height
        self.output_dir = Path(getattr(args, "video_dir", "videos"))
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.training_recorded = False
        self.eval_count = 0
        self.closed = False
        self.output = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rl-viewer")
        try:
            self.pool.submit(self._initialize).result()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.output = SceneOutput(
            self.model, self.data, width=self.width, height=self.height,
            browser=self.browser, port=getattr(self.args, "viewer_port", 8080),
        )

    def begin(self, mode, config):
        if not 0 <= self.index < config.num_envs:
            raise ValueError(f"view-env {self.index} outside [0, {config.num_envs})")
        record = self.video and (mode == "evaluation" or not self.training_recorded)
        if not self.browser and not record:
            return None
        self.mode = mode
        self.record = record
        self.finished_episode = False
        self.frame_count = 0
        self.last_wall = 0.0
        dt = float(config.model_dt) * int(config.control_decimation)
        self.stride = max(1, round(1 / (30 * dt)))
        self.frame_dt = dt * self.stride
        self.max_frames = max(1, math.ceil(self.duration / self.frame_dt))
        self.pool.submit(self._begin_video).result()
        return self

    def _begin_video(self):
        if not self.record:
            return
        if self.mode == "evaluation":
            self.eval_count += 1
        name = (
            "training" if self.mode == "training" else f"evaluation-{self.eval_count}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output_dir / f"{self.run_id}-{name}-env{self.index}.mp4"
        self.output.start_video(path=self.video_path, fps=1 / self.frame_dt)
        print(f"Recording {self.video_path}")

    def observe(self, step, state, done):
        source = state.mjx_batch if hasattr(state, "mjx_batch") else state
        selected_done = done[self.index]
        sampled = (step % self.stride == 0) | selected_done
        if not self.browser and self.mode == "training":
            sampled = sampled & (step < self.max_frames * self.stride)

        def emit(_):
            return io_callback(
                self._receive,
                jax.ShapeDtypeStruct((), jnp.int32),
                source.qpos[self.index],
                selected_done,
                ordered=True,
            )

        jax.lax.cond(sampled, emit, lambda _: jnp.int32(0), operand=None)

    def _receive(self, qpos, done):
        self.pool.submit(self._frame, np.asarray(qpos), bool(done)).result()
        return np.int32(0)

    def _frame(self, qpos, done):
        if self.finished_episode:
            return
        now = time.monotonic()
        connected = self.browser and bool(self.output.browser.server.get_clients())
        show = connected and (
            self.mode == "evaluation" or now - self.last_wall >= 1 / 30
        )
        if show or self.output.recorder.writer is not None:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
        if show:
            if self.mode == "evaluation" and self.last_wall:
                time.sleep(max(0, self.frame_dt - (now - self.last_wall)))
            self.output.update_viewer()
            self.last_wall = time.monotonic()
        if self.output.recorder.writer is not None:
            self.output.camera.lookat[:] = qpos[:3]
            self.output.record_frame()
            self.frame_count += 1
            if self.mode == "training" and self.frame_count >= self.max_frames:
                self._finish_video()
        if done and self.mode == "evaluation":
            self.finished_episode = True
            self._finish_video()

    def _finish_video(self):
        if self.output.recorder.writer is not None:
            self.output.recorder.finish()
            if self.mode == "training":
                self.training_recorded = True
            print(f"Video saved: {self.video_path}")

    def finish(self):
        self.pool.submit(self._finish_video).result()

    def _close(self):
        if self.output is not None:
            self.output.close()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.pool.submit(self._close).result()
            finally:
                self.pool.shutdown()
