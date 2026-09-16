"""Classical simulation adapter for the shared scene and streaming recorder."""
from datetime import datetime
from pathlib import Path
import math

from smallsat_sim import SMALLSAT_STEWARD_ROOT_DIR
from .scene import SceneOutput


class ClassicalRendering:
    def _setup_rendering(self, args):
        self.visualization = None
        self.viewer = self.renderer = None
        self._last_video_time = None
        self._next_video_time = None
        browser = bool(getattr(args, "viewer", False)) or (
            not args.headless and bool(getattr(self.env_cfg.viewer, "use_viser", False))
        )
        native = not args.headless and not browser
        if not (browser or native or args.video):
            return
        self.visualization = SceneOutput(
            self.model, self.data, width=self.env_cfg.renderer.width,
            height=self.env_cfg.renderer.height, browser=browser, native=native,
            port=getattr(args, "viewer_port", self.env_cfg.viewer.viser_port),
            key_callback=self._key_callback,
        )
        self.viewer = self.visualization.native

    def _update_viewer(self):
        if self.visualization is not None:
            self.visualization.update_viewer()

    def _update_renderer(self):
        """Capture at configured simulation times, without buffering RGB frames."""
        if self.visualization is None or not self.args.video:
            return
        config = self.env_cfg.renderer
        timestamp = float(self.data.time)
        if not config.start_recording <= timestamp <= config.end_recording:
            return
        control_dt = float(self.model.opt.timestep) * self.env_cfg.control.control_decimation
        fps = float(config.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Recording fps must be finite and positive")
        fps = min(fps, 1 / control_dt)
        if self._next_video_time is None or timestamp < self._last_video_time:
            self._next_video_time = timestamp
        self._last_video_time = timestamp
        if timestamp + 1e-9 < self._next_video_time:
            return
        if self.visualization.recorder.writer is None:
            self.visualization.start_video(fps=fps)
        self.visualization.record_frame()
        # Advance the sampling grid rather than rounding every interval up to a step.
        intervals = max(1, math.floor((timestamp - self._next_video_time) * fps + 1e-8) + 1)
        self._next_video_time += intervals / fps

    def get_sim_rendering(self, output_filename, output_dir=None):
        """Finalize the streamed clip at the caller's requested destination."""
        vis = getattr(self, "visualization", None)
        if vis is None or vis.recorder.path is None:
            return
        directory = Path(output_dir or "videos")
        if not directory.is_absolute():
            directory = Path(SMALLSAT_STEWARD_ROOT_DIR) / directory
        stamp = getattr(self, "sim_start_time", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        path = vis.recorder.finish(directory / f"{output_filename}_{stamp}.mp4")
        vis.recorder.path = None
        self._last_video_time = self._next_video_time = None
        print(f"Video saved: {path}")
        return path

    def close(self):
        for planner in getattr(self, "_managed_planners", []):
            planner.close()
        self._managed_planners = []
        vis = getattr(self, "visualization", None)
        try:
            if vis is not None:
                vis.close()
        finally:
            self.visualization = None
            self.viewer = self.renderer = None
