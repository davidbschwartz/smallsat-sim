"""Vector environment lifecycle adapter for shared rendering resources."""
from .scene import SceneOutput


class VectorRendering:
    def _create_viewer(self, args):
        self.visualization = SceneOutput(
            self.model, self.data_vec[0], width=self.env_cfg.renderer.width,
            height=self.env_cfg.renderer.height, native=True,
            key_callback=self._key_callback, distance=3., azimuth=10.,
        )
        self.viewer = self.visualization.native

    def rollout_visualization(self, mode, step_config):
        vis = getattr(self, "_rl_visualization", None)
        return vis.begin(mode, step_config) if vis is not None else None

    def get_sim_rendering(self, *args, **kwargs):
        if getattr(self, "_rl_visualization", None) is None:
            return super().get_sim_rendering(*args, **kwargs)

    def close(self):
        vis = getattr(self, "_rl_visualization", None)
        try:
            if vis is not None:
                vis.close()
        finally:
            self._rl_visualization = None
            super().close()
