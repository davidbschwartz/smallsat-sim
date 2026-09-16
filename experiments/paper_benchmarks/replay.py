"""Record evaluation poses and render saved episodes without rerunning controllers."""

import argparse
from pathlib import Path

import numpy as np


class EpisodeRecording:
    """Capture full model poses at control boundaries, including the initial state."""

    def __init__(self, env):
        self.times = []
        self.poses = []
        self.add(env)

    def add(self, env):
        self.times.append(float(env.data.time))
        self.poses.append(env.data.qpos.copy())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, time=self.times, qpos=self.poses)


def frame_indices(times, fps):
    """Resample at video rate using the most recent recorded pose, without interpolation."""
    times = np.asarray(times)
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be finite and positive")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Recording must contain finite timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Recording timestamps must increase strictly")
    grid = times[0] + np.arange(int(np.ceil(round((times[-1] - times[0]) * fps, 10))) + 1) / fps
    return np.searchsorted(times, grid, side="right").clip(1, len(times)) - 1


def render(
    recording, output, *, width=1280, height=720, fps=30, azimuth=135, elevation=-25, distance=None
):
    """Export an MP4 and start/middle/end stills with a fixed, trajectory-centered camera."""
    import cv2
    import mujoco

    from smallsat_sim.envs.rendering.video import VideoRecorder

    recording, output = Path(recording), Path(output)
    run = next((p for p in recording.parents if (p / "model.xml").exists()), None)
    if run is None:
        raise ValueError("Recording must be inside a run directory containing model.xml")
    with np.load(recording, allow_pickle=False) as archive:
        times, poses = archive["time"], archive["qpos"]
    indices = frame_indices(times, fps)
    model = mujoco.MjModel.from_xml_path(str(run / "model.xml"))
    if poses.shape != (len(times), model.nq) or not np.isfinite(poses).all():
        raise ValueError("Recording poses must be finite and match the saved model")
    if width <= 0 or height <= 0 or (distance is not None and distance <= 0):
        raise ValueError("Image dimensions and camera distance must be positive")
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    low, high = poses[:, :3].min(axis=0), poses[:, :3].max(axis=0)
    camera.lookat[:] = (low + high) / 2
    camera.distance = (
        distance if distance is not None else max(3.0, np.linalg.norm(high - low) * 1.8)
    )
    camera.azimuth, camera.elevation = azimuth, elevation
    output.mkdir(parents=True, exist_ok=True)
    video = VideoRecorder()
    stills = {"start": 0, "middle": len(indices) // 2, "end": len(indices) - 1}
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            video.start(width=width, height=height, fps=fps, path=output / "episode.mp4")
            for frame_number, index in enumerate(indices):
                data.qpos[:] = poses[index]
                data.time = times[index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                cv2.putText(
                    frame,
                    f"{recording.parent.name} / {recording.stem}  |  t = {times[index]:.2f} s",
                    (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                video.write(frame)
                for name, still_index in stills.items():
                    if frame_number == still_index:
                        path = output / f"{name}.png"
                        if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                            raise RuntimeError(f"Cannot write screenshot: {path}")
    finally:
        video.close()
    return output / "episode.mp4"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Run's recordings/<condition>/<trial>.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--azimuth", type=float, default=135)
    parser.add_argument("--elevation", type=float, default=-25)
    parser.add_argument("--distance", type=float)
    args = parser.parse_args(argv)
    print(render(**vars(args)))


if __name__ == "__main__":
    main()
