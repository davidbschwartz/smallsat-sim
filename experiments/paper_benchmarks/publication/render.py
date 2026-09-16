"""Render deterministic saved campaign poses; no controller execution or pose interpolation."""

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import mujoco
from .generate import ROOT, representative


class Replay:
    def __init__(self, source, output, width=1600, height=1000, docking_results=None):
        self.source, self.output = source.resolve(), output.resolve()
        self.docking_results = docking_results.resolve() if docking_results else None
        self.command = "python -m experiments.paper_benchmarks.publication.render"
        if self.docking_results:
            self.command += " --docking-results " + shlex.quote(os.path.relpath(self.docking_results, ROOT))
        if self.output == self.source or self.output.is_relative_to(self.source):
            raise ValueError("Output must be outside source artifacts")
        self.width, self.height = width, height
        self.entries = []
        self.destination = self.output / "figures/renders"
        self.destination.mkdir(parents=True, exist_ok=True)

    def recording(self, row):
        if self.docking_results:
            run = self.docking_results / "paper/exp4_docking" / row.run_id
            if run.is_dir():
                return run / "recordings" / row.condition / f"{int(row.trial):04d}.npz"
        p = (
            self.source
            / "demonstration_all_recordings_traces/results/demonstration_campaign_final/paper"
        )
        candidates = list(
            p.glob(f"*/{row.run_id}/recordings/{row.condition}/{int(row.trial):04d}.npz")
        )
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one recording: {row.run_id}/{row.trial}")
        return candidates[0]

    def model(self, row):
        paths = list((self.source / "paper-results/raw_runs").glob(f"*/{row.run_id}/model.xml"))
        if self.docking_results:
            path = self.docking_results / "paper/exp4_docking" / row.run_id / "model.xml"
            if path.exists():
                paths = [path]
        if len(paths) != 1:
            raise ValueError(f"Missing model: {row.run_id}")
        path = paths[0]
        xml = ET.fromstring(path.read_text())
        compiler = xml.find("compiler")
        for key in ["meshdir", "texturedir"]:
            compiler.set(key, str(ROOT / "src/smallsat_sim/model"))
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        return model, path

    def frame(self, row, index, *, lookat=None, distance=2, azimuth=135, elevation=-20):
        recording = self.recording(row)
        with np.load(recording, allow_pickle=False) as archive:
            poses, times = archive["qpos"], archive["time"]
        if not np.isfinite(poses).all() or not np.all(np.diff(times) > 0):
            raise ValueError(f"Invalid recording: {recording}")
        index = int(index if index >= 0 else len(times) + index)
        model, path = self.model(row)
        if poses.shape != (len(times), model.nq):
            raise ValueError("Pose/model mismatch")
        data = mujoco.MjData(model)
        data.qpos[:] = poses[index]
        data.time = times[index]
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = poses[index, :3] if lookat is None else lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        option.sitegroup[:] = 0
        with mujoco.Renderer(model, height=self.height, width=self.width) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            rgb = renderer.render().copy()
        info = dict(
            run_id=row.run_id,
            trial=int(row.trial),
            frame_index=index,
            time_s=float(times[index]),
            recording=str(recording.relative_to(ROOT)),
            model=str(path.relative_to(ROOT)),
            camera=dict(
                lookat=camera.lookat.tolist(),
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
            ),
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [recording, path]
            },
        )
        return Image.fromarray(rgb), info

    def save(self, img, name, infos, purpose, selection):
        p = self.destination / (name + ".png")
        img.save(p)
        self.entries.append(
            dict(
                filename=str(p.relative_to(self.output)),
                purpose=purpose,
                script="experiments/paper_benchmarks/publication/render.py",
                command=self.command,
                selection=selection,
                frames=infos,
                render_settings="Saved MuJoCo background, floor, materials and lighting unchanged; collision group 3 and task sites hidden; no overlays. Mesh paths relocated to local repository.",
                caveats="Replay of saved poses only. No new dynamics or hardware measurements.",
            )
        )

    def run(self):
        docking = pd.read_csv(self.output / "tables/docking_selection.csv").iloc[0]
        recording = self.recording(docking)
        with np.load(recording) as a:
            times, poses = a["time"], a["qpos"]
        contact = docking.first_contact_time
        requested = [times[0], contact, contact + 2, times[-1]]
        # At contact, use latest saved state at or before physics-step contact event.
        indices = np.searchsorted(times, requested, side="right").clip(1, len(times)) - 1
        center = (poses[0, :3] + poses[-1, :3]) / 2
        images = []
        infos = []
        for label, index in zip(["approach", "first_contact", "settling", "final"], indices):
            img, info = self.frame(
                docking, index, lookat=center, distance=3.5, azimuth=60, elevation=-20
            )
            info["requested_time_s"] = float(requested[len(images)])
            self.save(
                img,
                "docking_" + label,
                [info],
                "Docking sequence: " + label,
                docking.selection_rule,
            )
            images.append(img)
            infos.append(info)
        canvas = Image.new("RGB", (self.width * 2, self.height * 2), "white")
        for i, img in enumerate(images):
            canvas.paste(img, ((i % 2) * self.width, (i // 2) * self.height))
        self.save(
            canvas,
            "docking_sequence",
            infos,
            "Approach/contact/settling/final, row-major",
            docking.selection_rule,
        )
        # Standalone camera candidate uses the identical representative contact state.
        img, info = self.frame(docking, indices[1], distance=3.0, azimuth=60, elevation=-20)
        self.save(img, "docking_standalone", [info], "Contact close-up", docking.selection_rule)
        # The fault campaign is free space; use the saved Gateway approach for
        # inspection-context candidates and disclose this provenance explicitly.
        row = docking
        for name, distance, azimuth in [
            ("inspection_close", 3.2, 45),
            ("inspection_context", 8, 30),
        ]:
            img, info = self.frame(row, 0, distance=distance, azimuth=azimuth, elevation=-20)
            self.save(
                img,
                name,
                [info],
                "Astrobee/Gateway approach context; docking campaign, not a separate inspection trial",
                docking.selection_rule + "; initial approach frame",
            )
        portable = pd.read_csv(self.output / "tables/spacecraft_portability_trials.csv")
        images = []
        infos = []
        for vehicle in ["astrobee", "cubesat", "sprint"]:
            row, rule = representative(
                portable[portable.spacecraft == vehicle], "final_position_error"
            )
            img, info = self.frame(row, 0, distance=1.15, azimuth=135, elevation=-20)
            images.append(img)
            infos.append(info)
        canvas = Image.new("RGB", (self.width * len(images), self.height), "white")
        for i, img in enumerate(images):
            canvas.paste(img, (i * self.width, 0))
        self.save(
            canvas,
            "spacecraft_comparison",
            infos,
            "Left to right: Astrobee, CubeSat, idealized AERCam Sprint. Equal camera distance and image scale.",
            "Each vehicle: lower median successful terminal position error; initial saved frame",
        )
        rl = pd.read_csv(self.output / "tables/exp3_rl_robustness_trials.csv")
        subset = (
            rl[
                (rl.controller == "ppo")
                & (rl.regime == "randomized")
                & (rl.training_seed == 0)
                & (rl.condition == "combined")
            ]
            .sort_values("evaluation_seed")
            .head(12)
        )
        images = []
        infos = []
        for _, row in subset.iterrows():
            img, info = self.frame(row, 0, distance=1.15, azimuth=135, elevation=-20)
            images.append(img)
            infos.append(info)
        canvas = Image.new("RGB", (self.width * 4, self.height * 3), "white")
        for i, img in enumerate(images):
            canvas.paste(img, ((i % 4) * self.width, (i // 4) * self.height))
        self.save(
            canvas,
            "randomized_batch",
            infos,
            "Twelve recorded randomized spacecraft attitudes, consistent camera centered on each instance; contact-sheet montage.",
            "First 12 evaluation seeds; PPO randomized training seed 0, combined suite; initial frames, no success filtering",
        )
        (self.output / "renders_manifest.json").write_text(
            json.dumps(self.entries, indent=2) + "\n"
        )
        readme = self.output / "README.md"
        text = readme.read_text().split("\n## Candidate renders\n")[0]
        text += "\n## Candidate renders\n\nRegenerate: `python -m experiments.paper_benchmarks.publication.render`. Exact source paths, hashes, frame times, cameras, and selection rules for each image are in [renders_manifest.json](renders_manifest.json). All renders are clean PNGs; sequence contact frames use the latest recorded pose at or before first contact (not interpolated). Replay uses the saved MuJoCo background, floor, materials and lighting without overrides (black space skybox where present). Collision proxies and task sites are hidden for the clean view. Mesh/texture paths are relocated to the local checkout.\n\n"
        for a in self.entries:
            text += f"- [{a['filename']}]({a['filename']}): {a['purpose']}. Selection: {a['selection']}. {len(a['frames'])} saved frames.\n"
        text = text.replace("Regenerate: `python -m experiments.paper_benchmarks.publication.render`", "Regenerate: `" + self.command + "`")
        readme.write_text(text)
        print(f"Wrote {len(self.entries)} candidate render artifacts")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "smallsat-demonstration")
    p.add_argument("--output", type=Path, default=ROOT / "paper")
    p.add_argument("--camera-audit", action="store_true")
    p.add_argument("--docking-results", type=Path, help="Replacement docking campaign root containing paper/exp4_docking")
    args = p.parse_args()
    replay = Replay(args.source, args.output, docking_results=args.docking_results)
    if args.camera_audit:
        row = pd.read_csv(args.output / "tables/docking_selection.csv").iloc[0]
        images = [
            replay.frame(row, -1, distance=3, azimuth=a, elevation=-15)[0]
            for a in [0, 90, 180, 270]
        ]
        canvas = Image.new("RGB", (3200, 2000), "white")
        for i, img in enumerate(images):
            canvas.paste(img, ((i % 2) * 1600, (i // 2) * 1000))
        canvas.save("/tmp/smallsat-camera-audit.png")
    else:
        replay.run()


if __name__ == "__main__":
    main()
