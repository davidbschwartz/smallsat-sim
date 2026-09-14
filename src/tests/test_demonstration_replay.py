"""Saved poses replay independently of controller execution."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.demonstration_use_cases.replay import EpisodeRecording, frame_indices


def test_recording_copies_initial_and_final_poses(tmp_path):
    env = SimpleNamespace(data=SimpleNamespace(time=0.0, qpos=np.array([1.0, 2.0])))
    recording = EpisodeRecording(env)
    env.data.time = 0.25
    env.data.qpos[:] = [3.0, 4.0]
    recording.add(env)
    path = tmp_path / "recordings" / "0000.npz"
    recording.save(path)
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["time"], [0.0, 0.25])
        np.testing.assert_array_equal(saved["qpos"], [[1.0, 2.0], [3.0, 4.0]])


def test_replay_preserves_timing_and_includes_final_pose():
    np.testing.assert_array_equal(frame_indices([0.0, 0.25, 0.5], 8), [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(frame_indices([0.0], 30), [0])
    assert frame_indices([0.0, 0.27], 30)[-1] == 1


@pytest.mark.parametrize("times", [[], [0.0, 0.0], [1.0, 0.0], [float("nan")]])
def test_replay_rejects_invalid_timestamps(times):
    with pytest.raises(ValueError):
        frame_indices(times, 30)
