"""Incremental MP4 recording shared by classical and functional simulations."""
from pathlib import Path
import shutil
import tempfile

import cv2


class VideoRecorder:
    """Write RGB frames immediately; optionally choose the final path after recording."""

    def __init__(self):
        self.writer = None
        self.path = None
        self._temporary = None

    def start(self, *, width, height, fps, path=None):
        self.finish()
        if path is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="smallsat-video-")
            path = Path(self._temporary.name) / "recording.mp4"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not self.writer.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open video writer: {path}")

    def write(self, frame):
        if self.writer is None:
            raise RuntimeError("Start the recorder before writing frames")
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def finish(self, path=None):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if path is not None and self.path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination != self.path:
                shutil.move(str(self.path), str(destination))
            self.path = destination
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
        return self.path

    def close(self):
        self.finish()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self.path = None
