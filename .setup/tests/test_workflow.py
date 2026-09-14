"""Exercise setup/launcher behavior without apt, a GPU, or native compilation.

Run with: python3 -m unittest discover -s .setup/tests -v
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="smallsat setup ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(REPO / ".setup", self.root / ".setup")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "commands.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        SMALLSAT_TEST_LOG=str(self.log))
        self.stub("uname", "#!/bin/sh\necho Linux\n")
        self.stub("uv", """#!/usr/bin/env python3
import json, os, sys
with open(os.environ['SMALLSAT_TEST_LOG'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(),
                        'acados': os.environ.get('ACADOS_SOURCE_DIR')}) + '\\n')
""")
        self.stub("nvidia-smi", "#!/bin/sh\nexit 0\n")
        self.stub("sudo", "#!/bin/sh\necho 'Unexpected sudo' >&2\nexit 99\n")
        (self.root / ".setup/native/install_mpc.sh").write_text(
            '#!/bin/sh\ntouch "$(dirname "$0")/../../native-built"\n')
        (self.root / ".venv/bin").mkdir(parents=True)
        (self.root / ".venv/bin/python").symlink_to(shutil.which("python3"))

    def stub(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def run_script(self, script, *args):
        return subprocess.run(["bash", str(self.root / script), *args],
                              cwd="/", env=self.env, text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_default_installs_full_stack_without_implicit_sudo(self):
        result = self.run_script(".setup/ubuntu/setup.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "native-built").exists())
        sync, check = self.calls()
        self.assertEqual(sync["args"], ["sync", "--locked", "--extra", "cuda12",
                                        "--extra", "warp", "--extra", "mpc"])
        self.assertEqual(check["args"], ["run", "--no-sync", "python",
                                         ".setup/check_install.py", "--gpu", "--mpc"])
        self.assertEqual(check["acados"], str(self.root / "deps/acados"))

    def test_cpu_without_mpc_needs_no_gpu_or_native_build(self):
        self.stub("nvidia-smi", "#!/bin/sh\nexit 98\n")
        result = self.run_script(".setup/ubuntu/setup.sh", "--cpu", "--without-mpc")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "native-built").exists())
        self.assertEqual(self.calls()[0]["args"], ["sync", "--locked"])

    def test_launcher_preserves_arguments_and_environment(self):
        result = self.run_script(".setup/smallsat", "run", "python", "file with spaces.py", "--headless")
        self.assertEqual(result.returncode, 0, result.stderr)
        call, = self.calls()
        self.assertEqual(call["args"], ["run", "--no-sync", "python", "file with spaces.py", "--headless"])
        self.assertEqual(Path(call["cwd"]).resolve(), self.root.resolve())
        self.assertEqual(call["acados"], str(self.root / "deps/acados"))

    def test_macos_sets_native_library_path(self):
        self.stub("uname", "#!/bin/sh\necho Darwin\n")
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; printf "%s" "$DYLD_LIBRARY_PATH"',
             "bash", str(self.root / ".setup/env.sh")],
            env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.split(":")[0]).resolve(),
                         (self.root / "deps/acados/lib").resolve())

    def test_failed_sync_stops_before_install_check(self):
        self.stub("uv", "#!/bin/sh\nexit 17\n")
        result = self.run_script(".setup/ubuntu/setup.sh", "--cpu", "--without-mpc")
        self.assertEqual(result.returncode, 17)

    def test_invalid_options_do_not_install_anything(self):
        result = self.run_script(".setup/ubuntu/setup.sh", "--typo")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.log.exists())
        self.assertFalse((self.root / "native-built").exists())


if __name__ == "__main__":
    unittest.main()
