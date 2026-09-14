"""Build and inspect a wheel from clean sources, then smoke-test outside the checkout."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='smallsat-release-') as directory:
        clean = Path(directory)
        for name in ('pyproject.toml', 'README.md', 'LICENSE'):
            shutil.copy2(ROOT / name, clean / name)
        shutil.copytree(ROOT / 'src/smallsat_sim', clean / 'src/smallsat_sim',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
        subprocess.run([sys.executable, '-c',
                        'from setuptools.build_meta import build_wheel; build_wheel("wheel")'],
                       cwd=clean, check=True)
        wheel, = (clean / 'wheel').glob('*.whl')
        installed = clean / 'installed'
        with ZipFile(wheel) as archive:
            names = set(archive.namelist())
            expected = {str(p.relative_to(clean / 'src'))
                        for p in (clean / 'src/smallsat_sim').rglob('*')
                        if p.is_file() and p.suffix in
                        {'.py', '.yaml', '.xml', '.obj', '.png', '.mtl', '.STL'}}
            missing = expected - names
            stale = {n for n in names if n.endswith('.py')} - expected
            if missing or stale:
                raise RuntimeError(f'Wheel mismatch: missing={missing}, stale={stale}')
            metadata, = [n for n in names if n.endswith('.dist-info/METADATA')]
            text = archive.read(metadata).decode()
            if 'Requires-Dist: pytest' in text:
                raise RuntimeError('pytest must remain a development dependency')
            for package in ('gpytorch', 'linear-operator', 'acados-template', 'l4acados'):
                if not any(line.startswith(f'Requires-Dist: {package}')
                           and ' @ git+https://' in line for line in text.splitlines()):
                    raise RuntimeError(f'Missing fork provenance: {package}')
            archive.extractall(installed)
        smoke = '''from pathlib import Path
import smallsat_sim
assert Path(smallsat_sim.__file__).is_relative_to(Path.cwd())
from smallsat_sim.api.experiments import make_experiment
for vehicle in ("astrobee", "cubesat", "sprint"):
    experiment = make_experiment(vehicle=vehicle, controller="pd", planner="oracle",
                                 headless=True, log=False)
    try:
        experiment.env.step(input=experiment.controller.get_control_input(experiment.env))
    finally:
        experiment.env.close()
    print(vehicle, "wheel smoke passed")
'''
        subprocess.run([sys.executable, '-c', smoke], cwd=installed,
                       env={**os.environ, 'PYTHONPATH': str(installed)}, check=True)
        destination = output / wheel.name
        shutil.copy2(wheel, destination)
        print(f'Validated wheel: {destination}')


if __name__ == '__main__':
    main()
