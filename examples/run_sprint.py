"""Run the small Sprint PPO preset from the repository root."""
from pathlib import Path

from smallsat_sim.api.experiments import make_experiment


def main():
    experiment = make_experiment(Path(__file__).with_name('sprint.yaml'))
    try:
        experiment.runner.learn()
        experiment.runner.evaluate()
    finally:
        experiment.env.close()


if __name__ == '__main__':
    main()
