"""Paired Monte Carlo MPC evaluation under actuator faults."""

from .common import execute, parser
from .runtime import evaluate


def main(argv=None):
    execute("exp2_fault_robustness", parser("exp2_fault_robustness").parse_args(argv), evaluate)


if __name__ == "__main__":
    main()
