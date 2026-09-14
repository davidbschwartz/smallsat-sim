"""Run the nominal MPC classical-controller benchmark."""

from .rl_benchmarker import Benchmarker


def main():
    benchmarker = Benchmarker(run_name="nominal_mpc")
    benchmarker.deploy_and_test_classic(controller_type="nominal_mpc")


if __name__ == "__main__":
    main()
