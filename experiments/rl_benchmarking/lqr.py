"""Run the LQR classical-controller benchmark."""

from .rl_benchmarker import Benchmarker


def main():
    benchmarker = Benchmarker(run_name="lqr")
    benchmarker.deploy_and_test_classic(controller_type="lqr")


if __name__ == "__main__":
    main()
