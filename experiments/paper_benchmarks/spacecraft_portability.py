"""The same task and controller workflow with three physical assets."""

from .common import execute, parser
from .runtime import evaluate


def main(argv=None):
    execute("spacecraft_portability", parser("spacecraft_portability").parse_args(argv), evaluate)


if __name__ == "__main__":
    main()
