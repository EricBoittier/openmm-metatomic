"""Export the SOAP-BPNN TorchScript twin used by the C++ spike."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from _bpnn import export_bpnn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="soap-bpnn.pt")
    args = parser.parse_args()
    export_bpnn(args.path)
    print(f"wrote {args.path}")


if __name__ == "__main__":
    main()
