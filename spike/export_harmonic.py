"""Export TorchScript harmonic wells for the M0 spike and gallery."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from _harmonic import K, SYSTEMS, export_system  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="harmonic.pt")
    parser.add_argument("--system", default="water", choices=sorted(SYSTEMS))
    parser.add_argument("--all", dest="export_all", action="store_true")
    args = parser.parse_args()
    if args.export_all:
        outdir = Path(args.path)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, system in SYSTEMS.items():
            export_system(system, str(outdir / f"{name}.pt"), K)
            print(f"wrote {outdir / name}.pt")
        return
    export_system(SYSTEMS[args.system], args.path, K)
    print(f"wrote {args.path}")


if __name__ == "__main__":
    main()
