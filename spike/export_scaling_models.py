"""Export TorchScript harmonic wells sized for the C++ scaling benchmark.

Same model family as HarmonicModel.h (E = 0.5 * k * sum ||r - r0||^2, k=1,
r0=0, nm/kJ-mol) so the torch and core backends are evaluating the exact
same math at each atom count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from _harmonic import export_system  # noqa: E402

TYPE_TABLE = [1, 6, 8]


def main():
    out_dir = Path(sys.argv[1])
    sizes = [int(x) for x in sys.argv[2:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        system = {
            "title": f"harmonic-{n}",
            "types": [TYPE_TABLE[i % 3] for i in range(n)],
            "rest": [[0.0, 0.0, 0.0] for _ in range(n)],
        }
        path = out_dir / f"harmonic-{n}.pt"
        export_system(system, str(path))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
