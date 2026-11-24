"""Read target XYZ from CSV, export nearest 1-based VTK nodes, and print five."""

import csv
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MESH_FILE = PROJECT_DIR / "out" / "mesh_3d_with_facies.vtk"
TARGET_POINTS_FILE = PROJECT_DIR / "data" / "wells_location.csv"
OUTPUT_FILE = PROJECT_DIR / "out" / "sourcing_nodes.csv"


def main():
    with TARGET_POINTS_FILE.open(newline="", encoding="utf-8-sig") as stream:
        target_points = [
            [float(row[axis]) for axis in ("X", "Y", "Z")]
            for row in csv.DictReader(stream)
        ]

    with MESH_FILE.open(encoding="ascii") as stream:
        for _ in range(4):
            next(stream)
        node_count = int(next(stream).split()[1])
        points = np.loadtxt(stream, max_rows=node_count, ndmin=2)

    indices = list(dict.fromkeys(
        int(np.argmin(np.sum((points - target) ** 2, axis=1)))
        for target in np.atleast_2d(target_points)
    ))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "X", "Y", "Z"])
        print("ID,X,Y,Z")
        for position, index in enumerate(indices):
            row = [index + 1, *(f"{value:.17g}" for value in points[index])]
            writer.writerow(row)
            if position < 10:
                print(*row, sep=",")

    print(f"\nExported {len(indices)} nodes to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
