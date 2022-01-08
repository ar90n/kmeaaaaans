import csv
import json
from pathlib import Path
from time import time

import numpy as np
import typer
from numpy.lib.shape_base import row_stack
from sklearn.cluster import KMeans


def main(n_clusters: int, input_path: Path, output_path: Path):
    with input_path.open("r") as fp:
        data = np.array([list(map(float, row)) for row in csv.reader(fp)])

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    n = 1
    t0 = time()
    for _ in range(n):
        kmeans.fit(data)
    duration = int(1e9 * (time() - t0) // n)
    print(json.dumps({"duration_ns": duration}))

    with output_path.open("w") as fp:
        csv.writer(fp).writerows(kmeans.cluster_centers_)


if __name__ == "__main__":
    typer.run(main)
