from __future__ import annotations

import csv
import json
from pathlib import Path
from time import time
from contextlib import contextmanager
from enum import Enum
from typing import Optional

import numpy as np
import typer
from sklearn.cluster import KMeans, MiniBatchKMeans


def benchmark(kmeans: KMeans, data: np.array, n: int):
    t0 = time()
    for _ in range(n):
        kmeans.fit(data)
    duration = int(1e9 * (time() - t0) // n)
    return json.dumps({"duration_ns": duration})


@contextmanager
def csv_io_context(input_path: Path, output_path: Path, delimiter: str):
    def _task():
        cluster_centers, benchmark_result = yield
        with output_path.open("w") as fp:
            csv.writer(fp).writerows(cluster_centers)
        print(benchmark_result)

    with input_path.open("r") as fp:
        data = np.array(
            [list(map(float, row)) for row in csv.reader(fp, delimiter=delimiter)]
        )

    generator = _task()
    next(generator)
    try:
        yield data, generator
    except StopIteration:
        pass


class Algorithm(str, Enum):
    vanila = "vanila"
    minibatch = "minibatch"


def main(
    n_clusters: int,
    input_path: Path,
    output_path: Path,
    algorithm: Algorithm = Algorithm.vanila,
    delimiter: str = ",",
    batch_size: Optional[int] = None,
):
    KmeansClass = {
        Algorithm.vanila: KMeans,
        Algorithm.minibatch: MiniBatchKMeans,
    }[algorithm]

    args = {}
    if batch_size is not None:
        args["batch_size"] = batch_size
    with csv_io_context(input_path, output_path, delimiter) as (data, c):
        kmeans = KmeansClass(n_clusters=n_clusters, **args)
        kmeans.fit(data)
        benchmark_result = benchmark(kmeans, data, 8)
        c.send((kmeans.cluster_centers_, benchmark_result))


if __name__ == "__main__":
    typer.run(main)
