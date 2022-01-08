from __future__ import annotations

import csv
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
import json

import numpy as np
import typer
from sklearn import metrics


@contextmanager
def csv_io_context(input_path: Path, pred_path: Path, gt_path: Path, delimiter: str):
    with input_path.open("r") as fp:
        data = np.array(
            [list(map(float, row)) for row in csv.reader(fp, delimiter=delimiter)]
        )
    with pred_path.open("r") as fp:
        pred = np.array(
            [list(map(float, row)) for row in csv.reader(fp, delimiter=delimiter)]
        )

    gt = None
    if gt_path is not None:
        with gt_path.open("r") as fp:
            gt = np.array(
                [list(map(float, row)) for row in csv.reader(fp, delimiter=delimiter)]
            )

    yield data, pred, gt


def calc_labels(data: np.array, pred: np.array):
    labels = []
    inertia = 0.0
    for i in range(data.shape[0]):
        dists = np.linalg.norm(data[i] - pred, axis=1)
        label = np.argmin(dists)
        inertia += dists[label]
        labels.append(label)
    return labels, inertia


def calc_metrics(labels: np.array, gt: np.array):
    clustering_metrics = [
        ("homogeneity", metrics.homogeneity_score),
        ("completeness", metrics.completeness_score),
        ("v_measure", metrics.v_measure_score),
        ("adjusted_rand", metrics.adjusted_rand_score),
        ("adjusted_mutual_info", metrics.adjusted_mutual_info_score),
    ]
    return {l: m(labels, gt) for (l, m) in clustering_metrics}


def main(
    input_path: Path,
    pred_path: Path,
    gt_path: Optional[Path] = None,
    delimiter: str = ",",
):
    with csv_io_context(input_path, pred_path, gt_path, delimiter) as (data, pred, gt):
        pred_labels, pred_inertia = calc_labels(data, pred)
        metrics = {}
        if gt is not None:
            gt_labels, _ = calc_labels(data, gt)
            metrics = calc_metrics(pred_labels, gt_labels)
        metrics["inertia"] = pred_inertia
    print(json.dumps(metrics))


if __name__ == "__main__":
    typer.run(main)
