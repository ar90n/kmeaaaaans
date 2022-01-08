from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import typer


def make_table(dataset_name: str, keys: List[str], benchmark):
    result = []
    result.append(f"## {dataset_name}")
    result.append("|implement|" + "|".join(keys) + "|")
    result.append("|---|" + "|".join([":---:"] * len(keys)) + "|")
    for impl, source in benchmark.items():
        row = [impl]
        for key in keys:
            row.append(str(source.get(key, "-")))
        result.append("|" + "|".join(row) + "|")
    return "\n".join(result)


def extract_keys(benchmarks):
    keys = list(
        set(
            sum(
                [
                    sum([list(result.keys()) for result in benchmark.values()], [])
                    for benchmark in benchmarks.values()
                ],
                [],
            )
        )
    )
    keys.remove("duration_ns")
    keys.insert(0, "duration_ns")
    return keys


def main(benchmark_json_path: Path):
    benchmarks = json.loads(benchmark_json_path.read_text())

    report = []
    report.append("# Benchmark")
    keys = extract_keys(benchmarks)
    for dataset, benchmark in benchmarks.items():
        result = make_table(dataset, keys, benchmark)
        report.append(result)
        report.append("\n")
    print("\n".join(report))


if __name__ == "__main__":
    typer.run(main)
