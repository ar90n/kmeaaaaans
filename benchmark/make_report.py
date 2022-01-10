from __future__ import annotations

import csv
import json
from os import getcwd
from pathlib import Path
from typing import List
from cpuinfo import get_cpu_info
import psutil

import typer


def make_benchmark_table(dataset_name: str, keys: List[str], benchmark):
    result = []
    result.append(f"#### {dataset_name}")
    result.append("|implement|" + "|".join(keys) + "|")
    result.append("|---|" + "|".join([":---:"] * len(keys)) + "|")
    for impl, source in benchmark.items():
        row = [impl]
        for key in keys:
            row.append(str(source.get(key, "-")))
        result.append("|" + "|".join(row) + "|")
    return "\n".join(result)

def make_dataset_table(datasets):
    result = []
    result.append("|dataset| source |")
    result.append("|---|:---:|")
    for (name, source) in datasets:
        result.append(f"|{name}|{source}|")
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
    keys.sort()
    keys.remove("inertia")
    keys.insert(0, "inertia")
    keys.remove("duration_ns")
    keys.insert(0, "duration_ns")
    return keys

# from https://www.thepythoncode.com/article/get-hardware-system-information-python
def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def make_env():
    cpu_brand = get_cpu_info()["brand_raw"]
    mem_size = get_size(psutil.virtual_memory().total)
    return f"""\
### Environment
* CPU: {cpu_brand}
* MEMORY: {mem_size}
"""

def make_dataset():
    dataset = [
        ("s1", "[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)"),
        ("s2", "[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)"),
        ("s3", "[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)"),
        ("s4", "[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)"),
        ("mnist", "[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)"),
    ]
    result = []
    result.append("### Dataset")
    result.append(make_dataset_table(dataset))
    return "\n".join(result)

def make_benchmark(benchmarks):
    report = []
    report.append("### Result")
    keys = extract_keys(benchmarks)
    for dataset, benchmark in benchmarks.items():
        result = make_benchmark_table(dataset, keys, benchmark)
        report.append(result)
        report.append("\n")
    return "\n".join(report)



def main(benchmark_json_path: Path):
    benchmarks = json.loads(benchmark_json_path.read_text())

    report = ["## Benchmark"]
    report.append(make_env())
    report.append(make_dataset())
    report.append(make_benchmark(benchmarks))
    print("\n".join(report))


if __name__ == "__main__":
    typer.run(main)
