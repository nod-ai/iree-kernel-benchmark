import argparse
import csv
import glob
import json
import os
from typing import Optional
import pandas as pd


def create_sizes_lookup(sizes_path: str):
    sizes_path = os.path.dirname(sizes_path)
    size_cat_files = glob.glob(f"{sizes_path}/sizes_cat*.txt")

    size_to_cat = {}

    for size_file in size_cat_files:
        category = size_file.split("sizes_")[1].split(".txt")[0]
        with open(size_file, "r") as f:
            for line in f.read().splitlines():
                try:
                    M, N, B, K = [int(x.strip()) for x in line.strip("[]").split(",")]
                    shape = (M, N, K)
                    size_to_cat[shape] = category
                except:
                    print(f"|{line}|", line.strip("[]"))

    return size_to_cat


def parse_csv(csv_path: str, sizes_path: str, title: Optional[str] = None):
    df = pd.read_csv(csv_path)
    size_to_cat = create_sizes_lookup(sizes_path)

    parsed_shapes = set()

    results = []
    for i, result in enumerate(df.iloc):
        m, n, k = int(result["m"]), int(result["n"]), int(result["k"])
        shape = (m, n, k)
        tA = "T"
        tB = "N"
        dtype = "bf16"

        if shape not in size_to_cat:
            continue

        assert shape not in parsed_shapes, "Duplicate shape found in csv"
        parsed_shapes.add(shape)

        runtime_us = float(result["us"])
        byte_count = (m + n) * k * 2
        flops = 2 * m * n * k
        arithmetic_intensity = flops / byte_count
        tflops = (flops / 1e12) / (runtime_us / 1e6)

        results.append(
            {
                "index": i,
                "machine": "MI350X",
                "kernel_type": "gemm",
                "backend": title or "hipblaslt",
                "tag": size_to_cat[shape],
                "name": f"gemm_{m}_{n}_{k}_{dtype}_{tA}{tB}",
                "shape": {
                    "M": m,
                    "N": n,
                    "K": k,
                    "tA": tA,
                    "tB": tB,
                    "dtype": dtype,
                },
                "problem": {
                    "M": m,
                    "N": n,
                    "K": k,
                    "tA": tA,
                    "tB": tB,
                    "dtype": dtype,
                },
                "tuning_config": None,
                "mean_microseconds": runtime_us,
                "arithmetic_intensity": arithmetic_intensity,
                "tflops": tflops,
                "ok": True,
            }
        )

    return results


def save_to_json(json_data, output_path):
    print(json_data[0])
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Convert hipBLASLt CSV results to JSON format"
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        default="tuned_hipblaslt_results.csv",
        help="Path to input CSV file (default: tuned_hipblaslt_results.csv)",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="gemm_hipblaslt_tuned.json",
        help="Path to output JSON file (default: gemm_hipblaslt_tuned.json)",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="hipblaslt-tuned",
        help="Backend title for the results (default: hipblaslt-tuned)",
    )

    parser.add_argument(
        "--sizes-path",
        type=str,
        default="Sizes_BF16",
        help="Path to sizes directory (default: Sizes_BF16)",
    )

    args = parser.parse_args()

    result_data = parse_csv(args.input_csv, args.sizes_path, args.title)
    save_to_json(result_data, args.output_json)

    print(f"Successfully converted {args.input_csv} to {args.output_json}")
    print(f"Processed {len(result_data)} results")


if __name__ == "__main__":
    main()
