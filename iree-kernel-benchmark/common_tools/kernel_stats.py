# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import csv
import os
import re

from collections import defaultdict
from dataclasses import astuple, dataclass
from pathlib import Path


@dataclass
class IsaStats:
    instruction_count: int = -1
    vgpr_count: int = -1
    agpr_count: int = -1
    vgpr_spill_count: int = -1

    @staticmethod
    def get_csv_header() -> list[str]:
        return ["Instructions", "VGPRs", "AGPRs", "Spills"]


def calculate_isa_stats(kernel: Path) -> IsaStats:
    stats = IsaStats()
    with open(kernel.resolve(), "r", encoding="utf-8", errors="ignore") as file:
        for idx, line in enumerate(file):
            if "s_endpgm" in line:
                stats.instruction_count = idx
                continue
            if ".vgpr_count:" in line:
                stats.vgpr_count = int(line.split()[-1])
                continue
            if ".agpr_count:" in line:
                stats.agpr_count = int(line.split()[-1])
                continue
            if ".vgpr_spill_count:" in line:
                stats.vgpr_spill_count = int(line.split()[-1])
                continue

    return stats


@dataclass
class ConfiguredMlirStats:
    codegen_pipeline: str = None
    tile_sizes: list[int] = None
    workgroup_size: list[int] = None
    mfma_intrinsic: str = None

    @staticmethod
    def get_csv_header() -> list[str]:
        return ["Pipeline", "Tile Sizes", "Workgroup Sizes", "Intrinsic"]


def calculate_mlir_stats(mlir: Path) -> ConfiguredMlirStats:
    stats = ConfiguredMlirStats()
    list_re = r"\[((\d+,\s*)*(\d+)?)\]"
    with open(mlir.resolve(), "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if "#iree_codegen.lowering_config" in line:
                tile_sizes_re = rf"<tile_sizes = \[{list_re}\]>"
                match = re.search(tile_sizes_re, line)
                if match:
                    tile_sizes_str = match.group(1)
                    split_sizes = re.split(r",\s*", tile_sizes_str)
                    stats.tile_sizes = list(map(int, split_sizes))
                continue

            if "#iree_codegen.translation_info" in line:
                pipeline_re = rf"iree_codegen.translation_info<([a-zA-Z]+)\s+workgroup_size = {list_re}"
                match = re.search(pipeline_re, line)
                if match:
                    stats.codegen_pipeline = match.group(1)
                    workgroup_sizes_str = match.group(2)
                    split_workgroup_sizes = re.split(r",\s*", workgroup_sizes_str)
                    stats.workgroup_size = list(map(int, split_workgroup_sizes))

                mfma_re = r"#iree_gpu.mma_layout<([A-Z0-9_x]+)>"
                match = re.search(mfma_re, line)
                if match:
                    stats.mfma_intrinsic = match.group(1)
                continue
    return stats


@dataclass
class KernelStats:
    name: str = ""
    isa_stats: IsaStats = None
    mlir_stats: ConfiguredMlirStats = None

    @staticmethod
    def get_csv_header() -> list[str]:
        return (
            ["name"] + IsaStats.get_csv_header() + ConfiguredMlirStats.get_csv_header()
        )

    def get_values(self):
        return [self.name, *astuple(self.isa_stats), *astuple(self.mlir_stats)]


def process_directory(directory: Path) -> list[KernelStats]:
    """Search for .rocmasm and .mlir files and calculate the stats"""
    dir_to_result: dict[str, KernelStats] = defaultdict(KernelStats)
    for root, _dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            dir_name = Path(root).name
            if file_path.suffix == ".rocmasm":
                stats: IsaStats = calculate_isa_stats(file_path)
                dir_to_result[dir_name].name = dir_name
                dir_to_result[dir_name].isa_stats = stats
                continue
            if file_path.name.endswith("_benchmark.mlir"):
                stats: ConfiguredMlirStats = calculate_mlir_stats(file_path)
                dir_to_result[dir_name].name = dir_name
                dir_to_result[dir_name].mlir_stats = stats
                continue

    return list(dir_to_result.values())


def write_results_to_csv(results: list[KernelStats], output_file: Path) -> None:
    """Write the results to a CSV file."""
    results.sort(key=lambda x: x.name)
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(KernelStats.get_csv_header())
        for stats in results:
            csv_writer.writerow(stats.get_values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data collection tool targeting HSA dumps."
    )
    parser.add_argument(
        "dir",
        help="The directory from which to scan for ISA file dumps (.rocmasm).",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    output_file = Path("results/kernel_stats.csv")

    results: list[KernelStats] = process_directory(Path(args.dir).resolve())
    if results:
        write_results_to_csv(results, output_file)
        print(f"Results written to {output_file}\n")
    else:
        print("No .rocmasm files found.\n")
