import argparse
import csv
import os

from dataclasses import astuple, dataclass
from pathlib import Path

@dataclass
class IsaStats:
    instruction_count: int = -1
    vgpr_count: int = -1
    agpr_count: int = -1
    vgpr_spill_count: int = -1


def calculate_isa_stats(kernel: Path):
    stats = IsaStats()
    with open(kernel.resolve(), 'r', encoding='utf-8', errors='ignore') as file:
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


def process_directory(directory: Path) -> list[IsaStats]:
    """Search for .rocmasm files and count their lines."""
    results: list[IsaStats] = []
    for root, _dirs, files in os.walk(directory.resolve()):
        for file in files:
            if file.endswith('.rocmasm'):
                file_path = Path(root) / file
                line_count: IsaStats = calculate_isa_stats(file_path)
                results.append((file_path, line_count))
    return results


def write_results_to_csv(results: list[IsaStats], output_file: Path):
    """Write the results to a CSV file."""
    # Sort results by line count (second item in tuple)
    results.sort(key=lambda x: x[0])
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'Instruction Count', 'VGPRs', 'AGPRs', 'Spills'])
        for file_path, stats in results:
            csv_writer.writerow([file_path, *astuple(stats)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data collection tool targeting HSA dumps.")
    parser.add_argument("dir", help="The directory from which to scan for ISA file dumps (.rocmasm).", type=str, default=None)
    args = parser.parse_args()
    output_file = Path('kernel_stats.csv')

    results = process_directory(Path(args.dir).resolve())
    if results:
        write_results_to_csv(results, output_file)
        print(f"Results written to {output_file}\n")
    else:
        print("No .rocmasm files found.\n")
