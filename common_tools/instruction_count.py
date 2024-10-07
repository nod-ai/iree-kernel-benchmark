import os
import argparse
import csv
from typing import Dict


def count_instr_in_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for idx, line in enumerate(file):
            if "s_endpgm" in line:
                return idx
    return -1


def get_metadata_dict(file_path, keys) -> Dict:
    metadata = dict()
    start = None
    end = None
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for idx, line in enumerate(file):
            if ".end_amdgpu_metadata" in line:
                end = idx
            elif ".amdgpu_metadata" in line:
                start = idx
            elif start and not end:
                f_line = line.lstrip(" -.").rstrip(" \n")
                key_end = f_line.find(":")
                key = f_line[0:key_end]
                if not key in keys:
                    continue
                value = f_line[key_end + 1 :].lstrip(" ")
                if key not in metadata.keys():
                    metadata[key] = value
                else:
                    if isinstance(metadata[key], list):
                        metadata[key].append(value)
                    else:
                        metadata[key] = [metadata[key], value]
    return metadata


def search_directory(directory):
    """Search for .rocmasm files and count their lines."""
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".rocmasm"):
                file_path = os.path.join(root, file)
                line_count = count_instr_in_file(file_path)
                results.append((file_path, line_count))
    return results


def write_results_to_csv(results, output_file, metadata_items):
    """Write the results to a CSV file."""
    # Sort results by line count (second item in tuple)
    results.sort(key=lambda x: x[1], reverse=True)
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = ["Filename", "Instruction Count"]
        for item in metadata_items:
            headers.append(item)
        csv_writer.writerow(headers)
        csv_writer.writerows(results)


if __name__ == "__main__":
    default_metadata = ["agpr_count", "vgpr_count", "vgpr_spill_count"]
    parser = argparse.ArgumentParser(
        description="Data collection tool targeting HSA dumps."
    )
    parser.add_argument(
        "dir",
        help="The directory from which to scan for ISA file dumps (.rocmasm).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--metadata",
        nargs="*",
        default=default_metadata,
        help="Manually specify which metadata items to extract from ISA files.",
    )
    args = parser.parse_args()
    output_file = "rocmasm_data.csv"

    results = search_directory(args.dir)
    for i, r in enumerate(results):
        assert len(r) == 2
        f = r[0]
        metadata_dict = get_metadata_dict(f, args.metadata)
        r = [f, r[1]]
        for d in args.metadata:
            r.append(metadata_dict[d])
        results[i] = tuple(r)

    if results:
        write_results_to_csv(results, output_file, args.metadata)
        print(f"Results written to {output_file}\n")
    else:
        print("No .rocmasm files found.\n")
