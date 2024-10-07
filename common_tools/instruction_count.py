import os
import argparse
import csv

def count_instr_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for idx, line in enumerate(file):
            if "s_endpgm" in line:
               return idx
    return -1

def search_directory(directory):
    """Search for .rocmasm files and count their lines."""
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.rocmasm'):
                file_path = os.path.join(root, file)
                line_count = count_instr_in_file(file_path)
                results.append((file_path, line_count))
    return results

def write_results_to_csv(results, output_file):
    """Write the results to a CSV file."""
    # Sort results by line count (second item in tuple)
    results.sort(key=lambda x: x[1], reverse=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'Instruction Count'])
        csv_writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data collection tool targeting HSA dumps.")
    parser.add_argument("dir", help="The directory from which to scan for ISA file dumps (.rocmasm).", type=str, default=None)
    args = parser.parse_args()
    output_file = 'rocmasm_instr_counts.csv'

    results = search_directory(args.dir)

    if results:
        write_results_to_csv(results, output_file)
        print(f"Results written to {output_file}\n")
    else:
        print("No .rocmasm files found.\n")
