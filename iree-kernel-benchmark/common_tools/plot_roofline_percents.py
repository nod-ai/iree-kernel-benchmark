import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_roofline_vs_column(kernel_stat_path, benchmark_stat_path, out_path, param_name, boxplot):
    kernel_df = pd.read_csv(kernel_stat_path)
    benchmark_df = pd.read_csv(benchmark_stat_path)
    if param_name not in kernel_df.columns and param_name not in benchmark_df.columns:
        print(f"`{param_name}` column not found in {kernel_stat_path} or {benchmark_stat_path}.\n")
        return False
    if "roofline_percent" not in benchmark_df.columns:
        print(f"`roofline_percent` column not found in {benchmark_stat_path}.\n")
        return False
    if "name" not in benchmark_df.columns or "name" not in kernel_df.columns:
        print(f"`name` column not found in {kernel_stat_path} and {benchmark_stat_path}.\n")
        return False
    df = kernel_df.merge(benchmark_df, on="name")
    if boxplot:
        axes = df[[param_name, "roofline_percent"]].boxplot(
            by=param_name,
            figsize=(12,12)
        )
    else:
        axes = df.plot(
            param_name,
            "roofline_percent",
            kind="scatter",
            figsize=(12,12)
        )
    plt.xlabel(param_name)
    plt.ylabel("roofline_percent")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotting tool to correlate kernel parameters with roofline percentages."
    )
    parser.add_argument(
        "--kernel_stats_csv",
        help="The path to the input csv containing kernel metrics.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--benchmark_csv",
        help="The path to the input csv containing benchmarks.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--out_path",
        help="The path to save the resulting plot image.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--parameter",
        help="The name of the column with the parameter to use as the x-axis.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--boxplot",
        help="Use a boxplot graph, with one boxplot per parameter value.",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False
    )
    args = parser.parse_args()

    succeeded = plot_roofline_vs_column(
        args.kernel_stats_csv, args.benchmark_csv, args.out_path, args.parameter, args.boxplot
    )
    if succeeded:
        print(f"Plot saved to {args.out_path}\n")
    else:
        print(f"Failed to generate plot.\n")
