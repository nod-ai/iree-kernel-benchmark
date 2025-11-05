import os
import argparse
from kernel_bench.utils.plot_utils import *
from kernel_bench.utils.print_utils import get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from existing benchmark results."
    )
    parser.add_argument(
        "--kernel_type",
        type=str,
        required=True,
        help="Kernel type (e.g., gemm, vanilla_attention, bshd_attention, extend_attention, conv)",
    )
    parser.add_argument(
        "--machine",
        type=str,
        required=True,
        help="Machine to filter results by (e.g., mi300x, mi325x)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Backend(s) to compare (comma-separated, e.g., wave,torch,triton)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for the plot (e.g., comparison.png)",
    )
    parser.add_argument(
        "--use_tag",
        action="store_true",
        help="Use tag names on x-axis instead of shape configurations",
    )

    args = parser.parse_args()

    # Parse backends
    backend_names = [b.strip() for b in args.backend.split(",")]

    # Load results for each backend
    backend_results = load_all_backend_results(
        backend_names, args.kernel_type, args.backend, args.machine
    )

    # Create comparison plot
    create_comparison_plot(
        backend_results=backend_results,
        kernel_type=args.kernel_type,
        machine=args.machine,
        backend_names=backend_names,
        save_path=args.out,
        use_tag=args.use_tag,
    )


if __name__ == "__main__":
    main()
