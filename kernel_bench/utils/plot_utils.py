import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from kernel_bench.utils.print_utils import get_logger


def load_results(kernel_type: str, backend: str) -> List[Dict]:
    """Load benchmark results from JSON file."""
    results_path = Path(f"results/json/{kernel_type}/{kernel_type}_{backend}.json")

    if not results_path.exists():
        return []

    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        get_logger().warning(f"Failed to load {results_path}: {e}")
        return []


def load_all_backend_results(
    backend_names: list[str], kernel_type: str, backend: str, machine: str
):
    logger = get_logger()
    backend_results = {}
    for backend in backend_names:
        results = load_results(kernel_type, backend)
        if not results:
            logger.warning(
                f"No results found for backend '{backend}' and kernel type '{kernel_type}'"
            )
            continue

        # Filter by machine
        filtered_results = filter_results_by_machine(results, machine)
        if not filtered_results:
            logger.warning(
                f"No results found for backend '{backend}' on machine '{machine}'"
            )
            continue

        backend_results[backend] = filtered_results
        logger.info(
            f"Loaded {len(filtered_results)} results for backend '{backend}' on machine '{machine}'"
        )

    if not backend_results:
        logger.error("No results loaded for any backend. Exiting.")
        return {}

    return backend_results


def shape_to_string(shape: Dict) -> str:
    """Convert shape dictionary to string representation."""
    # Sort keys for consistent ordering, but prioritize common dimension names
    priority_keys = ["M", "N", "K", "B", "H", "H_KV", "N_Q", "N_KV", "D_Q", "D_KV"]

    # Get all keys
    all_keys = list(shape.keys())

    # Separate priority keys (in order) and remaining keys
    ordered_keys = [k for k in priority_keys if k in all_keys]
    remaining_keys = sorted(
        [
            k
            for k in all_keys
            if k not in priority_keys and k != "dtype" and k != "transpose"
        ]
    )

    # Combine: priority keys first, then remaining, then dtype/transpose at the end
    final_keys = ordered_keys + remaining_keys

    # Add dtype and transpose at the end if present
    if "dtype" in shape:
        final_keys.append("dtype")
    if "transpose" in shape:
        final_keys.append("transpose")

    # Build string representation
    parts = [str(shape[k]) for k in final_keys]
    return "x".join(parts)


def filter_results_by_machine(results: List[Dict], machine: str) -> List[Dict]:
    """Filter results to only include specified machine."""
    return [r for r in results if r.get("machine", "").upper() == machine.upper()]


def get_common_kernels(
    backend_results: Dict[str, List[Dict]], use_tag: bool = False
) -> Set[str]:
    """Find kernels common across all backends."""
    if not backend_results:
        return set()

    # Get key for each result (either tag or shape string)
    def get_key(result: Dict) -> str:
        if use_tag:
            return result.get("tag", "")
        else:
            return shape_to_string(result.get("shape", {}))

    # Get sets of keys for each backend
    backend_keys = {
        backend: set(get_key(r) for r in results if r.get("ok", False))
        for backend, results in backend_results.items()
    }

    # Find intersection
    common_keys = set.intersection(*backend_keys.values()) if backend_keys else set()
    return common_keys


def create_comparison_plot(
    backend_results: Dict[str, List[Dict]],
    kernel_type: str,
    machine: str,
    backend_names: List[str],
    save_path: str,
    use_tag: bool = False,
):
    """Create comparison plot across backends."""
    logger = get_logger()

    if not backend_results:
        logger.error("No results to plot")
        return

    # Find common kernels
    common_keys = get_common_kernels(backend_results, use_tag)

    if not common_keys:
        logger.error("No common kernels found across all backends")
        return

    logger.info(f"Found {len(common_keys)} common kernels across backends")

    # Build data dictionary: {kernel_key: {backend: tflops}}
    data = {key: {} for key in common_keys}

    def get_key(result: Dict) -> str:
        if use_tag:
            return result.get("tag", "")
        else:
            return shape_to_string(result.get("shape", {}))

    for backend, results in backend_results.items():
        for result in results:
            if not result.get("ok", False):
                continue
            key = get_key(result)
            if key in common_keys:
                data[key][backend] = result.get("tflops", 0)

    # Order backends as provided
    ordered_backends = [b for b in backend_names if b in backend_results]

    # Sort kernel keys for consistent display
    ordered_keys = sorted(common_keys)

    # Set up plotting
    sns.set_palette("colorblind")
    num_backends = len(ordered_backends)

    x = np.arange(len(ordered_keys))
    width = 0.8 / num_backends
    fig, ax = plt.subplots(figsize=(max(12, len(ordered_keys) * 0.8), 8))
    plt.rcParams["font.size"] = 14

    # Create bars for each backend
    bars_dict = {}
    colors = plt.cm.Set1(np.linspace(0, 1, num_backends))

    for i, backend in enumerate(ordered_backends):
        backend_tflops = []
        for key in ordered_keys:
            backend_tflops.append(data[key].get(backend, 0))

        offset = (i - (num_backends - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            backend_tflops,
            width,
            label=backend.capitalize(),
            color=colors[i],
        )
        bars_dict[backend] = bars

    # Customize plot
    xlabel = "Tag" if use_tag else "Problem Configuration"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("TFLOPs", fontsize=14)
    ax.set_title(
        f"{kernel_type.replace('_', ' ').title()} Performance Comparison ({machine.upper()})",
        fontsize=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_keys, rotation=45, ha="right")
    ax.legend()

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    for bars in bars_dict.values():
        autolabel(bars)

    # Add grid
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Comparison plot saved to {save_path}")
    plt.close()
