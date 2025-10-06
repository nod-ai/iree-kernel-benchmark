from typing import List, Dict


def compare_kernels(
    old_kernels: List[Dict], new_kernels: List[Dict] = None
) -> tuple[dict, dict]:
    if new_kernels is None:
        new_kernels = []

    def create_kernel_lookup(kernels: List[Dict]) -> Dict:
        """Create lookup dictionary for kernels by unique identifier (tag + name)"""
        lookup = {}
        for kernel in kernels:
            key = (kernel["tag"], kernel["name"])
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(kernel)
        return lookup

    def group_and_average_kernels(kernel_lookup: Dict, common_keys: set) -> Dict:
        """Group kernels by machine/backend/kernel_type and calculate averages"""
        grouped = {}

        # Group kernels by machine, backend, and kernel_type
        for key in common_keys:
            for kernel in kernel_lookup[key]:
                machine_name = kernel["machine"]
                backend_name = kernel["backend"]
                kernel_type = kernel["kernel_type"]

                if machine_name not in grouped:
                    grouped[machine_name] = {}
                if backend_name not in grouped[machine_name]:
                    grouped[machine_name][backend_name] = {}
                if kernel_type not in grouped[machine_name][backend_name]:
                    grouped[machine_name][backend_name][kernel_type] = []

                grouped[machine_name][backend_name][kernel_type].append(kernel)

        # Calculate averages
        stats = {}
        for machine_name, backends in grouped.items():
            stats[machine_name] = {}
            for backend_name, kernel_types in backends.items():
                stats[machine_name][backend_name] = {}
                for kernel_type, kernels in kernel_types.items():
                    # Filter out kernels with ok=False or invalid data
                    valid_kernels = [
                        k
                        for k in kernels
                        if k.get("ok", False) and k.get("tflops", 0) > 0
                    ]

                    if valid_kernels:
                        avg_tflops = sum(k["tflops"] for k in valid_kernels) / len(
                            valid_kernels
                        )
                        avg_runtime = sum(
                            k["mean_microseconds"] for k in valid_kernels
                        ) / len(valid_kernels)
                    else:
                        avg_tflops = 0.0
                        avg_runtime = 0.0

                    stats[machine_name][backend_name][kernel_type] = {
                        "tflops": avg_tflops,
                        "runtime": avg_runtime,
                    }

        return stats

    # Create lookups and find common kernels
    old_kernel_lookup = create_kernel_lookup(old_kernels)
    new_kernel_lookup = create_kernel_lookup(new_kernels)
    common_keys = set(old_kernel_lookup.keys()) & set(new_kernel_lookup.keys())

    # Calculate stats for both old and new kernels
    old_stats = group_and_average_kernels(old_kernel_lookup, common_keys)
    new_stats = group_and_average_kernels(new_kernel_lookup, common_keys)

    return old_stats, new_stats
