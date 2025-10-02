#!/usr/bin/env python3
"""
Mock benchmarking script that generates randomized benchmark results for testing backend integration.
This script mimics the full benchmarking infrastructure without actually running it.
"""

import argparse
import json
import random
import os
from pathlib import Path
import time
from typing import Dict, List, Any, Optional


def generate_default_hyperparams(kernel_type: str) -> Dict[str, Any]:
    """Generate default hyperparameters for different kernel types."""
    if kernel_type.lower() == "gemm":
        mfma_variants = ["F32_16x16x16_F16", "F32_32x32x8_F16", "F32_32x32x16_F16"]
        block_m_options = [16, 32, 64, 128, 256]
        block_n_options = [16, 32, 64, 128, 256]
        block_k_options = [16, 32, 64, 128]
        group_size_m_options = [1, 2, 4, 8, 16]

        return {
            "MFMA_VARIANT": random.choice(mfma_variants),
            "BLOCK_M": random.choice(block_m_options),
            "BLOCK_N": random.choice(block_n_options),
            "BLOCK_K": random.choice(block_k_options),
            "GROUP_SIZE_M": random.choice(group_size_m_options),
        }

    elif kernel_type.lower() == "conv":
        mfma_variants = ["F32_16x16x16_F16", "F32_32x32x8_F16", "F32_32x32x16_F16"]
        block_m_options = [16, 32, 64, 128, 176, 256]
        block_n_options = [16, 32, 64, 128, 152, 256]
        block_k_options = [16, 32, 64, 80, 128]
        elems_per_thread_options = [1, 2, 4, 8]

        return {
            "MFMA_VARIANT": random.choice(mfma_variants),
            "BLOCK_M": random.choice(block_m_options),
            "BLOCK_N": random.choice(block_n_options),
            "BLOCK_K": random.choice(block_k_options),
            "ELEMS_PER_THREAD": random.choice(elems_per_thread_options),
        }

    elif kernel_type.lower() == "attention":
        mfma_variants = [
            "F32_16x16x16_F16",
            "F32_32x32x8_F16",
            "F32_32x32x16_F16",
            "F32_32x32x16_K8_F16",
        ]
        return {
            "mfma_variant": [random.choice(mfma_variants), random.choice(mfma_variants)]
        }

    else:
        return {"UNKNOWN": "PARAM"}


def generate_kernel_name(kernel_type: str, problem: Dict[str, Any]) -> str:
    """Generate kernel name based on kernel type and problem specification."""
    if kernel_type.lower() == "gemm":
        m = problem.get("M", "1024")
        n = problem.get("N", "1024")
        k = problem.get("K", "1024")
        dtype = problem.get("dtype", "f16")
        transpose = problem.get("transpose", "NT")

        # Convert transpose format NT -> tB (N=no transpose, T=transpose B)
        ta = "N" if transpose[0] == "N" else "T"
        tb = "T" if transpose[1] == "T" else "N"
        transpose_suffix = f"t{transpose[1]}" if transpose[1] == "T" else ""

        return f"gemm_{m}_{n}_{k}_{dtype}_{dtype}_{transpose_suffix}"

    elif kernel_type.lower() == "conv":
        # Simplified conv naming - can be extended based on actual conv problem format
        n = problem.get("N", "1")
        c = problem.get("C", "256")
        h = problem.get("H", "32")
        w = problem.get("W", "32")
        dtype = problem.get("dtype", "f16")
        return f"conv_{n}_{c}_{h}_{w}_{dtype}"

    elif kernel_type.lower() == "attention":
        b = problem.get("B", 1)
        m = problem.get("M", 1024)
        n = problem.get("N", 64)
        k1 = problem.get("K1", 64)
        k2 = problem.get("K2", 1024)
        dtype = problem.get("dtype", "f16")

        return f"attention_bmnk1k2_{b}x{m}x{n}x{k1}x{k2}x{dtype}"

    else:
        return f"{kernel_type}_{random.randint(1000, 9999)}"


def calculate_tflops(
    kernel_type: str, problem: Dict[str, Any], microseconds: float
) -> float:
    """Calculate TFLOPS based on kernel type and problem size."""
    if kernel_type.lower() == "gemm":
        m = int(problem.get("M", 1024))
        n = int(problem.get("N", 1024))
        k = int(problem.get("K", 1024))
        ops = 2 * m * n * k  # 2 operations per element (multiply-add)

    elif kernel_type.lower() == "attention":
        b = int(problem.get("B", 1))
        m = int(problem.get("M", 1024))
        n = int(problem.get("N", 64))
        k1 = int(problem.get("K1", 64))
        k2 = int(problem.get("K2", 1024))
        # Simplified attention ops calculation
        ops = b * m * n * (k1 + k2) * 4

    else:
        # Default calculation for other kernel types
        ops = random.uniform(1e9, 1e12)

    seconds = microseconds / 1e6
    tflops = (ops / seconds) / 1e12
    return round(tflops, 4)


def calculate_arithmetic_intensity(kernel_type: str, problem: Dict[str, Any]) -> float:
    """Calculate arithmetic intensity based on kernel type and problem."""
    if kernel_type.lower() == "gemm":
        m = int(problem.get("M", 1024))
        n = int(problem.get("N", 1024))
        k = int(problem.get("K", 1024))
        # Simplified calculation: ops / memory_access
        ops = 2 * m * n * k
        memory = (m * k + k * n + m * n) * 2  # Assuming f16 (2 bytes)
        return round(ops / memory, 4)

    # For other kernel types, return a random reasonable value
    return round(random.uniform(50.0, 1000.0), 4)


def load_tuned_configs(tuned_configs_path: str) -> Dict[str, Dict[str, Any]]:
    """Load tuned configurations from JSON file."""
    try:
        with open(tuned_configs_path, "r") as f:
            tuned_data = json.load(f)

        # Extract tuned configs that have improvement=true
        improved_configs = {}
        for kernel_name, kernel_data in tuned_data.items():
            if kernel_data.get("improvement", False):
                improved_configs[kernel_name] = kernel_data.get("hyperparams", {})

        return improved_configs
    except Exception as e:
        print(f"Warning: Could not load tuned configs from {tuned_configs_path}: {e}")
        return {}


def get_tuning_config(
    kernel_name: str,
    kernel_type: str,
    tuned_configs: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Get tuning configuration for a kernel, using tuned config if available."""
    if tuned_configs and kernel_name in tuned_configs:
        return tuned_configs[kernel_name]
    else:
        return None


def generate_benchmark_result(
    kernel_spec: Dict[str, Any],
    backend: str,
    machine: str,
    tuned_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a complete benchmark result for a kernel specification."""
    kernel_type = kernel_spec["kernelType"].lower()
    problem = kernel_spec["problem"]

    # Generate kernel name
    kernel_name = generate_kernel_name(kernel_type, problem)

    # Generate randomized performance metrics
    mean_microseconds = round(random.uniform(10.0, 500.0), 1)
    arithmetic_intensity = calculate_arithmetic_intensity(kernel_type, problem)
    tflops = calculate_tflops(kernel_type, problem, mean_microseconds)

    # Get tuning configuration (from tuned configs if available, otherwise default)
    tuning_config = get_tuning_config(kernel_name, kernel_type, tuned_configs)

    # Set up dimensions based on kernel type
    if kernel_type == "gemm":
        dims = ["M", "N", "K", "tA", "tB", "dtype"]
        # Convert problem format to shape format for GEMM
        shape = {
            "M": int(problem.get("M", 1024)),
            "N": int(problem.get("N", 1024)),
            "K": int(problem.get("K", 1024)),
            "tA": "N" if problem.get("transpose", "NT")[0] == "N" else "T",
            "tB": "T" if problem.get("transpose", "NT")[1] == "T" else "N",
            "dtype": problem.get("dtype", "f16"),
        }
    elif kernel_type == "attention":
        dims = ["B", "M", "N", "K1", "K2", "dtype"]
        shape = {
            "B": int(problem.get("B", 1)),
            "M": int(problem.get("M", 1024)),
            "N": int(problem.get("N", 64)),
            "K1": int(problem.get("K1", 64)),
            "K2": int(problem.get("K2", 1024)),
            "dtype": problem.get("dtype", "f16"),
        }
    elif kernel_type == "conv":
        dims = ["N", "C", "H", "W", "dtype"]
        shape = {
            "N": int(problem.get("N", 1)),
            "C": int(problem.get("C", 256)),
            "H": int(problem.get("H", 32)),
            "W": int(problem.get("W", 32)),
            "dtype": problem.get("dtype", "f16"),
        }
    else:
        dims = ["dtype"]
        shape = {"dtype": problem.get("dtype", "f16")}

    # Build the benchmark result structure
    result = {
        "machine": machine.upper(),
        "kernel_type": kernel_type,
        "backend": backend,
        "tag": kernel_spec.get("tag", "test"),
        "name": kernel_name,
        "dims": dims,
        "shape": shape,
        "problem": {},  # Empty as requested
        "tuning_config": tuning_config,
        "mean_microseconds": mean_microseconds,
        "arithmetic_intensity": arithmetic_intensity,
        "tflops": tflops,
        "ok": True,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Mock benchmarking script for testing backend integration"
    )
    parser.add_argument(
        "--load_problems",
        required=True,
        help="Path to input JSON file with kernel problems",
    )
    parser.add_argument(
        "--machine", required=True, help="Machine name (e.g., mi300x, mi325x)"
    )
    parser.add_argument(
        "--load_tuned_configs", help="Optional path to tuned configurations JSON file"
    )

    args = parser.parse_args()

    # Load input problems
    try:
        with open(args.load_problems, "r") as f:
            problems = json.load(f)
    except Exception as e:
        print(f"Error loading problems file {args.load_problems}: {e}")
        return 1

    if not isinstance(problems, list):
        print("Error: Input JSON must be a list of kernel problems")
        return 1

    time.sleep(30)

    # Load tuned configurations if provided
    tuned_configs = None
    if args.load_tuned_configs:
        tuned_configs = load_tuned_configs(args.load_tuned_configs)
        print(f"Loaded {len(tuned_configs)} tuned configurations")

    # Determine unique backends from the problems (we'll use "wave" as default for now)
    # In a real scenario, this might be specified as an argument or inferred from the problems
    backends = ["wave"]  # Default backend for mock purposes

    # Group results by kernel type and backend
    results_by_type_and_backend = {}

    for problem in problems:
        kernel_type = problem["kernelType"].lower()

        for backend in backends:
            # Generate result for this problem
            result = generate_benchmark_result(
                problem, backend, args.machine, tuned_configs
            )

            if kernel_type not in results_by_type_and_backend:
                results_by_type_and_backend[kernel_type] = {}

            if backend not in results_by_type_and_backend[kernel_type]:
                results_by_type_and_backend[kernel_type][backend] = []

            results_by_type_and_backend[kernel_type][backend].append(result)

    # Create output directories and save results
    base_output_dir = Path("results/json")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for kernel_type, backends_data in results_by_type_and_backend.items():
        # Create subdirectory for kernel type
        kernel_dir = base_output_dir / kernel_type
        kernel_dir.mkdir(exist_ok=True)

        for backend, results in backends_data.items():
            # Create subdirectory for backend (if needed, but typically files are directly in kernel_type dir)
            # output_filename = f"{kernel_type}_{backend}.json"
            # Actually, based on the existing structure, files are directly in the kernel_type directory
            output_filename = f"{kernel_type}_{backend}.json"
            output_path = kernel_dir / output_filename

            # Save results as a list
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)

            print(
                f"Generated {len(results)} benchmark results for {kernel_type}/{backend} -> {output_path}"
            )

    print(f"Mock benchmarking completed for machine '{args.machine}'")
    if tuned_configs:
        print(f"Used tuned configurations from {args.load_tuned_configs}")

    return 0


if __name__ == "__main__":
    exit(main())
