#!/usr/bin/env python3
"""
Mock tuning script that generates randomized tuning results for testing backend integration.
This script mimics the full tuning infrastructure without actually running it.
"""

import argparse
import json
import random
import os
from pathlib import Path
import time
from typing import Dict, List, Any


def generate_gemm_hyperparams() -> Dict[str, Any]:
    """Generate randomized hyperparameters for GEMM kernels."""
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


def generate_conv_hyperparams() -> Dict[str, Any]:
    """Generate randomized hyperparameters for Conv kernels."""
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


def generate_attention_hyperparams() -> Dict[str, Any]:
    """Generate randomized hyperparameters for Attention kernels."""
    mfma_variants = ["F32_16x16x16_F16", "F32_32x32x8_F16", "F32_32x32x16_F16"]
    block_b_options = [1, 2, 4, 7, 8]
    block_m_options = [32, 64, 128, 256, 544]
    block_n_options = [32, 48, 64, 128]
    block_k2_options = [16, 32, 64, 128]

    return {
        "MFMA_VARIANT": [random.choice(mfma_variants), random.choice(mfma_variants)],
        "BLOCK_B": random.choice(block_b_options),
        "BLOCK_M": random.choice(block_m_options),
        "BLOCK_N": random.choice(block_n_options),
        "BLOCK_K2": random.choice(block_k2_options),
    }


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
        return f"conv_{random.randint(1, 1000)}_{random.randint(1, 1000)}_{problem.get('dtype', 'f16')}"

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


def generate_benchmark_result(
    kernel_spec: Dict[str, Any], backend: str, machine: str
) -> Dict[str, Any]:
    """Generate a complete benchmark result for a kernel specification."""
    kernel_type = kernel_spec["kernelType"].lower()
    problem = kernel_spec["problem"]

    # Generate randomized performance metrics
    mean_microseconds = round(random.uniform(10.0, 500.0), 1)
    improvement = random.choice([True, False])
    speedup = round(random.uniform(1.01, 2.99), 4) if improvement else 0

    # Generate hyperparameters based on kernel type
    if kernel_type == "gemm":
        hyperparams = generate_gemm_hyperparams()
        dims = ["M", "N", "K", "transpose", "dtype"]
    elif kernel_type == "conv":
        hyperparams = generate_conv_hyperparams()
        dims = ["N", "C", "H", "W", "dtype"]  # Simplified conv dims
    elif kernel_type == "attention":
        hyperparams = generate_attention_hyperparams()
        dims = ["B", "M", "N", "K1", "K2", "dtype"]
    else:
        hyperparams = {"UNKNOWN": "PARAM"}
        dims = ["dtype"]

    # Generate kernel name
    kernel_name = generate_kernel_name(kernel_type, problem)

    # Calculate performance metrics
    tflops = calculate_tflops(kernel_type, problem, mean_microseconds)
    arithmetic_intensity = calculate_arithmetic_intensity(kernel_type, problem)

    # Build the complete result structure
    result = {
        "name": kernel_name,
        "benchmark": {
            "machine": machine.upper(),
            "kernel_type": kernel_type,
            "backend": backend,
            "tag": kernel_spec.get("tag", "test"),
            "name": kernel_name,
            "dims": dims,
            "shape": problem.copy(),
            "problem": problem.copy(),
            "tuning_config": hyperparams,
            "mean_microseconds": mean_microseconds,
            "arithmetic_intensity": arithmetic_intensity,
            "tflops": tflops,
            "ok": True,
        },
        "improvement": improvement,
        "speedup": speedup,
        "hyperparams": hyperparams,
    }

    return {kernel_name: result}


def main():
    parser = argparse.ArgumentParser(
        description="Mock tuning script for testing backend integration"
    )
    parser.add_argument(
        "--load_problems",
        required=True,
        help="Path to input JSON file with kernel problems",
    )
    parser.add_argument(
        "--backend", required=True, help="Backend name (e.g., wave, rocwmma)"
    )
    parser.add_argument(
        "--machine", required=True, help="Machine name (e.g., mi300x, mi325x)"
    )

    args = parser.parse_args()

    time.sleep(30)

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

    # Group problems by kernel type
    results_by_type = {}

    for problem in problems:
        kernel_type = problem["kernelType"].lower()

        # Generate result for this problem
        result = generate_benchmark_result(problem, args.backend, args.machine)

        if kernel_type not in results_by_type:
            results_by_type[kernel_type] = {}

        results_by_type[kernel_type].update(result)

    # Create output directories and save results
    base_output_dir = Path("results/tuning")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for kernel_type, results in results_by_type.items():
        # Create subdirectory for kernel type
        kernel_dir = base_output_dir / kernel_type
        kernel_dir.mkdir(exist_ok=True)

        # Create output filename
        output_filename = f"{kernel_type}_{args.backend}_tuned_results.json"
        output_path = kernel_dir / output_filename

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        print(
            f"Generated {len(results)} results for {kernel_type} kernels -> {output_path}"
        )

    print(
        f"Mock tuning completed for backend '{args.backend}' on machine '{args.machine}'"
    )
    return 0


if __name__ == "__main__":
    exit(main())
