import os
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import torch
from torch.testing import assert_close
from kernel_bench.utils.print_utils import get_logger


def get_backend_outputs(outputs_dir: Path) -> Dict[str, Set[str]]:
    """
    Scan outputs directory and collect available config files per backend.

    Returns:
        Dictionary mapping backend name to set of config names (without .pt extension)
    """
    backend_outputs = {}

    if not outputs_dir.exists():
        return backend_outputs

    for backend_dir in outputs_dir.iterdir():
        if not backend_dir.is_dir():
            continue

        backend_name = backend_dir.name
        config_names = set()

        for output_file in backend_dir.glob("*.pt"):
            # Remove .pt extension to get config name
            config_name = output_file.stem
            config_names.add(config_name)

        if config_names:
            backend_outputs[backend_name] = config_names

    return backend_outputs


def find_common_configs(backend_outputs: Dict[str, Set[str]]) -> Set[str]:
    """
    Find configs that are common across all backends.

    Args:
        backend_outputs: Dictionary mapping backend to set of config names

    Returns:
        Set of config names present in all backends
    """
    if not backend_outputs:
        return set()

    # Start with the first backend's configs
    common_configs = set(list(backend_outputs.values())[0])

    # Intersect with all other backends
    for configs in backend_outputs.values():
        common_configs = common_configs.intersection(configs)

    return common_configs


def load_tensor(backend: str, config_name: str, outputs_dir: Path) -> torch.Tensor:
    """Load a tensor from the outputs directory."""
    tensor_path = outputs_dir / backend / f"{config_name}.pt"
    return torch.load(tensor_path, weights_only=False)


def get_tolerance_for_dtype(dtype: torch.dtype) -> Tuple[float, float]:
    """
    Get appropriate tolerances based on dtype.

    Returns:
        Tuple of (atol, rtol)
    """
    if dtype in [torch.float32, torch.float]:
        return 1e-4, 1e-5
    elif dtype in [torch.float16, torch.half]:
        return 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        return 1e-2, 1e-2
    elif dtype in [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]:
        return 3e-2, 5e-2
    else:
        # Default tolerances
        return 1e-3, 1e-3


def validate_config(
    config_name: str,
    backends: List[str],
    outputs_dir: Path,
    reference_backend: str = None,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Validate that outputs for a config match across backends.

    Args:
        config_name: Name of the config to validate
        backends: List of backend names
        outputs_dir: Path to outputs directory
        reference_backend: Backend to use as reference (default: first backend)
        verbose: Whether to print detailed comparison info

    Returns:
        Tuple of (success, error_message)
    """
    logger = get_logger()

    if reference_backend is None:
        reference_backend = backends[0]

    # Load reference tensor
    try:
        ref_tensor = load_tensor(reference_backend, config_name, outputs_dir)
    except Exception as e:
        return False, f"Failed to load reference tensor from {reference_backend}: {e}"

    # Get tolerances based on dtype
    atol, rtol = get_tolerance_for_dtype(ref_tensor.dtype)

    if verbose:
        logger.info(
            f"Reference tensor shape: {ref_tensor.shape}, dtype: {ref_tensor.dtype}"
        )
        logger.info(f"Using tolerances: atol={atol}, rtol={rtol}")

    # Compare against other backends
    for backend in backends:
        if backend == reference_backend:
            continue

        try:
            tensor = load_tensor(backend, config_name, outputs_dir)
        except Exception as e:
            return False, f"Failed to load tensor from {backend}: {e}"

        # Check shapes match
        if tensor.shape != ref_tensor.shape:
            return (
                False,
                f"Shape mismatch: {backend} has {tensor.shape}, reference has {ref_tensor.shape}",
            )

        # Compare tensors
        try:
            assert_close(
                tensor,
                ref_tensor,
                atol=atol,
                rtol=rtol,
                check_dtype=False,
                check_device=False,
            )
            if verbose:
                logger.info(f"✓ {backend} matches reference")
        except AssertionError as e:
            error_msg = f"Numerical mismatch between {backend} and {reference_backend}: {str(e)}"
            return False, error_msg

    return True, ""


def main():
    parser = argparse.ArgumentParser(
        description="Validate numerical accuracy of saved kernel outputs across backends."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="results/outputs",
        help="Path to outputs directory (default: results/outputs)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="Comma-separated list of backends to validate (default: all available backends)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Backend to use as reference (default: first backend alphabetically)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Specific config to validate (default: validate all common configs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed comparison information",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue validation even if errors are found",
    )

    args = parser.parse_args()
    logger = get_logger()

    outputs_dir = Path(args.outputs_dir)

    if not outputs_dir.exists():
        logger.error(f"Outputs directory does not exist: {outputs_dir}")
        return 1

    # Get all backend outputs
    backend_outputs = get_backend_outputs(outputs_dir)

    if not backend_outputs:
        logger.error(f"No backend outputs found in {outputs_dir}")
        return 1

    # Filter backends if specified
    if args.backends:
        specified_backends = set(b.strip() for b in args.backends.split(","))
        backend_outputs = {
            k: v for k, v in backend_outputs.items() if k in specified_backends
        }

        if not backend_outputs:
            logger.error(f"None of the specified backends found in {outputs_dir}")
            return 1

    available_backends = sorted(backend_outputs.keys())
    logger.info(f"Found backends: {', '.join(available_backends)}")

    for backend, configs in backend_outputs.items():
        logger.info(f"  {backend}: {len(configs)} configs")

    # Determine reference backend
    reference_backend = args.reference
    if reference_backend:
        if reference_backend not in available_backends:
            logger.error(f"Reference backend '{reference_backend}' not found")
            return 1
    else:
        # Use torch as reference if available, otherwise first alphabetically
        if "torch" in available_backends:
            reference_backend = "torch"
        else:
            reference_backend = available_backends[0]

    logger.info(f"Using '{reference_backend}' as reference backend")

    # Find common configs
    if args.config:
        # Validate specific config
        configs_to_validate = [args.config]

        # Check if config exists in all backends
        for backend in available_backends:
            if args.config not in backend_outputs[backend]:
                logger.error(f"Config '{args.config}' not found in backend '{backend}'")
                return 1
    else:
        # Validate all common configs
        configs_to_validate = sorted(find_common_configs(backend_outputs))

        if not configs_to_validate:
            logger.error("No common configs found across all backends")
            return 1

        logger.info(
            f"Found {len(configs_to_validate)} common configs across all backends"
        )

    # Validate configs
    passed = 0
    failed = 0
    errors = []

    for i, config_name in enumerate(configs_to_validate, 1):
        if not args.verbose:
            logger.info(f"[{i}/{len(configs_to_validate)}] Validating {config_name}...")
        else:
            logger.info(f"\n{'='*80}")
            logger.info(f"[{i}/{len(configs_to_validate)}] Validating {config_name}")
            logger.info(f"{'='*80}")

        success, error_msg = validate_config(
            config_name,
            available_backends,
            outputs_dir,
            reference_backend,
            args.verbose,
        )

        if success:
            passed += 1
            logger.info(f"  ✓ PASSED")
        else:
            failed += 1
            logger.error(f"  ✗ FAILED: {error_msg}")
            errors.append((config_name, error_msg))

            if not args.continue_on_error:
                logger.error(
                    "Stopping validation due to failure (use --continue_on_error to continue)"
                )
                break

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total configs validated: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

    if errors:
        logger.info(f"\nFailed configs:")
        for config_name, error_msg in errors:
            logger.error(f"  - {config_name}")
            if args.verbose:
                logger.error(f"    {error_msg}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
