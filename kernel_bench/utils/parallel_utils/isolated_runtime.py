import multiprocessing as mp
import pickle
import traceback
from typing import List, Tuple, Optional


def _subprocess_validate_wrapper(bench_pickle: bytes, device: str, queue: mp.Queue):
    """Worker function that runs in the subprocess"""
    try:
        # Unpickle the benchmark object
        bench = pickle.loads(bench_pickle)

        # Run the validation
        result = bench.validate_numerics(device)

        # Send back the result
        queue.put(("success", result))
    except Exception as e:
        # Send back the error with full traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        queue.put(("error", error_msg))


def isolated_validate_numerics(bench, device: str) -> Tuple[bool, Optional[str]]:
    """
    Run validate_numerics in an isolated subprocess to prevent GPU context corruption.

    Returns:
        (success, error_message) - success is True/False for validation result,
                                   error_message is None on success or contains error details
    """
    # Use spawn to ensure clean CUDA/HIP context
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    try:
        # Serialize the benchmark object
        bench_pickle = pickle.dumps(bench)
    except Exception as e:
        return False, f"Failed to serialize benchmark: {e}"

    # Create and start the subprocess
    process = ctx.Process(
        target=_subprocess_validate_wrapper, args=(bench_pickle, device, queue)
    )

    process.start()
    process.join()

    # Check if process crashed
    if process.exitcode != 0:
        return False, f"Subprocess crashed with exit code {process.exitcode}"

    # Get the result from the queue
    try:
        status, result = queue.get(timeout=1.0)  # 1 second timeout for queue
        if status == "success":
            return result, None
        else:
            return False, result
    except:
        return False, "Failed to retrieve result from subprocess"
