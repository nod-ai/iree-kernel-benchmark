import os
from pathlib import Path
import argparse


def compile_dumps(mlir_dir: str):
    dumps_dir = Path(args.dir).parent.joinpath("dumps")
    i_o_paths = []
    for root, _, files in os.walk(mlir_dir):
        for file in files:
            if not file.endswith(".mlir") and not file.endswith(".mlirbc"):
                continue
            f_path = Path(root).joinpath(file)
            d_path = str(dumps_dir.joinpath(f_path.stem))
            os.makedirs(d_path, exist_ok=True)
            i_o_paths.append((str(f_path), d_path))
    num_jobs = len(i_o_paths)
    for job, (f_path, d_path) in enumerate(i_o_paths):
        print(f"Compiling {job} of {num_jobs}...", end="\r")
        script = f"iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 {f_path} -o {os.path.join(d_path, 'gemm.vmfb')} --iree-hal-dump-executable-files-to={d_path}"
        os.system(script)
    print(
        f"All jobs completed. Check for dumps in {Path(dumps_dir).absolute()}"
        + 20 * " ",
        end="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates IREE compilation executable dumps for all mlir files in a directory."
    )
    parser.add_argument(
        "dir",
        help="The directory from which to scan for mlir files.",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    compile_dumps(args.dir)
