import os
import tempfile
import numpy as np
from typing import List
from multiprocessing import Pool, cpu_count, Manager
import argparse
from utils import *

def compile_attention_mlir_file(mlir_file : str, target_device: str, hip_target: str, extra_flags: list):
    vmfb_file = os.path.splitext(mlir_file)[0] + ".vmfb"
    exec_args = [
    "iree-compile",
    # Input file
    f"{mlir_file}",
    # Output file
    "-o",
    f"{vmfb_file}",
    # Target Device: hip
    f"--iree-hal-target-device={target_device}",
    # Device: MI300x
    f"--iree-hip-target={hip_target}",
    ] + extra_flags
    ret_value, stdout, stderr = run_iree_command(exec_args)
    if ret_value == 0:
        print(f"Successfully compiled {mlir_file} to {vmfb_file}")
    else:
        print(f"Failed to compile {mlir_file}")
        return None
    return vmfb_file

def get_output_npy(vmfb_file : str, inputs : List[str], target_device: str):
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as out:
        exec_args = [
            "iree-run-module",
            f"--device={target_device}",
            "--device_allocator=caching",
            f"--module={vmfb_file}",
            "--function=main",
            f"--input=@{inputs[0].name}",
            f"--input=@{inputs[1].name}",
            f"--input=@{inputs[2].name}",
            f"--output=@{out.name}"
        ]
        ret_value, stdout, stderr = run_iree_command(exec_args)
        if ret_value == 0:
            print(f"Successfully ran {vmfb_file}")
            return out
        else:
            print(f"Failed to run {vmfb_file}")
            print(stderr.decode("utf-8"))
            out.close()
            return None


def create_test_inputs(shapes : List[List[int]]):
    inputfiles = list()
    for shape in shapes:
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp:
            arr = np.random.randn(*shape).astype(np.float16)
            np.save(temp, arr)
            inputfiles.append(temp)
    return inputfiles

def get_shapes(shape_strs : List[List[str]]) -> List[List[int]]:
    shapes = list()
    for shape_str in shape_strs:
        shape_str_split = shape_str.split('x')
        shape_int_split = [int(dim) for dim in shape_str_split]
        shapes.append(shape_int_split)
    return shapes

def close_files(files):
    for file in files:
        file.close()
        os.remove(file.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Verify numerical results of a mlir program with a reference mlir program.
                       NOTE: the main function should accept same typed tensors for this to work."""
    )
    parser.add_argument("--mlir-uut", help="The unit under test", type=str, required=True)
    parser.add_argument("--mlir-ref", help="The reference mlir file", type=str, required=True)
    parser.add_argument("--input-shape", help="input shape(s)", type=str, action='append', required=True)
    parser.add_argument("--device", help="iree device", type=str, default="hip")
    parser.add_argument("--hip-target", help="iree hip target", type=str, default="gfx942")
    parser.add_argument("--extra-compiler-flags", help="extra iree-compile flags", type=str, action='append', default="--iree-codegen-gpu-native-math-precision")

    args = parser.parse_args()
    mlir_uut_file = args.mlir_uut
    extra_compiler_flags = args.extra_compiler_flags
    if type(extra_compiler_flags) == str:
        extra_compiler_flags = [extra_compiler_flags]
    mlir_uut_vmfb = compile_attention_mlir_file(mlir_uut_file, args.device, args.hip_target, extra_compiler_flags)
    mlir_ref_file = args.mlir_ref
    mlir_ref_vmfb = compile_attention_mlir_file(mlir_ref_file, args.device, args.hip_target, extra_compiler_flags)

    shape_strs = args.input_shape
    shapes = get_shapes(shape_strs)
    inputfiles = create_test_inputs(shapes)

    mlir_uut_out_file = get_output_npy(mlir_uut_vmfb, inputfiles, args.device)
    mlir_ref_out_file = get_output_npy(mlir_ref_vmfb, inputfiles, args.device)

    uut_out = np.load(mlir_uut_out_file.name)
    print("uut:")
    print(uut_out)

    ref_out = np.load(mlir_ref_out_file.name)
    print("ref:")
    print(ref_out)

    diff = uut_out - ref_out
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    print(f"max_diff:{max_diff}")
    print(f"avg_diff:{avg_diff}")
    print("diff:")
    print(diff)

    close_files([mlir_uut_out_file, mlir_ref_out_file])
    close_files(inputfiles)
