import argparse
import shlex
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import logging
import itertools
from pathlib import Path
import csv
import argparse
import sys
from utils import *
from conv_utils import *
from wave_conv_utils import compile_wave_conv_config
import re

def compile_conv_iree(tag, config, kernel_dir, vmfb_dir, extra_compiler_args):
    if config == "N.A.":
        return (None, None, None, None, None)

    mlir_file, vmfb_file, dump_path = compile_conv_config(config, kernel_dir, vmfb_dir, extra_compiler_args)
    return (tag, config, mlir_file, vmfb_file, dump_path)

def compile_conv_wave(tag, config, kernel_dir, vmfb_dir, extra_compiler_args):
    mlir_file, vmfb_file, dump_path = compile_wave_conv_config(config, kernel_dir, vmfb_dir, extra_compiler_args)
    return (tag, config, mlir_file, vmfb_file, dump_path)

def isValid(args):
# TODO (nirvedhmeshram) : pass the command_line here and make a file with (reason, command_line)
# for rejected command lines.
    if (args.forw != 1):
        return False
    if (args.group_count != 1):
        return False
    if(args.dilation_h != 1):
        return False
    if(args.dilation_w != 1):
        return False
    if(args.in_layout != "NCHW" and args.in_layout != "NHWC"):
        return False
    if(args.out_layout != "NCHW" and args.out_layout != "NHWC"):
        return False
    if(args.fil_layout != "NCHW" and args.fil_layout != "NHWC"):
        return False
    return True



def parse_command(command_line, configs):
    parser = argparse.ArgumentParser(description="Convolution Argument Parser")

    # Positional arguments
    # You can comment these postional arguments if they dont exist in your
    # files
    #parser.add_argument('count', type=int, help="Count parameter")
    #parser.add_argument('exec', type=str, help="Executable path")
    parser.add_argument('operation', type=str, help="Type of operation")
    
    parser.add_argument('--in_h', '-H', type=int, default=32, help="Input Height")
    parser.add_argument('--in_layout', '-I', type=str, default="NCHW", help="Input Layout")
    parser.add_argument('--vector_length', '-L', type=int, default=1, help="Tensor vectorization length")
    parser.add_argument('--out_layout', '-O', type=str, default="NCHW", help="Output Layout")
    parser.add_argument('--wei_cast_type', '-R', type=str, help="Weight tensor cast type")
    parser.add_argument('--solution', '-S', help="Solution ID or symbolic name", default=-1)
    parser.add_argument('--out_cast_type', '-T', type=str, help="Output tensor cast type")
    parser.add_argument('--in_cast_type', '-U', type=str, help="Input tensor cast type")
    parser.add_argument('--verify', '-V', type=int, default=1, help="Verify Each Layer")
    parser.add_argument('--in_w', '-W', type=int, default=32, help="Input Width")
    parser.add_argument('--trans_output_pad_w', '-X', type=int, default=0, help="Zero Padding Output for Width")
    parser.add_argument('--trans_output_pad_h', '-Y', type=int, default=0, help="Zero Padding Output for Height")
    parser.add_argument('--tensor_vect', '-Z', type=str, default="0", help="Tensor vectorization type")
    parser.add_argument('--dilation_d', '-^', type=int, default=1, help="Dilation of Filter Depth")
    parser.add_argument('--spatial_dim', '-_', type=int, default=2, help="Convolution spatial dimension")
    parser.add_argument('--in_bias', '-a', type=str, default="", help="Input bias filename")
    parser.add_argument('--bias', '-b', type=int, default=0, help="Use Bias")
    parser.add_argument('--in_channels', '-c', type=int, default=3, help="Number of Input Channels")
    parser.add_argument('--in_data', '-d', type=str, default="", help="Input data filename")
    parser.add_argument('--weights', '-e', type=str, default="", help="Input weights filename")
    parser.add_argument('--fil_layout', '-f', type=str, default="NCHW", help="Filter Layout")
    parser.add_argument('--group_count', '-g', type=int, default=1, help="Number of Groups")
    parser.add_argument('--iter', '-i', type=int, default=10, help="Number of Iterations")
    parser.add_argument('--dilation_w', '-j', type=int, default=1, help="Dilation of Filter Width")
    parser.add_argument('--out_channels', '-k', type=int, default=32, help="Number of Output Channels")
    parser.add_argument('--dilation_h', '-l', type=int, default=1, help="Dilation of Filter Height")
    parser.add_argument('--mode', '-m', type=str, choices=['conv', 'trans'], default='conv', help="Convolution Mode")
    parser.add_argument('--batchsize', '-n', type=int, default=100, help="Mini-batch size")
    parser.add_argument('--dump_output', '-o', type=int, default=0, help="Dumps the output buffers")
    parser.add_argument('--pad_h', '-p', type=int, default=0, help="Zero Padding for Height")
    parser.add_argument('--pad_w', '-q', type=int, default=0, help="Zero Padding for Width")
    parser.add_argument('--pad_d', type=int, default=0, help="Zero Padding for Depth")
    parser.add_argument('--pad_val', '-r', type=int, default=0, help="Padding Value")
    parser.add_argument('--search', '-s', type=int, default=0, help="Search Kernel Config")
    parser.add_argument('--time', '-t', type=int, default=0, help="Time Each Layer")
    parser.add_argument('--conv_stride_h', '-u', type=int, default=1, help="Convolution Stride for Height")
    parser.add_argument('--conv_stride_w', '-v', type=int, default=1, help="Convolution Stride for Width")
    parser.add_argument('--conv_stride_d',  type=int, default=1, help="Convolution Stride for Depth")
    parser.add_argument('--wall', '-w', type=int, choices=[0, 1, 2], default=0, help="Wall-clock Time Each Layer")
    parser.add_argument('--fil_w', '-x', type=int, default=3, help="Filter Width")
    parser.add_argument('--fil_h', '-y', type=int, default=3, help="Filter Height")
    parser.add_argument('--fil_d', type=int, default=1, help="Filter Depth")
    parser.add_argument('--pad_mode', '-z', type=str, choices=['same', 'valid', 'default'], default='default', help="Padding Mode")
    parser.add_argument('--forw', '-F', type=int, default=0, help="Flag enables fwd, bwd, wrw convolutions 0 fwd+bwd+wrw (default), 1 fwd only, 2 bwd only, 4 wrw only, 3 fwd+bwd, 5 fwd+wrw, 6 bwd+wrw")
    
    args = parser.parse_args(shlex.split(command_line))

    if(not isValid(args)):
        configs.append((command_line, "N.A."))
        return

    # MIOpen provides the input conv shapes but IREE expects the ouput shapes so we need to convert them
    N = args.batchsize
    C = args.in_channels
    P = args.fil_h
    Q = args.fil_w
    F = args.out_channels
    SH = args.conv_stride_h
    SW = args.conv_stride_w
    H = (args.in_h + 2 * args.pad_h - P) // SH + 1
    W = (args.in_w + 2 * args.pad_w - Q) // SW + 1
    thisconfig =[]
    thisconfig.append(ConvConfig(N,H,W,C,P,Q,F,SH, SW,"conv_2d_nhwc_hwcf", "bf16", "f32"))
        
    configs += [(command_line, x) for x in thisconfig]


def process_commands(args):
    """
    Reads a file line by line, parses commands, and executes them.
    """
    configs = []
    with open(args.file_path, "r") as file:
        for line in file:
            parse_command(line.strip(), configs)
    print(configs)

    print(f"Generated {len(configs)} conv configs.")

    num_cpus = max(1, cpu_count() - 20)
    print(f"Using {num_cpus} CPUs for parallel processing.")

    manager = Manager()
    vmfb_dict = manager.dict()

    repo_root = Path(__file__).parent.parent
    kernel_dir = repo_root / "conv" / "mlir"
    vmfb_dir = repo_root / "conv" / "vmfb"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    vmfb_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    extra_compiler_args = ['--' + x for x in list(args.Xiree_compile)]
    compile_args = itertools.starmap(
        lambda tag, config: (tag, config, kernel_dir, vmfb_dir, extra_compiler_args), configs
    )
    compile_conv = compile_conv_wave if args.tk else compile_conv_iree
    with Pool(num_cpus) as pool:
        compilation_results = list(tqdm(pool.starmap(compile_conv, list(compile_args))))

    # Create a list to store results in the order of input configurations  
    results = ["N.A."] * len(configs)  

    error_count = 0
    for i, (tag, config, mlir_file, vmfb_file, dump_path) in enumerate(compilation_results):
        if vmfb_file:
            vmfb_dict[vmfb_file] = (tag, config, dump_path)
            results[i] = vmfb_file
        else:
            error_count += 1
    print(
        f"{len(configs) - error_count} Success, {error_count} Failed out of {len(configs)} configs"
    )

    print("Compilation process completed.")
    index = 0
    output_csv = "results/iree_conv_tk.csv" if args.tk else "results/iree_conv.csv"
    entrypoint = "isolated_benchmark" if args.tk else "main"
    csv_dir = os.path.dirname(output_csv)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    #for vmfb_filename, value in vmfb_dict.items():
    for result in results:  
        if result == "N.A.":
            print("N.A.")
            continue

        vmfb_filename = result
        value = vmfb_dict[vmfb_filename]

        tag, config, dump_path = value
        name = config.get_name()

        image_shape = config.get_img_shape()
        filter_shape = config.get_kernel_shape()

        exec_args = [
            "iree-benchmark-module",
            f"--device={device}",
            "--device_allocator=caching",
            f"--module={vmfb_filename}",
            f"--function={entrypoint}",
            #"--benchmark_repetitions=3",
            f"--input={image_shape}",
            f"--input={filter_shape}",
        ]

        if args.tk:
            out_shape = config.get_out_shape()
            exec_args.append(f"--input={out_shape}")

        #print(f"Running {vmfb_filename}...")
        # iree benchmark kernels
        ret_value, cmd_out, cmd_stderr = run_iree_command(exec_args)
        #print(cmd_out)
        cmd_str = cmd_out.decode()
        times = [float(x) for x in re.findall(r'Kernel execution time \(ms\): ([\d.]+)', cmd_str)]
        print(min(times)*1000)
        """
        ok = ret_value == 0
        benchmark_conv_mean_time_ms = bench_summary_process(ret_value, cmd_out)
        benchmark_conv_mean_time_us = benchmark_conv_mean_time_ms * 1000

        flops = config.get_flops()
        byte_count = config.get_byte_count()

        arithmetic_intensity = flops / byte_count
        tflops_per_second = (flops / 1e12) / (benchmark_conv_mean_time_us / 1e6)

        # Compute percentage of the roofline.
        # TODO: Make this target specific and move to common utils.
        tflops_map = {
            "f32": 653.7,
            "f16": 1307.4,
            "bf16": 1307.4,
            "f8E4M3FNUZ": 2614.9,
            "i8": 2614.9,
        }
        roofline_tflops = tflops_map[config.input_dtype]

        results.append(
            (
                index,
                tag,
                name,
                config.N,
                config.H,
                config.W,
                config.C,
                config.P,
                config.Q,
                config.F,
                config.SH,
                config.SW,
                config.input_dtype,
                config.output_dtype,
                round(benchmark_conv_mean_time_us, 4),
                round(arithmetic_intensity, 4),
                round(tflops_per_second, 4),
                roofline_tflops,
                round(tflops_per_second / roofline_tflops, 4),
                ok,
            )
        )
        index += 1

    fieldnames = [
        "index",
        "tag",
        "name",
        "B",
        "H",
        "W",
        "C",
        "P",
        "Q",
        "F",
        "SH",
        "SW",
        "input_dtype",
        "output_dtype",
        "mean_microseconds",
        "arithmetic_intensity",
        "tflops",
        "roofline_tflops",
        "roofline_percent",
        "ok",
    ]

    write_results_to_csv(results, output_csv, fieldnames)
    print(f"Results written to {output_csv}")
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="file reader")
    parser.add_argument('file_path', type=str, help="file path")
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Set the logging level",
    )
    parser.add_argument("--device", help="The IREE device to execute benchmarks on", type=str, default="hip")
    parser.add_argument(
        "--Xiree_compile",
        nargs='+',
        default=[],
        help="Extra command line arguments passed to the IREE compiler. The flags need to be specified without the `--` or `-`."
    )
    parser.add_argument(
        "--roofline",
        help="Comma seperated csv file list to generate roofline plot with",
        default=None,
    )
    parser.add_argument("--plot", help="location to save plot", default=None)
    parser.add_argument("--batch", help="roofline on certain batch", type=int, default=None)
    parser.add_argument("--dtype", help="roofline on certain dtype", default=None)
    parser.add_argument("--model", help="roofline on certain model", default=None)
    parser.add_argument('--tk', help="Run conv kernels using Turbine Kernels", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    process_commands(args)
