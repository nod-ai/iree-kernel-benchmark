from ..utils import *
from .conv_utils import *
from typing import Optional


class IREEConvBenchmark(KernelBenchmark):
    def _generate_mlir(self, config: ConvConfig):
        n = config.N
        h = config.H
        w = config.W
        c = config.C
        p = config.P
        q = config.Q
        f = config.F
        stride = config.S
        operation = config.OP
        dtypes = f"{config.input_dtype}x{config.input_dtype}x{config.output_dtype}"
        elem_types = dtypes.split("x")
        in_h = str(int(h) * int(stride) + int(p) - 1)
        in_w = str(int(w) * int(stride) + int(q) - 1)
        if "nhwc" in operation:
            conv_type = "nhwc_hwcf"
            lhs = f"{n}x{in_h}x{in_w}x{c}x{elem_types[0]}"
            rhs = f"{p}x{q}x{c}x{f}x{elem_types[1]}"
            out = f"{n}x{h}x{w}x{f}x{elem_types[2]}"
        if "nchw" in operation:
            conv_type = "nchw_fchw"
            lhs = f"{n}x{c}x{in_h}x{in_w}x{elem_types[0]}"
            rhs = f"{f}x{c}x{p}x{q}x{elem_types[1]}"
            out = f"{n}x{f}x{h}x{w}x{elem_types[2]}"
        one = "1"
        zero = "0"
        if elem_types[0][0] == "f" or elem_types[0][0] == "b":
            one = "1.0"
            zero = "0.0"
        conv_template = CONV
        if "q" in operation:
            conv_template = CONV_Q
        operation = conv_template.format(
            INPUT_TYPE=lhs,
            FILTER_TYPE=rhs,
            OUTPUT_TYPE=out,
            CONV_TYPE=conv_type,
            STRIDE=stride,
        )

        constants = ""
        func_args = ""
        func_args = FUNC_ARGS.format(
            LHS_TYPE=lhs,
            RHS_TYPE=rhs,
        )

        mlir = TEST.format(
            FUNC_NAME="main",
            FUNC_ARGS=func_args,
            CONSTANT_INPUTS=constants,
            LHS_TYPE=lhs,
            RHS_TYPE=rhs,
            OUT_TYPE=out,
            OUT_ELEM_TYPE=elem_types[2],
            ZERO=zero,
            OPERATION=operation,
        )
        return mlir

    def compile_kernel(
        self,
        config,
        mlir_path,
        vmfb_path,
        extra_compiler_args=...,
        mfma_variant=None,
        spec=None,
    ):
        # Name with tag is used for filenames so that duplicate configs with
        # different tags will not clobber eachother.
        if self.dump_dir:
            dump_file = self.dump_dir / (config.get_name() + ".stderr.mlir")

        # Generate mlir content
        mlir_content = self._generate_mlir(config)

        # Write MLIR content to file
        with open(mlir_path, "w") as f:
            f.write(mlir_content)

        # TODO: Do not hardcode device information, instead pass it as a class
        # Compile MLIR to vmfb
        exec_args = [
            "iree-compile",
            # Input file
            f"{mlir_path}",
            # Output file
            "-o",
            f"{vmfb_path}",
            # Target Device: hip
            "--iree-hal-target-device=hip",
            # Device: MI300x
            "--iree-hip-target=gfx942",
        ] + extra_compiler_args

        if self.dump_dir:
            os.makedirs(self.dump_dir / "iree", exist_ok=True)
            dump_file = self.dump_dir / "iree" / (config.get_name() + ".debug.mlir")
            phase_dump = self.dump_dir / "iree" / config.get_name()
            exec_args.append(f"--dump-compilation-phases-to={phase_dump}")

        ret_value, stdout, stderr = run_iree_command(exec_args)
        if ret_value == 0:
            if stderr and self.dump_dir:
                with open(dump_file, "w") as f:
                    f.write(stderr.decode("utf-8"))
        else:
            if self.dump_dir:
                error_file = (
                    self.dump_dir / "iree" / "log" / (config.get_name() + "_error.txt")
                )
                os.makedirs(os.path.dirname(error_file), exist_ok=True)
                print(f"Failed to compile {mlir_path}. Error dumped in {error_file}")
                with open(error_file, "w") as f:
                    f.write(stderr.decode("utf-8"))
            else:
                print(f"Failed to compile {mlir_path}. No dump directory specified.")

            return False

        return True
