import numpy as np
import torch
import torch.nn as nn
import iree.turbine.aot as aot

from iree.compiler import compile_file, OutputFormat
from iree.runtime import VmModule
from iree.turbine.runtime import Launchable, Device

from tempfile import TemporaryDirectory
from torch.autograd import DeviceType
from torch.profiler import profile as torch_profile, ProfilerActivity
from pathlib import Path

from typing import Sequence, Any, Optional


def _compile_exported(exported: aot.ExportOutput, **kwargs):
    buffer = None
    with TemporaryDirectory() as tmp:
        exported_name = Path(tmp) / "exported.mlirbc"
        exported.save_mlir(str(exported_name))
        buffer = compile_file(str(exported_name), **kwargs)

    assert buffer is not None
    return buffer


def rel_error(x_true: np.ndarray | torch.Tensor, x: np.ndarray | torch.Tensor):
    """Compute the relative error between the true and the computed values."""
    assert type(x_true) == type(x)
    if isinstance(x_true, torch.Tensor):
        # Prefer f64 for computing the error.
        x = x.detach().to(dtype=torch.float64)
        x_true = x_true.detach().to(dtype=torch.float64)
        return torch.linalg.norm(x - x_true) / torch.linalg.norm(x_true)
    x_true = x_true.astype(np.float64)
    x = x.astype(np.float64)
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


class Run:
    """Base class for all model profiling runners."""

    def __init__(self, arguments: Sequence[Any]):
        self.arguments = list(arguments)

    def run(self):
        """Run the module. This method must be specialized by subclasses."""
        pass

    def profile(
        self, num_its: int = 10, print_profile: bool = True, row_limit: int = 20
    ):
        """Profile the run method."""
        torch.cuda.synchronize()
        with torch_profile(
            activities=[ProfilerActivity.CUDA],
            # Discard 15 iterations to gain stability.
            schedule=torch.profiler.schedule(
                wait=5, warmup=5, active=num_its, skip_first=5
            ),
            record_shapes=True,
        ) as prof:
            for i in range(num_its + 15):
                self.run()
                torch.cuda.synchronize()
                prof.step()
        events = prof.key_averages(group_by_input_shape=True)
        if print_profile:
            print(
                events.table(sort_by="self_cuda_time_total", row_limit=row_limit),
                flush=True,
            )
        # Return the average device time in milliseconds
        return np.array(
            [
                event.self_device_time_total
                for event in events
                if event.device_type == DeviceType.CUDA and ("emcpy" not in event.key)
            ]
        ).sum() / (num_its * 1000.0)


class TorchModule(Run):
    """Native torch runner."""

    def __init__(self, module: nn.Module, arguments: Sequence[Any]):
        """Construct the runner.
        Params:
          - `module`: the torch module to profile.
          - `arguments`: the arguments to use when calling the module.
        """
        super().__init__(arguments)
        self.module = module

    def run(self):
        return self.module(*self.arguments)


class IREEModule(Run):
    """IREE module runner."""

    def __init__(self, vmfb_bytes: bytes, arguments: Sequence[Any]):
        """Construct the runner.
        Params:
          - `vmfb_bytes`: the compiled IREE module to profile,
          - `arguments`: the arguments to use when calling the module.
        """
        super().__init__(list(arguments))

        def get_vmfb(device: Device):
            vm_instance = device.vm_instance
            return VmModule.copy_buffer(vm_instance, vmfb_bytes)

        self.kernel = Launchable.from_vm_module(get_vmfb, entry_point="main")

    @staticmethod
    def compile(
        backend: str,
        chip: str,
        module: nn.Module,
        arguments: Sequence[Any],
        intermediate_folder: Optional[str] = None,
        single_dispatch: bool = False,
        **kwargs,
    ):
        """Compile a torch module using IREE.
        Params:
          - `backend`: the target backend.
          - `chip`: the target chip.
          - `module`: the module to compile.
          - `arguments`: the arguments to use when calling the module.
          - `intermediate_folder`: a folder path to dump intermediate files to.
          - `single_dispatch`: whether to dispatch a single kernel.
        """
        # Export the module through turbine.
        exported = aot.export(
            module,
            args=(*arguments,),
            import_symbolic_shape_expressions=True,
            **kwargs,
        )

        # Dump the exported torch-mlir module.
        if intermediate_folder is not None:
            folder = Path(intermediate_folder)
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / "aot.mlir", "w") as file:
                print(exported.mlir_module, file=file)
        else:
            print(exported.mlir_module, flush=True)

        # Compile the module.
        ireecc_args = [
            f"--iree-hip-target={chip}",
            "--iree-opt-level=O3",
            "--iree-opt-strip-assertions=true",
        ]
        if single_dispatch:
            ireecc_args += [
                "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-preprocessing-make-single-dispatch))"
            ]
        vmfb_bytes = _compile_exported(
            exported,
            target_backends=[backend],
            optimize=True,
            extra_args=ireecc_args,
            output_format=OutputFormat.FLATBUFFER_BINARY,
            strip_source_map=True,
            strip_debug_ops=True,
            output_mlir_debuginfo=False,
        )
        return vmfb_bytes

    @staticmethod
    def from_torch(
        backend: str,
        chip: str,
        module: nn.Module,
        arguments: Sequence[Any],
        **kwargs,
    ):
        """Create a IREE module runner from a torch module."""
        vmfb_bytes = IREEModule.compile(
            backend=backend,
            chip=chip,
            module=module,
            arguments=arguments,
            **kwargs,
        )
        return IREEModule(vmfb_bytes, [a for a in arguments])

    def run(self):
        return self.kernel(*self.arguments)


# Aliases for floating point torch types.
torch_types = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
    "f64": torch.float64,
}
