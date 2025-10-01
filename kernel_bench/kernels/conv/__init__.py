from .backends.wave_conv import WaveConvBenchmark
from .backends.iree_conv import IREEConvBenchmark
from .backends.torch_conv import TorchConvBenchmark
from .problems import get_tk_conv_configs


def get_default_conv_configs(kernel_type: str, backend_name: str):
    configs = get_tk_conv_configs()
    if backend_name == "torch":
        configs = [(tag, config) for tag, config in configs if "nchw" in config.OP]
    return configs


CONV_BENCH = {
    "conv": {
        # "wave": WaveConvBenchmark,
        "iree": IREEConvBenchmark,
        "torch": TorchConvBenchmark,
    }
}
