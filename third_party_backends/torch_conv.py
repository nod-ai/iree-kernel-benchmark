import torch
import time
import csv
from typing import List, Tuple
from tqdm import tqdm
from utils import *

device = "cuda:0"

class TestModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(TestModule, self).__init__()
        # Define convolution weights and bias
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Perform 2D convolution with the specified stride and padding
        return torch.nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride)

def run_benchmark(shape: tuple[int, int, int, int, int, int, int, int]) -> tuple[float, float]:
    B, H, W, C, P, Q, F, S = shape

    torch_dtype = torch.float32
    in_h = H * S + P - 1
    in_w = W * S + Q - 1
    image = torch.rand([B, C, in_h, in_w], dtype=torch_dtype, device=device)
    
    test_module = torch.compile(TestModule(in_channels=C, out_channels=F, kernel_size=(P, Q), stride=S), dynamic=True)
    test_module.to(device=device, dtype=torch_dtype)
    
    # Warmup
    for _ in range(10):
        test_module(image)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    eval_steps = 3
    for _ in range(eval_steps):
        test_module(image)
    torch.cuda.synchronize()
    
    mean_microseconds = (time.time() - start_time) * 1e6 / eval_steps

    operation_per_pixel = P * Q * C * 2
    output_pixels_per_batch = H * W * F
    total_flops = operation_per_pixel * output_pixels_per_batch * B

    tflops_per_second = (total_flops / 1e12) / (mean_microseconds / 1e6)
    
    return mean_microseconds, tflops_per_second

def read_shapes_from_csv(filename: str) -> list[tuple]:
    shapes = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes.append((
                int(row['index']),
                row['tag'],
                row['name'],
                int(row['B']),
                int(row['H']),
                int(row['W']),
                int(row['C']),
                int(row['P']),
                int(row['Q']),
                int(row['F']),
                int(row['S']),
                float(row['arithmetic_intensity']),
                str(row['input_dtype']),
                str(row['output_dtype']),
                row['ok'],
            ))
    return shapes

def main():
    input_csv = "results/iree_conv.csv"
    output_csv = "results/torch_conv.csv"
    
    configs = read_shapes_from_csv(input_csv)
    results = []
    
    for config in tqdm(configs):
        index, tag, name, B, H, W, C, P, Q, F, S, arithmetic_intensity, input_dtype, output_dtype, ok = config
        if input_dtype != "f32":
            continue
        shape = (B, H, W, C, P, Q, F, S)
        mean_microseconds, tflops = run_benchmark(shape)
        
        results.append((
            index, tag, name, B, H, W, C, P, Q, F, S, input_dtype, output_dtype,
            round(mean_microseconds, 4),
            round(arithmetic_intensity, 4),
            round(tflops, 4),
            ok
        ))
    
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
        "S",
        "input_dtype",
        "output_dtype",
        "mean_microseconds",
        "arithmetic_intensity",
        "tflops",
        "ok",
    ]
    
    write_results_to_csv(results, output_csv, fieldnames)
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    main()
