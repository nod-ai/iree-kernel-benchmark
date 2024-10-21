import torch
import time
import csv
from typing import List, Tuple
from tqdm import tqdm
from utils import *

device = "cuda:0"

class TestModule(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

arg_to_torch_dtype = {
    'f16': torch.float16,
}

def run_benchmark(shape: tuple[int, int, int, int, int, str]) -> tuple[float, float]:
    B, H, S_Q, S_KV, DH, dtype = shape

    torch_dtype = arg_to_torch_dtype[dtype]
    
    q = torch.rand([B, H, S_Q, DH], dtype=torch_dtype, device=device)
    k = torch.rand([B, H, S_KV, DH], dtype=torch_dtype, device=device)
    v = torch.rand([B, H, S_KV, DH], dtype=torch_dtype, device=device)
    
    test_module = torch.compile(TestModule(), dynamic=True)
    test_module.to(device=device, dtype=torch_dtype)
    
    # Warmup
    for _ in range(10):
        test_module(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    eval_steps = 3
    for _ in range(eval_steps):
        test_module(q, k, v)
    torch.cuda.synchronize()
    
    mean_microseconds = (time.time() - start_time) * 1e6 / eval_steps
    
    qk_matmul_flops = 2 * B * H * S_Q * S_Q * DH
    pv_matmul_flops = 2 * B * H * S_Q * DH * S_Q
    total_flops = qk_matmul_flops + pv_matmul_flops

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
                int(row['M']),
                int(row['N']),
                int(row['K1']),
                int(row['K2']),
                float(row['arithmetic_intensity']),
                str(row['dtype']),
                row['ok'],
            ))
    return shapes

def main():
    input_csv = "results/iree_attention.csv"
    output_csv = "results/torch_attention.csv"
    
    configs = read_shapes_from_csv(input_csv)
    results = []
    
    for config in tqdm(configs):
        index, tag, name, B, M, N, K1, K2, arithmetic_intensity, dtype, ok = config
        if dtype != "f16":
            continue
        shape = (1, B, K2, K2, N, dtype)
        mean_microseconds, tflops = run_benchmark(shape)
        
        results.append((
            index, tag, name, B, M, N, K1, K2, dtype,
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
        "M",
        "N",
        "K1",
        "K2",
        "dtype",
        "mean_microseconds",
        "arithmetic_intensity",
        "tflops",
        "ok",
    ]
    
    write_results_to_csv(results, output_csv, fieldnames)
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    main()
