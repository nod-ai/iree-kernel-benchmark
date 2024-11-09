# Benchmarking Suite for IREE Kernels

## Setup

If you are not using a local iree build, install the iree pip packages:
```
pip install --find-links https://iree.dev/pip-release-links.html iree-base-compiler iree-base-runtime --upgrade
```

Create a python environment and install the requirements for the project:
```
python3.11 -m venv bench_venv
source bench_venv/bin/activate
pip install -r requirements.txt
pip install --no-compile --pre --upgrade -e common_tools
pip install iree-turbine@git+https://github.com/iree-org/iree-turbine.git@main
```

## Performance

Pick any of the following kernels to test through IREE.
Refer to the respective problems.py file in the folder to see which shapes are being tested.

### Convolution Benchmarking

```
python convbench/conv_bench.py
```

### GEMM Benchmarking

```
python gemmbench/gemm_bench.py
```

### TK GEMM Benchmarking

```
python gemmbench/gemm_bench.py --tk
```

### Attention Benchmarking

```
python attentionbench/attention_bench.py
```

### Roofline

If you want to generate a roofline plot, you can call any of the suites for now with the --roofline option (provide a commma seperated list if you want to generate for multiple benchmarks combined):

```
python convbench/conv_bench.py --roofline results/iree_conv.csv,results/iree_attention.csv --plot results/attn_conv.png
```

If you want to generate a roofline plot for a certain data type, model, or batch size you can do:

```
python attentionbench/attention_bench.py --roofline results/iree_attention --plot results/attn_conv_bs1_fp8_unet.png --model unet --dtype f8E4M3FNUZ --batch 1
```
