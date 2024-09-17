# Benchmarking Suite for IREE Kernels

# Setup

If you are not using a local iree build, install the iree pip packages:
```
pip install --find-links https://iree.dev/pip-release-links.html iree-compiler iree-runtime --upgrade
```

Install the requirements for the project:
```
pip install -r requirements.txt
pip install --no-compile --pre --upgrade -e common_tools
```

# Performance

Pick any of the following kernels to test through IREE.
Refer to the respective problems.py file in the folder to see which shapes are being tested.

## Convolution Benchmarking

```
python convbench/shark_conv.py
```

## GEMM Benchmarking

```
python gemmbench/gemm_bench.py
```

## Attention Benchmarking

```
python attentionbench/attention_bench.py
```
