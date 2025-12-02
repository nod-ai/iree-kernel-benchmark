# Benchmarking Suite for IREE Kernels

## Setup

```shell
docker build --network=host -t kernel-bench:v1 -f docker/Dockerfile .
docker run -it --device=/dev/kfd --device=/dev/dri  kernel-bench:v1 /bin/bash
```
