#!/bin/bash

set -euxo pipefail

configs=(
  "-t f32 -b 32 2048 -n 256 -a --bias --eps 1e-6"
  "-t f32 -b 32 1 2048 -n 256 -a --bias -p 0 2 1 3 --eps 1e-6"
  "-t f16 -b 32 1 2048 -n 256 -a --bias -p 0 2 1 3 --eps 1e-6"
  "-t bf16 -b 32 1 16 -n 2048 -a --bias -p 0 2 3 1 --eps 1e-6"
)

for conf in "${configs[@]}"; do
  python layer_norm.py ${conf} -m fwd
done

for conf in "${configs[@]}"; do
  python layer_norm.py ${conf} -m bwd
done
