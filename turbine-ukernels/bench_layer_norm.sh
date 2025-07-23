#!/bin/bash

set -euxo pipefail

configs=(
  "-t f32 -b 32 2048 -n 256 -a --bias --eps 1e-6"
)

for conf in "${configs[@]}"; do
  python layer_norm.py ${conf} -m fwd
done
