#!/bin/bash
# Time the server-style dry-run child under the validator's thread environment (Linux).
cd /mnt/d/dev/CCC-OMNI/workspace/colony-one-level4/env
PY=/home/andrei/.cache/ccc-omni-reference/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for m in "$@"; do
  /usr/bin/time -f "$m %e s wall %U+%S cpu" $PY -I colony_one/submission_check.py --child "$m" 2>&1 | tail -1
done
