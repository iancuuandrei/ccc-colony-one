#!/bin/bash
# Second-seed replication of the CNN control and the attention+conv candidate (sequential).
cd /mnt/d/dev/CCC-OMNI/workspace/colony-one-level4
DATA=$(grep '^DATA=' arch_attn/run_pair.sh | cut -d'"' -f2)
PY=/home/andrei/.cache/ccc-omni-reference/bin/python
COMMON="--data $DATA --holdout-last --steps 3000 --batch 512 --lr 1.5e-3 --seed 99"
$PY arch_attn/train_arch.py --arch cnn --out arch_attn/seed99/ckpt_cnn.pt $COMMON > arch_attn/seed99/log_cnn.txt 2>&1
$PY arch_attn/train_arch.py --arch attn --out arch_attn/seed99/ckpt_attn.pt --d 64 --layers 3 --heads 4 --ffn 2 --conv-layers 1 $COMMON > arch_attn/seed99/log_attn.txt 2>&1
