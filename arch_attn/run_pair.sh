#!/bin/bash
# Sequential paired runs (never two trainings at once). Usage: wsl -e bash .../wpy.sh is called per arm.
cd /mnt/d/dev/CCC-OMNI/workspace/colony-one-level4
DATA="data/ext_site_v1 data/gen02 data/sp_gen04-endgame_1792767959 data/sp_gen04-endgame_1792768054 data/sp_gen05-endgame_1792768158 data/sp_gen05-endgame_1792768231 data/sp_gen05-endgame_1792768343 data/sp_gen05-endgame_1792768409 "
PY=/home/andrei/.cache/ccc-omni-reference/bin/python
COMMON="--data $DATA --holdout-last --steps 3000 --batch 512 --lr 1.5e-3 --seed 1234"
for arm in "$@"; do
  case $arm in
    cnn)  $PY arch_attn/train_arch.py --arch cnn  --out arch_attn/ckpt_cnn.pt  $COMMON > arch_attn/log_cnn.txt 2>&1 ;;
    attn_l4) $PY arch_attn/train_arch.py --arch attn --out arch_attn/ckpt_attn_l4.pt --d 64 --layers 4 --heads 4 --ffn 2 $COMMON > arch_attn/log_attn_l4.txt 2>&1 ;;
    attn) $PY arch_attn/train_arch.py --arch attn --out arch_attn/ckpt_attn.pt --d 64 --layers 3 --heads 4 --ffn 2 --conv-layers 1 $COMMON > arch_attn/log_attn.txt 2>&1 ;;
  esac
done
