#!/bin/bash
set -e
# traffic with PatchTST backbone
# conda activate torch
# bash scripts/run_traffic_PatchTST.sh
GPU=1
SEED=2025   # 1:2024 2:2025

for PRED_LEN in 96 192 336 720; do  # 192 336 720
  if [ "$PRED_LEN" -eq 96 ]; then
    Chunk_size=0
  elif [ "$PRED_LEN" -eq 192 ]; then
    Chunk_size=0
  elif [ "$PRED_LEN" -eq 336 ]; then
    Chunk_size=2
  elif [ "$PRED_LEN" -eq 720 ]; then
    Chunk_size=1
  fi
echo ""; echo "═══ traffic H=${PRED_LEN} [PatchTST] ═══"

python stage1_main.py --seed $SEED --gpu $GPU \
  --backbone PatchTST \
  --data_name traffic --root_path ./dataset/ --data_path traffic.csv \
  --enc_in 862 --freq h --seq_len 96 --pred_len $PRED_LEN \
  --d_model 512 --n_heads 8 --e_layers 2 --d_ff 512 --dropout 0.1 \
  --patch_len 16 --stride 8 \
  --epochs 10 --batch_size 8 --learning_rate 0.0001 --patience 3 \
  --epochs_sigma 30 --lr_sigma 0.001 --patience_sigma 5

python stage2_main.py --seed $SEED --gpu $GPU \
  --backbone PatchTST \
  --data_name traffic --root_path ./dataset/ --data_path traffic.csv \
  --enc_in 862 --freq h --seq_len 96 --pred_len $PRED_LEN \
  --d_model 512 --n_heads 8 --e_layers 2 --d_ff 512 \
  --patch_len 16 --stride 8 \
  --d_model_den 512 --n_heads_den 8 --n_layers_den 4 --d_ff_den 512 \
  --T 500 --ddim_steps 20 \
  --epochs_s2 100 --batch_size 8 --lr_s2 0.001 --patience_s2 10 \
  --S 100 --test_batch_size 8 --sample_chunk_size $Chunk_size

done
echo ""; echo "═══ traffic [PatchTST] done ═══"
