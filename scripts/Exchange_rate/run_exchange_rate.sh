#!/bin/bash
set -e
GPU=0; SEED=2024

for PRED_LEN in 96 192 336 720; do
echo ""; echo "═══ exchange_rate H=${PRED_LEN} [PatchTST] ═══"

python stage1_main.py --seed $SEED --gpu $GPU \
  --backbone PatchTST \
  --data_name exchange_rate --root_path ./dataset/ --data_path exchange_rate.csv \
  --enc_in 8 --freq h --seq_len 96 --pred_len $PRED_LEN \
  --d_model 512 --n_heads 8 --e_layers 2 --d_ff 2048 --dropout 0.1 \
  --patch_len 16 --stride 8 \
  --epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 \
  --epochs_sigma 30 --lr_sigma 0.001 --patience_sigma 5

python stage2_main.py --seed $SEED --gpu $GPU \
  --backbone PatchTST \
  --data_name exchange_rate --root_path ./dataset/ --data_path exchange_rate.csv \
  --enc_in 8 --freq h --seq_len 96 --pred_len $PRED_LEN \
  --d_model 512 --n_heads 8 --e_layers 2 --d_ff 2048 \
  --patch_len 16 --stride 8 \
  --d_model_den 512 --n_heads_den 8 --n_layers_den 2 --d_ff_den 2048 \
  --T 500 --ddim_steps 20 \
  --epochs_s2 100 --batch_size 32 --lr_s2 0.0001 --patience_s2 10 \
  --S 100 --test_batch_size 32

done
echo ""; echo "═══ exchange_rate [PatchTST] done ═══"
