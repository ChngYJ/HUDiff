#!/bin/bash
set -e

# weather with PatchTST backbone
# conda activate torch
# bash scripts/run_weather_PatchTST.sh

GPU=3
SEED=2024

#for PRED_LEN in 96 192 336 720; do
for PRED_LEN in 720; do
  if [ "$PRED_LEN" -eq 96 ]; then
    N_HEADS=4
    N_HEADS_DEN=8
    Chunk_size=0
  elif [ "$PRED_LEN" -eq 192 ]; then
    N_HEADS=16
    N_HEADS_DEN=8
    Chunk_size=0
  elif [ "$PRED_LEN" -eq 336 ]; then
    N_HEADS=4
    N_HEADS_DEN=8
    Chunk_size=0
  elif [ "$PRED_LEN" -eq 720 ]; then
    N_HEADS=4
    N_HEADS_DEN=8
    Chunk_size=1
  fi

  echo ""
  echo "═══ weather H=${PRED_LEN} [PatchTST] | n_heads=${N_HEADS}, n_heads_den=${N_HEADS_DEN} ═══"

#  python stage1_main.py --seed $SEED --gpu $GPU \
#    --backbone PatchTST \
#    --data_name weather --root_path ./dataset/ --data_path weather.csv \
#    --enc_in 21 --freq h --seq_len 96 --pred_len $PRED_LEN \
#    --d_model 512 --n_heads $N_HEADS --e_layers 2 --d_ff 2048 --dropout 0.1 \
#    --patch_len 16 --stride 8 \
#    --epochs 10 --batch_size 32 --learning_rate 0.0001 --patience 3 \
#    --epochs_sigma 30 --lr_sigma 0.001 --patience_sigma 5

  python stage2_main.py --seed $SEED --gpu $GPU \
    --backbone PatchTST \
    --data_name weather --root_path ./dataset/ --data_path weather.csv \
    --enc_in 21 --freq h --seq_len 96 --pred_len $PRED_LEN \
    --d_model 512 --n_heads $N_HEADS --e_layers 2 --d_ff 2048 \
    --patch_len 16 --stride 8 \
    --d_model_den 512 --n_heads_den $N_HEADS_DEN --n_layers_den 3 --d_ff_den 512 \
    --T 500 --ddim_steps 20 \
    --epochs_s2 150 --batch_size 32 --lr_s2 0.0001 --patience_s2 15 \
    --S 100 --test_batch_size 2 --sample_chunk_size $Chunk_size

  # --d_model_den 256 --n_heads_den 8 --n_layers_den 4 --d_ff_den 1024 \
done

echo ""
echo "═══ weather [PatchTST] done ═══"
