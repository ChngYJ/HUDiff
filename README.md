# ✨ HUDiff 

The repo is the official implementation for the paper: Heterogeneous Uncertainty-Informed Diffusion for Probabilistic Time Series Forecasting

## Overview

<p align="center">   <img width="90%" src="fig/overview.png" /> </p>

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

The datasets can be download from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr).

## Usage

Train and evaluate the model; see `./scripts/` for more examples.

```bash
python stage1_main.py --data_name ETTm1 --data_path ETTm1.csv --enc_in 7 --seq_len 96 --pred_len 96 

python stage2_main.py --data_name ETTm1 --data_path ETTm1.csv --enc_in 7  --seq_len 96 --pred_len 96 
```



