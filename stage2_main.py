import argparse
import random
import os
import json
import numpy as np
import torch
from data_provider.data_factory import data_provider
from models.backbone import SigmaHead
from models.denoiser import DenoisingNetwork
from models.diffusion import GaussianDiffusion
from trainers.stage2_trainer import Stage2Trainer


DATA_CLASS_MAP = {
    'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2',
    'ETTm1': 'ETTm1', 'ETTm2': 'ETTm2',
    'weather': 'custom', 'electricity': 'custom', 'traffic': 'custom',
    'exchange_rate': 'custom',
}


def build_backbone(args):
    if args.backbone == 'iTransformer':
        from models.backbone import iTransformerBackbone
        return iTransformerBackbone(
            c_in=args.enc_in, seq_len=args.seq_len, pred_len=args.pred_len,
            d_model=args.d_model, n_heads=args.n_heads,
            e_layers=args.e_layers, d_ff=args.d_ff,
            dropout=args.dropout, activation=args.activation)
    elif args.backbone == 'PatchTST':
        from models.backbone_patchtst import PatchTSTBackbone
        return PatchTSTBackbone(
            c_in=args.enc_in, seq_len=args.seq_len, pred_len=args.pred_len,
            d_model=args.d_model, n_heads=args.n_heads,
            e_layers=args.e_layers, d_ff=args.d_ff,
            patch_len=args.patch_len, stride=args.stride,
            dropout=args.dropout)
    elif args.backbone == 'DLinear':
        from models.backbone_dlinear import DLinearBackbone
        return DLinearBackbone(
            c_in=args.enc_in, seq_len=args.seq_len, pred_len=args.pred_len,
            d_model=args.d_model, moving_avg_kernel=args.moving_avg,
            individual=True)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


def get_args():
    p = argparse.ArgumentParser(description='HUDiff Stage 2')
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--gpu', type=int, default=0)

    p.add_argument('--backbone', type=str, default='PatchTST',
                   choices=['iTransformer', 'PatchTST', 'DLinear'])

    # Data
    p.add_argument('--data', type=str, default=None)
    p.add_argument('--data_name', type=str, default='ETTm1')
    p.add_argument('--root_path', type=str, default='./dataset/')
    p.add_argument('--data_path', type=str, default='ETTm1.csv')
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--freq', type=str, default='h')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--enc_in', type=int, default=7)

    # Backbone architecture
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_heads', type=int, default=2)
    p.add_argument('--e_layers', type=int, default=1)
    p.add_argument('--d_ff', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--activation', type=str, default='gelu')
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--moving_avg', type=int, default=25)

    # Denoiser
    p.add_argument('--d_model_den', type=int, default=512)
    p.add_argument('--n_heads_den', type=int, default=8)
    p.add_argument('--n_layers_den', type=int, default=2)
    p.add_argument('--d_ff_den', type=int, default=512)
    p.add_argument('--dropout_den', type=float, default=0.1)

    # Diffusion
    p.add_argument('--T', type=int, default=500)
    p.add_argument('--ddim_steps', type=int, default=20)

    # Training
    p.add_argument('--epochs_s2', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr_s2', type=float, default=0.0001)
    p.add_argument('--patience_s2', type=int, default=15)

    # Inference
    p.add_argument('--S', type=int, default=100)
    p.add_argument('--sample_chunk_size', type=int, default=0,
                   help='Max samples per GPU pass during inference. '
                        '0=all at once. Set to 5 or 10 for large-d datasets '
                        '(ECL d=321, Traffic d=862) to avoid OOM.')

    # Paths
    p.add_argument('--s1_dir', type=str, default=None)
    p.add_argument('--save_dir', type=str, default='./checkpoints/')

    # Compat
    p.add_argument('--task_name', type=str, default='long_term_forecast')
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--seasonal_patterns', type=str, default=None)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--test_batch_size', type=int, default=32)

    args = p.parse_args()
    if args.data is None:
        args.data = DATA_CLASS_MAP.get(args.data_name, 'custom')
    if args.s1_dir is None:
        args.s1_dir = os.path.join(
            './pretrain_checkpoints', args.backbone,
            args.data_name, str(args.pred_len))
    return args


def load_module1(args, device):
    backbone = build_backbone(args)
    backbone.load_state_dict(
        torch.load(os.path.join(args.s1_dir, 'backbone.pth'),
                   map_location=device))
    backbone.to(device).eval()

    sigma_head = SigmaHead(d_model=args.d_model, pred_len=args.pred_len)
    sigma_head.load_state_dict(
        torch.load(os.path.join(args.s1_dir, 'sigma_head.pth'),
                   map_location=device))
    sigma_head.to(device).eval()

    return backbone, sigma_head


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_set, train_loader = data_provider(args, 'train')
    val_set, val_loader = data_provider(args, 'val')
    test_set, test_loader = data_provider(args, 'test')

    print(f"\n{'='*60}")
    print(f"[Stage 2] {args.backbone} | {args.data_name} | "
          f"L={args.seq_len} H={args.pred_len} d={args.enc_in}")
    print(f"{'='*60}")

    backbone, sigma_head = load_module1(args, device)

    denoiser = DenoisingNetwork(
        pred_len=args.pred_len, d_enc=args.d_model,
        d_model=args.d_model_den, n_heads=args.n_heads_den,
        n_layers=args.n_layers_den, d_ff=args.d_ff_den,
        dropout=args.dropout_den)
    diffusion = GaussianDiffusion(
        denoiser=denoiser, T=args.T, ddim_steps=args.ddim_steps)

    save_path = os.path.join(
        args.save_dir, args.backbone,
        args.data_name, str(args.pred_len))

    trainer = Stage2Trainer(backbone, sigma_head, diffusion, device)
    train_info = trainer.train(train_loader, val_loader, args, save_path)
    results = trainer.evaluate(test_loader, args)

    s1_results_path = os.path.join(args.s1_dir, 'results.json')
    s1_summary = {}
    if os.path.exists(s1_results_path):
        with open(s1_results_path, 'r') as f:
            s1_summary = json.load(f)

    import datetime
    summary = {
        # ── Experiment identity ──
        'backbone': args.backbone,
        'data_name': args.data_name,
        'pred_len': args.pred_len,
        'seq_len': args.seq_len,
        'enc_in': args.enc_in,
        'seed': args.seed,
        'timestamp': datetime.datetime.now().isoformat(),

        # ── Model architecture ──
        'denoiser_architecture': {
            'd_model_den': args.d_model_den,
            'n_heads_den': args.n_heads_den,
            'n_layers_den': args.n_layers_den,
            'd_ff_den': args.d_ff_den,
            'dropout_den': args.dropout_den,
        },
        'diffusion_config': {
            'T': args.T,
            'ddim_steps': args.ddim_steps,
            'beta_start': 1e-4,
            'beta_end': 0.02,
            'schedule': 'linear',
        },

        # ── Training info ──
        'training': {
            'lr_s2': args.lr_s2,
            'batch_size': args.batch_size,
            'max_epochs': args.epochs_s2,
            'patience': args.patience_s2,
            **train_info,
        },
        'data_split': {
            'train': len(train_set),
            'val': len(val_set),
            'test': len(test_set),
        },

        # ── Point prediction ──
        'point_prediction': {
            'diffusion_MSE': results['MSE'],
            'diffusion_MAE': results['MAE'],
        },

        # ── Probabilistic prediction ──
        'probabilistic_prediction': {
            'CRPS': results['CRPS_energy'],
            'CRPS_sum': results['CRPS_sum'],
            'QICE': results['QICE'],
            'S': args.S,
        },
    }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)



if __name__ == '__main__':
    main()
