import argparse
import random
import os
import json
import numpy as np
import torch
from data_provider.data_factory import data_provider
from models.backbone import SigmaHead
from trainers.stage1_trainer import Stage1Trainer


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
    p = argparse.ArgumentParser(description='HUDiff Stage 1')
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--gpu', type=int, default=0)

    # ── Backbone selection ──
    p.add_argument('--backbone', type=str, default='PatchTST',
                   choices=['iTransformer', 'PatchTST', 'DLinear'])

    # ── Data ──
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

    # ── Architecture ──
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_heads', type=int, default=2)
    p.add_argument('--e_layers', type=int, default=1)
    p.add_argument('--d_ff', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--activation', type=str, default='gelu')
    # PatchTST specific
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    # DLinear specific
    p.add_argument('--moving_avg', type=int, default=25)

    # ── Stage 1a: Backbone training ──
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=0.0001)
    p.add_argument('--patience', type=int, default=3)

    # ── Stage 1b: σ head training ──
    p.add_argument('--epochs_sigma', type=int, default=30)
    p.add_argument('--lr_sigma', type=float, default=0.001)
    p.add_argument('--patience_sigma', type=int, default=5)

    # ── Compat ──
    p.add_argument('--task_name', type=str, default='long_term_forecast')
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--seasonal_patterns', type=str, default=None)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--test_batch_size', type=int, default=32)
    p.add_argument('--save_dir', type=str, default='./pretrain_checkpoints/')

    args = p.parse_args()
    if args.data is None:
        args.data = DATA_CLASS_MAP.get(args.data_name, 'custom')
    return args


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
    print(f"[Stage 1] {args.backbone} | {args.data_name} | "
          f"L={args.seq_len} H={args.pred_len} d={args.enc_in}")
    print(f"  Train={len(train_set)}  Val={len(val_set)}  Test={len(test_set)}")
    print(f"{'='*60}")

    backbone = build_backbone(args)
    sigma_head = SigmaHead(d_model=args.d_model, pred_len=args.pred_len)

    # Save path includes backbone name
    save_path = os.path.join(
        args.save_dir, args.backbone,
        args.data_name, str(args.pred_len))
    os.makedirs(save_path, exist_ok=True)

    trainer = Stage1Trainer(backbone, sigma_head, device)
    train_info_bb = trainer.train_backbone(train_loader, val_loader, args, save_path)
    train_info_sh = trainer.train_sigma_head(train_loader, val_loader, args, save_path)
    results = trainer.evaluate(test_loader, args)

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
        'architecture': {
            'backbone': args.backbone,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'e_layers': args.e_layers,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'patch_len': getattr(args, 'patch_len', None),
            'stride': getattr(args, 'stride', None),
            'moving_avg': getattr(args, 'moving_avg', None),
        },

        # ── Training info ──
        'training': {
            'stage1a': {
                'lr': args.learning_rate,
                'batch_size': args.batch_size,
                'max_epochs': args.epochs,
                'patience': args.patience,
                **train_info_bb,
            },
            'stage1b': {
                'lr_sigma': args.lr_sigma,
                'max_epochs': args.epochs_sigma,
                'patience': args.patience_sigma,
                **train_info_sh,
            },
        },
        'data_split': {
            'train': len(train_set),
            'val': len(val_set),
            'test': len(test_set),
        },
        # ── σ quality ──
        'sigma_quality': {
            'spearman_rho': results['rank_corr'],
            'spearman_pval': results['p_value'],
            'stats': results['sigma_stats'],
            'calibration_bins': results['calibration_bins'],
        },
    }
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll saved to: {save_path}/")


if __name__ == '__main__':
    main()
