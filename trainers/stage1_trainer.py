import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.tools import EarlyStopping


class Stage1Trainer:
    def __init__(self, backbone, sigma_head, device):
        self.backbone = backbone
        self.sigma_head = sigma_head
        self.device = device

    def train_backbone(self, train_loader, val_loader, args, save_path):
        self.backbone.to(self.device)
        optimizer = optim.Adam(self.backbone.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)
        early_stopping = EarlyStopping(
            patience=args.patience, verbose=True, min_delta=1e-4)

        print("=" * 60)
        print(f"Stage 1: {args.backbone} Backbone — MSE Training")
        print(f"  epochs={args.epochs}, lr={args.learning_rate}, "
              f"patience={args.patience}, batch_size={args.batch_size}")
        print("=" * 60)

        for epoch in range(args.epochs):
            self.backbone.train()
            losses = []
            t0 = time.time()

            for bx, by, _, _ in train_loader:
                bx = bx.float().to(self.device)
                by = by[:, -args.pred_len:, :].float().to(self.device)

                mu = self.backbone.forecast(bx)
                loss = nn.functional.mse_loss(mu, by)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()
            val_mse = self._eval_backbone_mse(val_loader, args)
            elapsed = time.time() - t0

            print(f"  Ep {epoch+1:3d}/{args.epochs} | "
                  f"Train MSE={np.mean(losses):.6f} | "
                  f"Val MSE={val_mse:.6f} | {elapsed:.1f}s")

            ckpt_path = os.path.join(save_path, 'backbone.pth')
            early_stopping(val_mse, self.backbone, ckpt_path)
            if early_stopping.early_stop:
                print("  Early stopping.")
                break

        # Load best
        self.backbone.load_state_dict(
            torch.load(os.path.join(save_path, 'backbone.pth'),
                       map_location=self.device))
        best_val = early_stopping.best_score
        epochs_trained = epoch + 1
        print(f"Stage 1a done. Best backbone: {save_path}/backbone.pth\n")
        return {'epochs_trained': epochs_trained, 'best_val_mse': float(-best_val)}

    def _eval_backbone_mse(self, loader, args):
        self.backbone.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for bx, by, _, _ in loader:
                bx = bx.float().to(self.device)
                by = by[:, -args.pred_len:, :].float().to(self.device)
                mu = self.backbone.forecast(bx)
                total += nn.functional.mse_loss(mu, by, reduction='sum').item()
                n += by.numel()
        return total / n

    def train_sigma_head(self, train_loader, val_loader, args, save_path):
        """
        Loss: NLL = log(σ) + (y - μ)² / (2σ²)
        """
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.sigma_head.to(self.device)
        optimizer = optim.Adam(
            self.sigma_head.parameters(), lr=args.lr_sigma)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_sigma,
            eta_min=args.lr_sigma * 0.01)
        early_stopping = EarlyStopping(
            patience=args.patience_sigma, verbose=True, min_delta=1e-4)

        print("=" * 60)
        print("Stage 2: NLL Training")
        print(f"  epochs={args.epochs_sigma}, lr={args.lr_sigma}, "
              f"patience={args.patience_sigma}")
        print("=" * 60)

        for epoch in range(args.epochs_sigma):
            self.sigma_head.train()
            nll_losses = []
            sigma_means = []
            t0 = time.time()

            for bx, by, _, _ in train_loader:
                bx = bx.float().to(self.device)
                by = by[:, -args.pred_len:, :].float().to(self.device)

                with torch.no_grad():
                    h_enc, means, stdev = self.backbone.encode(bx)
                    mu = self.backbone.forecast(bx)

                sigma = self.sigma_head(h_enc.detach(), stdev.detach())

                # NLL loss
                residual_sq = (by - mu.detach()) ** 2
                nll = (torch.log(sigma) + 0.5 * residual_sq / (sigma ** 2 + 1e-8)).mean()

                optimizer.zero_grad()
                nll.backward()
                nn.utils.clip_grad_norm_(self.sigma_head.parameters(), max_norm=5.0)
                optimizer.step()

                nll_losses.append(nll.item())
                sigma_means.append(sigma.mean().item())

            scheduler.step()
            val_nll = self._eval_sigma_nll(val_loader, args)
            s_mean = np.mean(sigma_means)
            elapsed = time.time() - t0

            print(f"  Ep {epoch+1:3d}/{args.epochs_sigma} | "
                  f"Train NLL={np.mean(nll_losses):.4f} | "
                  f"Val NLL={val_nll:.4f} | "
                  f"σ̄={s_mean:.4f} | {elapsed:.1f}s")

            ckpt_path = os.path.join(save_path, 'sigma_head.pth')
            early_stopping(val_nll, self.sigma_head, ckpt_path)
            if early_stopping.early_stop:
                print("  Early stopping.")
                break

        # Load best
        self.sigma_head.load_state_dict(
            torch.load(os.path.join(save_path, 'sigma_head.pth'),
                       map_location=self.device))
        best_val = early_stopping.best_score
        epochs_trained = epoch + 1
        print(f"Stage 1b done. Best σ head: {save_path}/sigma_head.pth\n")
        return {'epochs_trained': epochs_trained, 'best_val_nll': float(-best_val)}

    def _eval_sigma_nll(self, loader, args):
        self.backbone.eval()
        self.sigma_head.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for bx, by, _, _ in loader:
                bx = bx.float().to(self.device)
                by = by[:, -args.pred_len:, :].float().to(self.device)

                h_enc, means, stdev = self.backbone.encode(bx)
                mu = self.backbone.forecast(bx)
                sigma = self.sigma_head(h_enc, stdev)

                residual_sq = (by - mu) ** 2
                nll = (torch.log(sigma) + 0.5 * residual_sq / (sigma ** 2 + 1e-8))
                total += nll.sum().item()
                n += by.numel()
        return total / n

    def evaluate(self, test_loader, args):

        self.backbone.eval()
        self.sigma_head.eval()

        all_mu, all_sigma, all_true, all_input = [], [], [], []

        with torch.no_grad():
            for bx, by, _, _ in test_loader:
                bx = bx.float().to(self.device)
                by = by[:, -args.pred_len:, :].float().to(self.device)

                h_enc, means, stdev = self.backbone.encode(bx)
                mu = self.backbone.forecast(bx)
                sigma = self.sigma_head(h_enc, stdev)

                all_mu.append(mu.cpu().numpy())
                all_sigma.append(sigma.cpu().numpy())
                all_true.append(by.cpu().numpy())
                all_input.append(bx.cpu().numpy())

        mu = np.concatenate(all_mu, 0)
        sigma = np.concatenate(all_sigma, 0)
        true = np.concatenate(all_true, 0)
        x_in = np.concatenate(all_input, 0)

        # ── σ vs |residual| rank correlation ──
        abs_resid = np.abs(mu - true)
        sf = sigma.flatten()
        rf = abs_resid.flatten()

        try:
            from scipy.stats import spearmanr
            rank_corr, p_val = spearmanr(sf, rf)
        except ImportError:
            # Manual Spearman fallback
            def _rankdata(a):
                order = a.argsort()
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.arange(len(a), dtype=float)
                return ranks
            rs, rr = _rankdata(sf), _rankdata(rf)
            n = len(rs)
            rank_corr = float(1 - 6 * np.sum((rs - rr) ** 2) / (n * (n ** 2 - 1)))
            p_val = 0.0

        print(f"  σ qual: Spearman ρ(σ, |resid|) = {rank_corr:.4f}  "
              f"(p = {p_val:.2e})")

        # ── Binned calibration table ──
        n_bins = 10
        sorted_idx = np.argsort(sf)
        bin_size = len(sorted_idx) // n_bins

        print(f"\n   Bin | {'σ_mean':>8s} | {'|resid|_mean':>12s} | {'bin MSE':>8s}")
        print("  " + "-" * 48)
        bin_s, bin_r = [], []
        for i in range(n_bins):
            lo = i * bin_size
            hi = lo + bin_size if i < n_bins - 1 else len(sorted_idx)
            idx = sorted_idx[lo:hi]
            sm = float(sf[idx].mean())
            rm = float(rf[idx].mean())
            bm = float((rf[idx] ** 2).mean())
            bin_s.append(sm)
            bin_r.append(rm)
            print(f"  {i+1:4d} | {sm:8.4f} | {rm:12.4f} | {bm:8.4f}")

        # σ/|resid| ratio
        sigma_resid_ratio_median = float('nan')
        sigma_resid_ratio_mean = float('nan')
        valid = abs_resid > 1e-6
        if valid.any():
            ratio = sigma[valid] / abs_resid[valid]
            sigma_resid_ratio_median = float(np.median(ratio))
            sigma_resid_ratio_mean = float(ratio.mean())

        return {
            'rank_corr': float(rank_corr), 'p_value': float(p_val),
            'sigma_stats': {
                'mean': float(sigma.mean()),
                'std': float(sigma.std()),
                'min': float(sigma.min()),
                'max': float(sigma.max()),
                'log_mean': float(np.log(sigma + 1e-8).mean()),
                'log_std': float(np.log(sigma + 1e-8).std()),
                'resid_ratio_median': sigma_resid_ratio_median,
                'resid_ratio_mean': sigma_resid_ratio_mean,
            },
            'calibration_bins': {
                'bin_sigma': bin_s, 'bin_resid': bin_r,
            },
            'mu': mu, 'sigma': sigma, 'true': true, 'x_input': x_in,
        }
