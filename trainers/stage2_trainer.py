import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.tools import EarlyStopping


class Stage2Trainer:
    def __init__(self, backbone, sigma_head, diffusion, device):
        self.backbone = backbone
        self.sigma_head = sigma_head
        self.diffusion = diffusion
        self.device = device
        self.r0_mean = None  # [d]
        self.r0_std = None   # [d]

    def precompute_residual_stats(self, train_loader, args):
        self.backbone.eval()
        all_r0 = []
        with torch.no_grad():
            for bx, by, _, _ in train_loader:
                bx = bx.float().to(self.device)
                by = by[:, -args.pred_len:, :].float().to(self.device)
                mu = self.backbone.forecast(bx)
                all_r0.append((by - mu).cpu())

        all_r0 = torch.cat(all_r0, dim=0)
        r0_mean = all_r0.mean(dim=(0, 1))
        r0_std = all_r0.std(dim=(0, 1)).clamp(min=0.01)

        self.r0_mean = r0_mean.to(self.device)
        self.r0_std = r0_std.to(self.device)

        r0_norm = (all_r0 - r0_mean) / r0_std
        print(f"  r0_std range: [{r0_std.min():.4f}, {r0_std.max():.4f}]")
        print(f"  r₀_norm stats: mean={r0_norm.mean():.4f}  "
              f"std={r0_norm.std():.4f}  "
              f"min={r0_norm.min():.2f}  max={r0_norm.max():.2f}")
        return r0_mean, r0_std

    def _normalize_residual(self, r_0):
        r_0_norm = (r_0 - self.r0_mean) / self.r0_std
        return r_0_norm.clamp(-6.0, 6.0)

    def _denormalize_residual(self, r_0_norm):
        return r_0_norm * self.r0_std + self.r0_mean

    def _get_module1_outputs(self, bx, by_full, args):
        bx = bx.float().to(self.device)
        by = by_full[:, -args.pred_len:, :].float().to(self.device)
        with torch.no_grad():
            h_enc, means, stdev = self.backbone.encode(bx)
            mu = self.backbone.forecast(bx)
            sigma = self.sigma_head(h_enc, stdev)
        r_0 = by - mu
        r_0_norm = self._normalize_residual(r_0)
        return r_0_norm, sigma, h_enc.detach(), mu

    def train(self, train_loader, val_loader, args, save_path):
        os.makedirs(save_path, exist_ok=True)

        self.backbone.to(self.device).eval()
        self.sigma_head.to(self.device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.sigma_head.parameters():
            p.requires_grad = False

        r0_mean, r0_std = self.precompute_residual_stats(train_loader, args)
        torch.save({'r0_mean': r0_mean.cpu(), 'r0_std': r0_std.cpu()},
                   os.path.join(save_path, 'residual_stats.pth'))

        self.diffusion.to(self.device)

        optimizer = optim.Adam(self.diffusion.parameters(), lr=args.lr_s2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_s2, eta_min=args.lr_s2 * 0.01)
        early_stopping = EarlyStopping(
            patience=args.patience_s2, verbose=True, min_delta=1e-5)

        print("=" * 60)
        print(f"  epochs={args.epochs_s2}, lr={args.lr_s2}, "
              f"patience={args.patience_s2}")
        print(f"  T={args.T}, ddim_steps={args.ddim_steps}")
        print("=" * 60)

        for epoch in range(args.epochs_s2):
            self.diffusion.train()
            losses = []
            t0 = time.time()

            for bx, by, _, _ in train_loader:
                r_0_norm, sigma, h_enc, _ = self._get_module1_outputs(
                    bx, by, args)
                loss, info = self.diffusion.training_loss(
                    r_0_norm, sigma, h_enc)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.diffusion.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(info['loss'])

            scheduler.step()
            val_loss = self._eval_loss(val_loader, args)
            elapsed = time.time() - t0

            print(f"  Ep {epoch+1:3d}/{args.epochs_s2} | "
                  f"Train DSM={np.mean(losses):.6f} | "
                  f"Val DSM={val_loss:.6f} | {elapsed:.1f}s")

            early_stopping(val_loss, self.diffusion,
                           os.path.join(save_path, 'diffusion.pth'))
            if early_stopping.early_stop:
                print("  Early stopping.")
                break

        self.diffusion.load_state_dict(
            torch.load(os.path.join(save_path, 'diffusion.pth'),
                       map_location=self.device))
        best_val = early_stopping.best_score
        epochs_trained = epoch + 1
        print(f"Stage 2 done. Best: {save_path}/diffusion.pth\n")
        return {
            'epochs_trained': epochs_trained,
            'best_val_dsm': float(-best_val),
            'r0_mean': self.r0_mean.cpu().tolist(),
            'r0_std': self.r0_std.cpu().tolist(),
        }

    def _eval_loss(self, loader, args):
        self.diffusion.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for bx, by, _, _ in loader:
                r_0_norm, sigma, h_enc, _ = self._get_module1_outputs(
                    bx, by, args)
                loss, _ = self.diffusion.training_loss(
                    r_0_norm, sigma, h_enc)
                total += loss.item() * r_0_norm.shape[0]
                n += r_0_norm.shape[0]
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, test_loader, args):
        """Full evaluation with online metric computation (memory-safe)."""
        from evaluators.metrics import OnlineMetrics

        print("=" * 60)
        chunk = getattr(args, 'sample_chunk_size', 0)
        print(f"  S={args.S} samples, ddim_steps={args.ddim_steps}"
              + (f", chunk_size={chunk}" if chunk > 0 else ", parallel"))
        print("=" * 60)

        self.backbone.eval()
        self.sigma_head.eval()
        self.diffusion.eval()

        om = OnlineMetrics()
        all_mu, all_sigma, all_true = [], [], []
        # For σ quality (subsample to save memory on large datasets)
        sigma_flat_sub, resid_flat_sub = [], []
        max_spearman = 500000  # max scalar pairs for Spearman

        n_total = 0

        for bx, by, _, _ in test_loader:
            bx = bx.float().to(self.device)
            by = by[:, -args.pred_len:, :].float().to(self.device)
            B = bx.shape[0]

            with torch.no_grad():
                h_enc, means, stdev = self.backbone.encode(bx)
                mu = self.backbone.forecast(bx)
                sigma = self.sigma_head(h_enc, stdev)

            # DDIM sample
            r_0_norm_samples = self.diffusion.ddim_sample(
                sigma, h_enc, S=args.S, eta=0.0,
                sample_chunk_size=chunk)

            # Denormalize
            r_0_samples = self._denormalize_residual(r_0_norm_samples)
            Y_samples = mu.unsqueeze(1) + r_0_samples

            # Move to numpy
            mu_np = mu.cpu().numpy()
            sigma_np = sigma.cpu().numpy()
            true_np = by.cpu().numpy()
            samples_np = Y_samples.cpu().numpy()

            # σ-guided point prediction
            raw_corr = samples_np.mean(axis=1) - mu_np
            sigma_med_batch = np.median(sigma_np)
            w = 1.0 / (1.0 + np.exp(
                -(sigma_np - sigma_med_batch) /
                (sigma_med_batch * 0.5 + 1e-8)))
            point_pred = mu_np + w * raw_corr

            # Online metrics update
            om.update(samples_np, true_np, point_pred)

            n_total += B * mu_np.shape[1] * mu_np.shape[2]

            # Store compact arrays (mu, sigma, true only — no samples)
            all_mu.append(mu_np)
            all_sigma.append(sigma_np)
            all_true.append(true_np)

            # Subsample for Spearman
            sf = sigma_np.flatten()
            rf = np.abs(mu_np - true_np).flatten()
            if len(sigma_flat_sub) * len(sf) < max_spearman:
                sigma_flat_sub.append(sf)
                resid_flat_sub.append(rf)

            # Free GPU memory
            del r_0_norm_samples, r_0_samples, Y_samples, samples_np

        # Final metrics
        metrics = om.compute()

        print(f"\n  === Point Prediction ===")
        print(f"  Diffusion:  MSE={metrics['MSE']:.6f}  "
              f"MAE={metrics['MAE']:.6f}")

        print(f"\n  === Probabilistic Prediction ===")
        print(f"  CRPS: {metrics['CRPS_energy']:.6f}")
        print(f"  CRPS_quantile: {metrics['CRPS_quantile']:.6f}")
        print(f"  CRPS_sum (norm): {metrics['CRPS_sum']:.6f}")
        print(f"  QICE (×100): {metrics['QICE']:.4f}")

        # σ quality (on subsampled data)
        rho, pval = 0.0, 1.0
        if sigma_flat_sub:
            sf_all = np.concatenate(sigma_flat_sub)
            rf_all = np.concatenate(resid_flat_sub)
            try:
                from scipy.stats import spearmanr
                rho, pval = spearmanr(sf_all, rf_all)
                rho, pval = float(rho), float(pval)
            except ImportError:
                pass
        print(f"\n  === σ Quality ===")
        print(f"  Spearman ρ(σ, |resid|) = {rho:.4f} (p={pval:.2e})")
        print(f"  Sample spread: mean={metrics['sample_spread_mean']:.4f}")

        mu_arr = np.concatenate(all_mu, 0)
        sigma_arr = np.concatenate(all_sigma, 0)
        true_arr = np.concatenate(all_true, 0)

        return {
            **metrics,
            'rank_corr': rho, 'rank_corr_pval': pval,
            'sample_spread_mean': metrics['sample_spread_mean'],
            'mu': mu_arr, 'sigma': sigma_arr, 'true': true_arr,
        }
