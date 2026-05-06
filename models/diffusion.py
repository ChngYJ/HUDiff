import numpy as np
import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):

    def __init__(self, denoiser, T=200, beta_start=1e-4, beta_end=0.02,
                 ddim_steps=20):
        super().__init__()
        self.denoiser = denoiser
        self.T = T
        self.ddim_steps = ddim_steps

        # ── Linear β schedule ──
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('alpha_bar', alpha_bar.float())

        # ── DDIM timestep subsequence ──
        ddim_seq = np.linspace(0, T - 1, ddim_steps + 1, dtype=int)
        self.register_buffer('ddim_seq',
                             torch.from_numpy(ddim_seq).long())

    def q_sample(self, r_0_norm, t, noise=None):

        if noise is None:
            noise = torch.randn_like(r_0_norm)

        ab = self.alpha_bar[t]
        sqrt_ab = ab.sqrt()[:, None, None]
        sqrt_1mab = (1 - ab).sqrt()[:, None, None]

        r_t = sqrt_ab * r_0_norm + sqrt_1mab * noise
        return r_t, noise

    def training_loss(self, r_0_norm, sigma_theta, h_enc):
        """
        DSM loss: E_t,ε [‖ε̂ - ε‖²].
        """
        B = r_0_norm.shape[0]
        device = r_0_norm.device

        t = torch.randint(0, self.T, (B,), device=device)
        r_t, eps = self.q_sample(r_0_norm, t)

        # Denoiser receives σ as condition, not as noise scale
        eps_hat = self.denoiser(r_t, t, sigma_theta, h_enc)

        loss = nn.functional.mse_loss(eps_hat, eps)

        info = {
            'loss': loss.item(),
            'r0_norm_std': r_0_norm.detach().std().item(),
        }
        return loss, info

    @torch.no_grad()
    def ddim_sample(self, sigma_theta, h_enc, S=10, eta=0.0,
                    sample_chunk_size=0):
        """
        DDIM reverse sampling with optional chunked processing.
        """
        if sample_chunk_size > 0 and S > sample_chunk_size:
            return self._ddim_sample_chunked(
                sigma_theta, h_enc, S, eta, sample_chunk_size)
        else:
            return self._ddim_sample_parallel(
                sigma_theta, h_enc, S, eta)

    def _ddim_sample_parallel(self, sigma_theta, h_enc, S, eta):
        B, H, d = sigma_theta.shape
        device = sigma_theta.device

        sigma_exp = sigma_theta.unsqueeze(1).expand(
            B, S, H, d).reshape(B * S, H, d)
        h_exp = h_enc.unsqueeze(1).expand(
            B, S, h_enc.shape[1], h_enc.shape[2]).reshape(
            B * S, h_enc.shape[1], h_enc.shape[2])

        r_t = torch.randn(B * S, H, d, device=device)

        seq = self.ddim_seq
        for i in range(len(seq) - 1, 0, -1):
            t_cur = seq[i]
            t_prev = seq[i - 1]
            t_batch = torch.full((B * S,), t_cur,
                                 device=device, dtype=torch.long)

            eps_hat = self.denoiser(r_t, t_batch, sigma_exp, h_exp)

            ab_cur = self.alpha_bar[t_cur]
            ab_prev = self.alpha_bar[t_prev]
            r_0_pred = (r_t - (1 - ab_cur).sqrt() * eps_hat) / ab_cur.sqrt()

            if eta > 0 and i > 1:
                sigma_ddim = eta * (
                    (1 - ab_prev) / (1 - ab_cur) *
                    (1 - ab_cur / ab_prev)).sqrt()
                dir_coeff = (1 - ab_prev - sigma_ddim ** 2).sqrt()
                noise = torch.randn_like(r_t)
                r_t = ab_prev.sqrt() * r_0_pred + \
                      dir_coeff * eps_hat + sigma_ddim * noise
            else:
                r_t = ab_prev.sqrt() * r_0_pred + \
                      (1 - ab_prev).sqrt() * eps_hat

        return r_t.reshape(B, S, H, d)

    def _ddim_sample_chunked(self, sigma_theta, h_enc, S, eta,
                              chunk_size):
        """Process S samples in chunks of chunk_size to save memory."""
        B, H, d = sigma_theta.shape
        device = sigma_theta.device
        all_chunks = []

        for s_start in range(0, S, chunk_size):
            s_end = min(s_start + chunk_size, S)
            S_chunk = s_end - s_start

            sigma_exp = sigma_theta.unsqueeze(1).expand(
                B, S_chunk, H, d).reshape(B * S_chunk, H, d)
            h_exp = h_enc.unsqueeze(1).expand(
                B, S_chunk, h_enc.shape[1], h_enc.shape[2]).reshape(
                B * S_chunk, h_enc.shape[1], h_enc.shape[2])

            r_t = torch.randn(B * S_chunk, H, d, device=device)

            seq = self.ddim_seq
            for i in range(len(seq) - 1, 0, -1):
                t_cur = seq[i]
                t_prev = seq[i - 1]
                t_batch = torch.full((B * S_chunk,), t_cur,
                                     device=device, dtype=torch.long)

                eps_hat = self.denoiser(r_t, t_batch, sigma_exp, h_exp)

                ab_cur = self.alpha_bar[t_cur]
                ab_prev = self.alpha_bar[t_prev]
                r_0_pred = (r_t - (1 - ab_cur).sqrt() * eps_hat) / \
                           ab_cur.sqrt()

                if eta > 0 and i > 1:
                    sigma_ddim = eta * (
                        (1 - ab_prev) / (1 - ab_cur) *
                        (1 - ab_cur / ab_prev)).sqrt()
                    dir_coeff = (1 - ab_prev - sigma_ddim ** 2).sqrt()
                    noise = torch.randn_like(r_t)
                    r_t = ab_prev.sqrt() * r_0_pred + \
                          dir_coeff * eps_hat + sigma_ddim * noise
                else:
                    r_t = ab_prev.sqrt() * r_0_pred + \
                          (1 - ab_prev).sqrt() * eps_hat

            all_chunks.append(r_t.reshape(B, S_chunk, H, d))

        return torch.cat(all_chunks, dim=1)  # [B, S, H, d]

    @torch.no_grad()
    def ddim_sample_with_interval_snapshots(self, sigma_theta, h_enc,
                                             S=50, snapshot_steps=None,
                                             sample_chunk_size=0):
        """
        DDIM sampling with prediction interval statistics at each snapshot.
        """
        if snapshot_steps is None:
            n = self.ddim_steps
            snapshot_steps = sorted(set([
                n, round(n * 0.75), round(n * 0.5), round(n * 0.25)
            ]) - {0}, reverse=True)

        B, H, d = sigma_theta.shape
        device = sigma_theta.device

        # Determine chunking
        if sample_chunk_size > 0 and S > sample_chunk_size:
            chunk_sizes = []
            for s_start in range(0, S, sample_chunk_size):
                chunk_sizes.append(min(sample_chunk_size, S - s_start))
        else:
            chunk_sizes = [S]

        # Accumulators: collect r_0_pred from each chunk at each step
        step_r0_all = {k: [] for k in snapshot_steps}
        final_all = []

        for S_chunk in chunk_sizes:
            sigma_exp = sigma_theta.unsqueeze(1).expand(
                B, S_chunk, H, d).reshape(B * S_chunk, H, d)
            h_exp = h_enc.unsqueeze(1).expand(
                B, S_chunk, h_enc.shape[1], h_enc.shape[2]).reshape(
                B * S_chunk, h_enc.shape[1], h_enc.shape[2])

            r_t = torch.randn(B * S_chunk, H, d, device=device)

            seq = self.ddim_seq
            for i in range(len(seq) - 1, 0, -1):
                t_cur = seq[i]
                t_prev = seq[i - 1]
                t_batch = torch.full((B * S_chunk,), t_cur,
                                     device=device, dtype=torch.long)

                eps_hat = self.denoiser(r_t, t_batch, sigma_exp, h_exp)

                ab_cur = self.alpha_bar[t_cur]
                ab_prev = self.alpha_bar[t_prev]
                r_0_pred = (r_t - (1 - ab_cur).sqrt() * eps_hat) / \
                           ab_cur.sqrt()

                if i in snapshot_steps:
                    step_r0_all[i].append(
                        r_0_pred.reshape(B, S_chunk, H, d).cpu())

                r_t = ab_prev.sqrt() * r_0_pred + \
                      (1 - ab_prev).sqrt() * eps_hat

            # Final (step 0)
            final_all.append(r_t.reshape(B, S_chunk, H, d).cpu())

        # Merge chunks → [B, S, H, d]
        final_samples = torch.cat(final_all, dim=1)

        # Step 0 snapshot = final_samples
        step_r0_all[0] = [final_samples]

        # Compute percentiles per step
        snapshots = {}
        all_steps = sorted(list(step_r0_all.keys()), reverse=True)
        for step in all_steps:
            merged = torch.cat(step_r0_all[step], dim=1).numpy()  # [B,S,H,d]
            snapshots[step] = {
                'p5': np.percentile(merged, 5, axis=1),
                'p25': np.percentile(merged, 25, axis=1),
                'median': np.median(merged, axis=1),
                'p75': np.percentile(merged, 75, axis=1),
                'p95': np.percentile(merged, 95, axis=1),
            }

        return final_samples.to(device), snapshots
