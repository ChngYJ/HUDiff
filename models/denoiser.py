import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, d_model, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    @staticmethod
    def sinusoidal_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t_freq = self.sinusoidal_embedding(t, self.freq_dim)
        return self.mlp(t_freq)

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, N, D] → [B, N, D]."""
        B, N, D = x.shape
        qkv = self.W_qkv(x).reshape(B, N, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, dk]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.d_k ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.W_o(out)


class AdaLNBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        """x: [B, d, D'], c: [B, d, D']."""
        params = self.adaLN_modulation(c)
        γ1, β1, α1, γ2, β2, α2 = params.chunk(6, dim=-1)

        # Sublayer A: Self-Attention + AdaLN-Zero
        x = x + α1 * self.attn(modulate(self.norm1(x), β1, γ1))

        # Sublayer B: FFN + AdaLN-Zero
        x = x + α2 * self.ffn(modulate(self.norm2(x), β2, γ2))

        return x


class FinalLayer(nn.Module):
    def __init__(self, d_model, pred_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, pred_len)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)

class DenoisingNetwork(nn.Module):

    def __init__(self, pred_len, d_enc, d_model=256, n_heads=8,
                 n_layers=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        # ── Token construction: concat(r_t, σ) → D' ──
        self.proj_token = nn.Linear(2 * pred_len, d_model)

        # ── Condition: h_enc → D' ──
        self.proj_cond = nn.Linear(d_enc, d_model)

        # ── Timestep embedder ──
        self.time_embed = TimestepEmbedder(d_model)

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Output ──
        self.final_layer = FinalLayer(d_model, pred_len)

        # ── Initialize ──
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, '_zero_inited'):
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        # Timestep MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

    def forward(self, r_t, t, sigma_theta, h_enc):
        B, H, d = r_t.shape

        # ── Token: concat(r_t, σ) per-variate → project ──
        r_t_T = r_t.permute(0, 2, 1)            # [B, d, H]
        sig_T = sigma_theta.permute(0, 2, 1)     # [B, d, H]
        x = self.proj_token(
            torch.cat([r_t_T, sig_T], dim=-1))   # [B, d, D']

        # ── Per-variate condition: t_embed + h_enc_proj ──
        e_t = self.time_embed(t)                  # [B, D']
        e_t = e_t.unsqueeze(1).expand(B, d, -1)  # [B, d, D']
        e_h = self.proj_cond(h_enc)               # [B, d, D']
        c = e_t + e_h                             # [B, d, D']

        # ── Transformer blocks ──
        for block in self.blocks:
            x = block(x, c)

        # ── Output ──
        out = self.final_layer(x, c)              # [B, d, H]
        eps_hat = out.permute(0, 2, 1)            # [B, H, d]

        return eps_hat
