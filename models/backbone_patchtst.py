import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len

        # Padding to ensure full coverage
        self.pad_len = stride - (seq_len - patch_len) % stride
        if self.pad_len == stride:
            self.pad_len = 0
        self.num_patches = (seq_len + self.pad_len - patch_len) // stride + 1

        # Patch projection
        self.proj = nn.Linear(patch_len, d_model)
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B_d, L = x.shape
        # Pad
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        # Unfold into patches: [B*d, num_patches, patch_len]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # Project: [B*d, num_patches, D]
        x = self.proj(x) + self.pos_embed
        return self.dropout(x)


class PatchTSTBackbone(nn.Module):
    def __init__(self, c_in, seq_len, pred_len,
                 d_model=512, n_heads=8, e_layers=3, d_ff=2048,
                 patch_len=16, stride=8, dropout=0.1, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            seq_len, patch_len, stride, d_model, dropout)
        num_patches = self.patch_embed.num_patches

        encoder_kwargs = dict(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True)
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                **encoder_kwargs, norm_first=True)
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(**encoder_kwargs)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(num_patches * d_model, pred_len)

    def _normalize(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def _forward_encoder(self, x_norm):
        B, L, d = x_norm.shape
        # Reshape to CI: [B*d, L]
        x_ci = x_norm.permute(0, 2, 1).reshape(B * d, L)
        # Patch embed: [B*d, num_patches, D]
        patches = self.patch_embed(x_ci)
        # Encoder: [B*d, num_patches, D]
        enc_out = self.encoder(patches)
        enc_out = self.encoder_norm(enc_out)
        # Reshape: [B, d, num_patches, D]
        enc_out = enc_out.reshape(B, d, -1, self.d_model)
        return enc_out

    def encode(self, x_enc):

        x_enc, means, stdev = self._normalize(x_enc)
        enc_out = self._forward_encoder(x_enc)  # [B, d, num_patches, D]
        # Average pool across patches → [B, d, D]
        h_enc = enc_out.mean(dim=2)
        return h_enc, means, stdev

    def forecast(self, x_enc):
        x_enc, means, stdev = self._normalize(x_enc)
        enc_out = self._forward_encoder(x_enc)  # [B, d, num_patches, D]
        B, d, num_p, D = enc_out.shape
        # Flatten: [B*d, num_patches * D]
        flat = enc_out.reshape(B * d, num_p * D)
        # Head: [B*d, H]
        mu_ci = self.head(flat)
        # Reshape: [B, d, H] → [B, H, d]
        mu = mu_ci.reshape(B, d, self.pred_len).permute(0, 2, 1)
        # Denormalize
        mu = mu * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        mu = mu + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return mu

    def forward(self, x_enc, x_mark_enc=None, x_dec=None,
                x_mark_dec=None, mask=None):
        return self.forecast(x_enc)
