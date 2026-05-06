import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class iTransformerBackbone(nn.Module):
    def __init__(self, c_in, seq_len, pred_len,
                 d_model=512, n_heads=8, e_layers=4, d_ff=512,
                 dropout=0.1, activation='gelu'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.d_model = d_model

        # Variate embedding: each variate's L-dim series → D-dim token
        self.enc_embedding = nn.Linear(seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder: attention across d variate tokens
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, attention_dropout=dropout,
                                 output_attention=False),
                    d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation
            ) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Prediction head: D-dim → H-dim per variate
        self.projection = nn.Linear(d_model, pred_len)

    def _normalize(self, x_enc):
        """Instance normalization. Returns normalized x, means, stdev."""
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def encode(self, x_enc):
        x_enc, means, stdev = self._normalize(x_enc)
        # [B, L, d] → [B, d, L] → embedding → [B, d, D]
        x_enc = x_enc.permute(0, 2, 1)
        enc_in = self.embed_dropout(self.enc_embedding(x_enc))
        h_enc, _ = self.encoder(enc_in)  # [B, d, D]
        return h_enc, means, stdev

    def forecast(self, x_enc):
        h_enc, means, stdev = self.encode(x_enc)
        # Project: [B, d, D] → [B, d, H] → [B, H, d]
        mu = self.projection(h_enc).permute(0, 2, 1)
        # Denormalize
        mu = mu * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        mu = mu + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return mu

    def forward(self, x_enc, x_mark_enc=None, x_dec=None,
                x_mark_dec=None, mask=None):
        return self.forecast(x_enc)


class SigmaHead(nn.Module):

    def __init__(self, d_model, pred_len, dropout=0.1,
                 log_sigma_min=-5.0, log_sigma_max=5.0):
        super().__init__()
        self.pred_len = pred_len
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -0.5)

    def forward(self, h_enc, stdev):

        log_sigma = self.mlp(h_enc)                               # [B, d, H]
        log_sigma = log_sigma.clamp(self.log_sigma_min,
                                     self.log_sigma_max)           # bounded
        sigma = torch.exp(log_sigma)                               # [B, d, H]
        sigma = sigma.permute(0, 2, 1)                             # [B, H, d]
        # Scale to original data space
        sigma = sigma * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len, 1)
        # return sigma
        return sigma.clamp(min=0.01, max=10.0)
