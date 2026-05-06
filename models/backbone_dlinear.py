import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to decompose series."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, d]
        # Pad on both sides
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))  # [B, d, L]
        x = x.permute(0, 2, 1)  # [B, L, d]
        return x


class series_decomp(nn.Module):
    """Series decomposition: trend + seasonal."""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearBackbone(nn.Module):
    def __init__(self, c_in, seq_len, pred_len, d_model=512,
                 moving_avg_kernel=25, individual=True, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.d_model = d_model
        self.individual = individual

        self.decomposition = series_decomp(moving_avg_kernel)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(c_in):
                self.Linear_Seasonal.append(nn.Linear(seq_len, pred_len))
                self.Linear_Trend.append(nn.Linear(seq_len, pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

        # Embedding for h_enc: project each variate's input to D dims
        self.enc_embedding = nn.Linear(seq_len, d_model)

    def _normalize(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def encode(self, x_enc):
        x_enc, means, stdev = self._normalize(x_enc)
        # [B, L, d] → [B, d, L] → Linear → [B, d, D]
        h_enc = self.enc_embedding(x_enc.permute(0, 2, 1))
        return h_enc, means, stdev

    def forecast(self, x_enc):
        x_enc, means, stdev = self._normalize(x_enc)
        seasonal, trend = self.decomposition(x_enc)

        B, L, d = seasonal.shape
        if self.individual:
            seasonal_out = torch.zeros(B, self.pred_len, d, device=x_enc.device)
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(d):
                seasonal_out[:, :, i] = self.Linear_Seasonal[i](seasonal[:, :, i])
                trend_out[:, :, i] = self.Linear_Trend[i](trend[:, :, i])
        else:
            seasonal_out = self.Linear_Seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.Linear_Trend(trend.permute(0, 2, 1)).permute(0, 2, 1)

        mu = seasonal_out + trend_out
        # Denormalize
        mu = mu * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        mu = mu + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return mu

    def forward(self, x_enc, x_mark_enc=None, x_dec=None,
                x_mark_dec=None, mask=None):
        return self.forecast(x_enc)
