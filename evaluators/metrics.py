import numpy as np

def MSE(pred, true):
    """Mean Squared Error. pred/true: [N, H, d]."""
    return float(np.mean((pred - true) ** 2))

def MAE(pred, true):
    """Mean Absolute Error. pred/true: [N, H, d]."""
    return float(np.mean(np.abs(pred - true)))

def CRPS_energy(samples, true):
    """
    CRPS = E|Y_hat - Y| - 0.5 * E|Y_hat - Y_hat'|
    Args:
        samples: [N, S, H, d]
        true:    [N, H, d]
    Returns:
        scalar CRPS averaged over all positions
    """
    N, S, H, d = samples.shape
    sf = samples.transpose(0, 2, 3, 1).reshape(-1, S)
    tf = true.reshape(-1)
    t1 = np.mean(np.abs(sf - tf[:, None]), axis=-1)
    ss = np.sort(sf, axis=-1)
    idx = np.arange(1, S + 1)
    gini = np.sum(ss * (2 * idx[None, :] - S - 1), axis=-1) / (S * S)
    return float(np.mean(t1 - gini))


def _quantile_loss(target, forecast, q):
    """Pinball loss for a single quantile."""
    return 2 * np.sum(np.abs(
        (forecast - target) * ((target <= forecast) * 1.0 - q)))


def CRPS_quantile(samples, true):
    """
    Quantile-based CRPS
    Uses 19 quantiles (0.05, 0.10, ..., 0.95).
    Normalized by sum(|true|) for scale-invariance.
    Args:
        samples: [N, S, H, d]
        true:    [N, H, d]
    Returns:
        scalar normalized CRPS
    """
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = max(np.sum(np.abs(true)), 1e-10)
    crps = 0.0
    for q in quantiles:
        q_preds = []
        for j in range(len(samples)):
            q_preds.append(np.quantile(samples[j:j+1], q, axis=1))
        q_pred = np.concatenate(q_preds, axis=0)
        crps += _quantile_loss(true, q_pred, q) / denom
    return float(crps / len(quantiles))


def CRPS_sum(samples, true):
    """
    CRPS on variable-summed series (joint distribution quality).
    Quantile-based, normalized by sum(|true_sum|).
    Args:
        samples: [N, S, H, d]
        true:    [N, H, d]
    Returns:
        scalar normalized CRPS_sum
    """
    target = true.sum(-1)            # [N, H]
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = max(np.sum(np.abs(target)), 1e-10)
    crps = 0.0
    for q in quantiles:
        q_pred = np.quantile(samples.sum(-1), q, axis=1)
        crps += _quantile_loss(target, q_pred, q) / denom
    return float(crps / len(quantiles))


def QICE(samples, true, n_bins=10):
    """
    Quantile Interval Calibration Error
    Measures whether the predicted quantile intervals contain
    the expected proportion of true values.
    Args:
        samples: [N, S, H, d]
        true:    [N, H, d]
        n_bins:  number of quantile bins (default: 10)
    Returns:
        scalar QICE (0 = perfect calibration, unit: percentage points)
    """
    N, S, H, d = samples.shape
    preds = samples.transpose(0, 2, 3, 1).reshape(-1, S)
    targets = true.reshape(-1)

    quantile_list = np.arange(n_bins + 1) * (100.0 / n_bins)
    y_pred_q = np.percentile(preds, q=quantile_list, axis=1)

    membership = ((targets[None, :] - y_pred_q) > 0).astype(int).sum(axis=0)

    bin_count = np.array(
        [(membership == v).sum() for v in range(n_bins + 2)])
    # Merge boundary bins
    bin_count[1] += bin_count[0]
    bin_count[-2] += bin_count[-1]
    bin_count = bin_count[1:-1]

    ratio = bin_count.astype(float) / len(targets)
    return float(np.mean(np.abs(ratio - 1.0 / n_bins)) * 100)


def compute_all_metrics(samples, true, point_pred=None):
    """
    Args:
        samples:    [N, S, H, d]  prediction samples
        true:       [N, H, d]     ground truth
        point_pred: [N, H, d]     optional (default: sample mean)
    Returns:
        dict with all metrics
    """
    if point_pred is None:
        point_pred = samples.mean(axis=1)
    return {
        'MSE': MSE(point_pred, true),
        'MAE': MAE(point_pred, true),
        'CRPS_energy': CRPS_energy(samples, true),
        'CRPS_quantile': CRPS_quantile(samples, true),
        'CRPS_sum': CRPS_sum(samples, true),
        'QICE': QICE(samples, true),
    }


# ═══════════════════════════════════════
# Online (batch-by-batch) metric accumulator
# ═══════════════════════════════════════

class OnlineMetrics:
    """
    Accumulate metrics batch-by-batch without storing full [N,S,H,d] in memory.

    Usage:
        om = OnlineMetrics()
        for batch_samples, batch_true, batch_point_pred in ...:
            om.update(batch_samples, batch_true, batch_point_pred)
        results = om.compute()
    """
    def __init__(self):
        self._n_positions = 0         # total scalar positions
        self._mse_sum = 0.0
        self._mae_sum = 0.0
        self._crps_e_sum = 0.0
        # CRPS_quantile accumulators
        self._quantiles = np.arange(0.05, 1.0, 0.05)
        self._ql_num = np.zeros(len(self._quantiles))  # pinball numerator
        self._ql_denom = 0.0                            # sum(|true|)
        # CRPS_sum accumulators
        self._ql_sum_num = np.zeros(len(self._quantiles))
        self._ql_sum_denom = 0.0
        # QICE accumulators
        self._qice_bins = 10
        self._qice_bin_count = np.zeros(self._qice_bins + 2)
        self._qice_total = 0
        # Sample spread
        self._spread_sum = 0.0
        self._spread_count = 0

    def update(self, samples, true, point_pred=None):
        """
        Process one batch.

        Args:
            samples:    [B, S, H, d]
            true:       [B, H, d]
            point_pred: [B, H, d] or None
        """
        B, S, H, d = samples.shape
        n_pos = B * H * d

        if point_pred is None:
            point_pred = samples.mean(axis=1)

        # ── MSE / MAE ──
        self._mse_sum += np.sum((point_pred - true) ** 2)
        self._mae_sum += np.sum(np.abs(point_pred - true))
        self._n_positions += n_pos

        # ── CRPS_energy ──
        sf = samples.transpose(0, 2, 3, 1).reshape(-1, S)
        tf = true.reshape(-1)
        t1 = np.mean(np.abs(sf - tf[:, None]), axis=-1)
        ss = np.sort(sf, axis=-1)
        idx = np.arange(1, S + 1)
        gini = np.sum(ss * (2 * idx[None, :] - S - 1), axis=-1) / (S * S)
        self._crps_e_sum += np.sum(t1 - gini)

        # ── CRPS_quantile (per-scalar, normalized) ──
        self._ql_denom += np.sum(np.abs(true))
        for qi, q in enumerate(self._quantiles):
            q_pred = np.quantile(samples, q, axis=1)  # [B, H, d]
            self._ql_num[qi] += 2 * np.sum(
                np.abs((q_pred - true) * ((true <= q_pred) * 1.0 - q)))

        # ── CRPS_sum (variate-summed) ──
        target_sum = true.sum(-1)           # [B, H]
        samples_sum = samples.sum(-1)       # [B, S, H]
        self._ql_sum_denom += np.sum(np.abs(target_sum))
        for qi, q in enumerate(self._quantiles):
            q_pred = np.quantile(samples_sum, q, axis=1)  # [B, H]
            self._ql_sum_num[qi] += 2 * np.sum(
                np.abs((q_pred - target_sum) *
                       ((target_sum <= q_pred) * 1.0 - q)))

        # ── QICE ──
        preds_flat = samples.transpose(0, 2, 3, 1).reshape(-1, S)
        targets_flat = true.reshape(-1)
        q_list = np.arange(self._qice_bins + 1) * (100.0 / self._qice_bins)
        y_pred_q = np.percentile(preds_flat, q=q_list, axis=1)
        membership = ((targets_flat[None, :] - y_pred_q) > 0).astype(int).sum(0)
        for v in range(self._qice_bins + 2):
            self._qice_bin_count[v] += (membership == v).sum()
        self._qice_total += len(targets_flat)

        # ── Sample spread ──
        self._spread_sum += float(samples.std(axis=1).sum())
        self._spread_count += B * H * d

    def compute(self):
        """Return final metric values."""
        n = max(self._n_positions, 1)
        denom_q = max(self._ql_denom, 1e-10)
        denom_qs = max(self._ql_sum_denom, 1e-10)
        nq = len(self._quantiles)

        # QICE
        bc = self._qice_bin_count.copy()
        bc[1] += bc[0]
        bc[-2] += bc[-1]
        bc = bc[1:-1]
        ratio = bc / max(self._qice_total, 1)
        qice = float(np.mean(np.abs(ratio - 1.0 / self._qice_bins)) * 100)

        return {
            'MSE': self._mse_sum / n,
            'MAE': self._mae_sum / n,
            'CRPS_energy': self._crps_e_sum / n,
            'CRPS_quantile': float(np.sum(self._ql_num / denom_q) / nq),
            'CRPS_sum': float(np.sum(self._ql_sum_num / denom_qs) / nq),
            'QICE': qice,
            'sample_spread_mean': self._spread_sum / max(self._spread_count, 1),
        }
