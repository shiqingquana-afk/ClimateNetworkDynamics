"""Stable isotropic-GMM KLD used by the original high-dimensional evaluator.

Kept local because evaluating Gaussian densities directly underflows at 768/1536
dimensions. Equal-weight GMMs, sigma=1 variance, 256 draws, seeds 0..4, chunk 64.
The implementation supports the selected CPU/CUDA device and never falls back to
a different estimator. Monte Carlo estimates can be slightly negative.
"""
import math


def estimate_kld(y_true, y_pred, n_mc=256, device="cpu", chunk_size=64, seed=0, n_avg=5):
    import numpy as np
    import torch

    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise ValueError("KLD inputs must have the same [windows, time, dimensions] shape")
    if min(*y_true.shape, n_mc, chunk_size, n_avg) <= 0:
        raise ValueError("KLD dimensions, sample counts, and chunk size must be positive")
    batch, length, dimensions = y_true.shape
    log_coeff = -0.5 * dimensions * math.log(2.0 * math.pi)
    log_length = math.log(length)
    accum = np.zeros(batch, dtype=np.float64)
    with torch.inference_mode():
        for run in range(n_avg):
            generator = torch.Generator(device="cpu").manual_seed(seed + run)
            for start in range(0, batch, chunk_size):
                end = min(start + chunk_size, batch)
                count = end - start
                true = torch.as_tensor(y_true[start:end], dtype=torch.float32, device=device)
                pred = torch.as_tensor(y_pred[start:end], dtype=torch.float32, device=device)
                indices = torch.randint(0, length, (count, n_mc), generator=generator, device="cpu").to(device)
                centers = torch.gather(true, 1, indices.unsqueeze(-1).expand(-1, -1, dimensions))
                noise = torch.randn(count, n_mc, dimensions, generator=generator, device="cpu").to(device=device, dtype=torch.float32)
                samples = centers + noise
                log_p = torch.logsumexp(log_coeff - 0.5 * torch.cdist(samples, true).square(), dim=2) - log_length
                log_q = torch.logsumexp(log_coeff - 0.5 * torch.cdist(samples, pred).square(), dim=2) - log_length
                accum[start:end] += (log_p - log_q).mean(dim=1).cpu().numpy()
    accum /= n_avg
    return [float(value) if np.isfinite(value) else np.nan for value in accum]
