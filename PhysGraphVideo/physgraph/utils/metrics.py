# File: physgraph/utils/metrics.py
# Torch-only metrics for videos. Always return 0-dim torch.Tensor on CPU
# so callers can safely use .item().

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert to float32 tensor in [0,1]."""
    if not torch.is_tensor(x):
        raise TypeError("Expect torch.Tensor for metrics.")
    x = x.detach()
    if x.dtype != torch.float32 and x.dtype != torch.float64:
        x = x.float()
    # assume input already in [0,1]; clamp for safety
    x = x.clamp(0.0, 1.0)
    return x


def _flatten_video(v: torch.Tensor) -> torch.Tensor:
    """
    Accept [T,C,H,W] or [T,B,C,H,W] and return a 2D tensor
    shaped [N, C, H, W] with N=T or N=T*B for frame-wise ops.
    """
    if v.ndim == 4:          # [T,C,H,W]
        return v
    if v.ndim == 5:          # [T,B,C,H,W] -> [T*B,C,H,W]
        T, B, C, H, W = v.shape
        return v.reshape(T * B, C, H, W)
    raise ValueError(f"Unexpected video shape {tuple(v.shape)}; expect [T,C,H,W] or [T,B,C,H,W].")


def video_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all frames/pixels; returns 0-dim tensor (CPU)."""
    pred = _to_float_tensor(pred)
    tgt  = _to_float_tensor(tgt)
    if pred.shape != tgt.shape:
        raise ValueError(f"MSE shape mismatch: {tuple(pred.shape)} vs {tuple(tgt.shape)}")
    mse = F.mse_loss(pred, tgt, reduction="mean")
    return mse.detach().to("cpu")


def video_psnr(pred: torch.Tensor, tgt: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    PSNR in dB over the whole clip; returns 0-dim tensor (CPU).
    """
    mse = video_mse(pred, tgt)  # 0-dim CPU tensor
    # Avoid log(0)
    eps = torch.finfo(torch.float32).eps
    psnr = 20.0 * torch.log10(torch.tensor(data_range, dtype=torch.float32)) - 10.0 * torch.log10(mse.clamp_min(eps).float())
    return psnr.detach().to("cpu")


# ---- SSIM implementation (torch only) ------------------------------------

def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)  # [1,W]
    window_1d = g
    window_2d = (window_1d.t() @ window_1d).unsqueeze(0).unsqueeze(0)  # [1,1,W,W]
    window_2d = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window_2d


def _ssim_per_frame(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> torch.Tensor:
    """
    SSIM for a single frame batch [N,C,H,W] on the same device; returns [N] tensor (CPU at the end).
    """
    C = x.shape[1]
    device = x.device
    dtype  = x.dtype
    window = _gaussian_window(window_size, sigma, C, device, dtype)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=C) - mu_xy

    # constants per original paper
    L = float(data_range)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    # mean over C,H,W -> [N]
    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    return ssim_val.detach().to("cpu")


def video_ssim(pred: torch.Tensor, tgt: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Average SSIM over all frames; returns 0-dim tensor (CPU).
    Supports [T,C,H,W] and [T,B,C,H,W].
    """
    pred = _to_float_tensor(pred)
    tgt  = _to_float_tensor(tgt)
    if pred.shape != tgt.shape:
        raise ValueError(f"SSIM shape mismatch: {tuple(pred.shape)} vs {tuple(tgt.shape)}")

    # Flatten frames
    if pred.ndim == 5:
        T, B, C, H, W = pred.shape
        x = pred.reshape(T * B, C, H, W)
        y = tgt.reshape(T * B, C, H, W)
    elif pred.ndim == 4:
        x = pred
        y = tgt
    else:
        raise ValueError(f"Unexpected video shape {tuple(pred.shape)}; expect [T,C,H,W] or [T,B,C,H,W].")

    vals = _ssim_per_frame(x, y, data_range=data_range)  # [N]
    return vals.mean().detach().to("cpu")
