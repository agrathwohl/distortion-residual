"""
Differentiable Distortion Residual Level (DRL) metric for PyTorch.

The DRL measures distortion introduced by nonlinear audio processing
using the nulling method:

1. Level-match: scale reference to match processed level (least squares)
2. Subtract: residual = processed - scaled_reference
3. Measure: DRL = 10*log10(||residual||^2 / ||scaled_reference||^2)

All operations maintain gradient flow for use as a loss function.

Accepted input shapes:
    - ``(T,)``       — single mono signal
    - ``(C, T)``     — single multichannel signal (level-match per channel)
    - ``(B, C, T)``  — batch of multichannel signals

For a batch of mono signals, use ``(B, 1, T)``.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def design_fir_bandpass(
    sample_rate: int,
    low_freq: float,
    high_freq: float,
    num_taps: int = 255,
) -> torch.Tensor:
    """
    Design a linear-phase FIR bandpass filter using windowed sinc method.

    Pure PyTorch implementation (no scipy dependency).

    Args:
        sample_rate: Sample rate in Hz.
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        num_taps: Number of filter taps (odd number recommended).

    Returns:
        FIR filter coefficients as torch.Tensor.
    """
    nyquist = sample_rate / 2
    low_norm = max(0.001, min(low_freq / nyquist, 0.999))
    high_norm = max(low_norm + 0.001, min(high_freq / nyquist, 0.999))

    n = torch.arange(num_taps, dtype=torch.float32) - (num_taps - 1) / 2
    n_safe = torch.where(n == 0, torch.ones_like(n) * 1e-10, n)

    lp_high = high_norm * torch.sinc(high_norm * n_safe)
    lp_low = low_norm * torch.sinc(low_norm * n_safe)
    bp = lp_high - lp_low

    center = num_taps // 2
    bp[center] = high_norm - low_norm

    window = torch.blackman_window(num_taps, dtype=torch.float32)
    bp = bp * window
    bp = bp / bp.sum()

    return bp


class DRL(nn.Module):
    """
    Differentiable Distortion Residual Level (DRL) metric.

    Measures distortion introduced by nonlinear audio processing (limiters,
    compressors, saturators, etc.) by comparing a reference signal to a
    processed signal using the nulling method.

    The level-matching step cancels any linear gain difference, so DRL
    measures only the nonlinear distortion component.

    Supports batch and multichannel inputs. Level-matching and DRL are
    computed **per channel**; ``total_drl_db`` is the mean across all
    batch elements and channels.

    Accepted shapes:
        - ``(T,)``       — single mono signal
        - ``(C, T)``     — multichannel (e.g. stereo), one item
        - ``(B, C, T)``  — batch of multichannel signals

    For a batch of mono signals use ``(B, 1, T)``.

    Args:
        sample_rate: Audio sample rate in Hz.
        frequency_bands: List of (low_hz, high_hz) tuples for band analysis.
            Default: ``[(20, 200), (200, 2000), (2000, 20000)]``.
            Pass ``None`` for broadband-only (no band decomposition).
        num_filter_taps: FIR filter length (higher = sharper cutoff).
        device: Torch device. Auto-detects CUDA if not specified.

    Example::

        drl = DRL(sample_rate=44100)
        result = drl(reference, processed)
        loss = result['total_drl_db']
        loss.backward()  # gradients flow through
    """

    DEFAULT_BANDS: List[Tuple[float, float]] = [
        (20, 200),
        (200, 2000),
        (2000, 20000),
    ]

    def __init__(
        self,
        sample_rate: int = 44100,
        frequency_bands: Optional[List[Tuple[float, float]]] = DEFAULT_BANDS,
        num_filter_taps: int = 255,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.num_filter_taps = num_filter_taps

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        self.frequency_bands = frequency_bands if frequency_bands is not None else []

        self._filter_kernels: Dict[Tuple[float, float], str] = {}
        for low, high in self.frequency_bands:
            kernel = design_fir_bandpass(sample_rate, low, high, num_filter_taps)
            band_name = f"filter_{int(low)}_{int(high)}"
            self.register_buffer(band_name, kernel.unsqueeze(0).unsqueeze(0))
            self._filter_kernels[(low, high)] = band_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_3d(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Normalise input to ``(B, C, T)`` and return original ndim."""
        ndim = x.dim()
        if ndim == 1:
            return x.unsqueeze(0).unsqueeze(0), ndim  # (1, 1, T)
        if ndim == 2:
            return x.unsqueeze(0), ndim  # (1, C, T)
        if ndim == 3:
            return x, ndim
        raise ValueError(f"Expected 1-3D input, got {ndim}D")

    def _apply_bandpass(
        self,
        signal: torch.Tensor,
        low: float,
        high: float,
    ) -> torch.Tensor:
        """Apply bandpass filter to ``(B, C, T)`` signal."""
        band_name = self._filter_kernels[(low, high)]
        kernel = getattr(self, band_name)  # (1, 1, num_taps)
        kernel = kernel.to(device=signal.device, dtype=signal.dtype)

        B, C, T = signal.shape
        x = signal.reshape(B * C, 1, T)
        padding = self.num_filter_taps // 2
        filtered = F.conv1d(x, kernel, padding=padding)
        return filtered.reshape(B, C, -1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def match_levels(
        reference: torch.Tensor,
        processed: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Scale reference to match processed level via least-squares projection.

        Computes ``g_hat = <x, y> / <x, x>`` **per channel** (last dim is
        time) and returns ``g_hat * x``.

        Works on any shape ``(..., T)``.

        Args:
            reference: Reference (unprocessed) signal.
            processed: Processed signal (same shape as *reference*).
            eps: Numerical stability constant.

        Returns:
            Level-matched reference signal (same shape as input).
        """
        numerator = (reference * processed).sum(dim=-1, keepdim=True)
        denominator = (reference * reference).sum(dim=-1, keepdim=True) + eps
        scale = numerator / denominator
        return reference * scale

    def compute_drl(
        self,
        reference: torch.Tensor,
        processed: torch.Tensor,
        eps: float = 1e-10,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DRL using the nulling method.

        Args:
            reference: Original signal — ``(T,)``, ``(C, T)``, or ``(B, C, T)``.
            processed: Processed signal (same shape as *reference*).
            eps: Numerical stability constant.

        Returns:
            Dictionary with keys:

            - ``total_drl_db``: Mean DRL in dB (scalar).
            - ``total_drl_percent``: Mean DRL as percentage (scalar).
            - ``channel_drl_db``: Per-channel DRL in dB.
            - ``channel_drl_percent``: Per-channel DRL as percentage.
            - ``band_drl_db``: Per-band mean DRL in dB (dict).
            - ``band_drl_percent``: Per-band mean DRL as percentage (dict).
            - ``residual``: Distortion residual signal.
            - ``residual_rms``: Per-channel residual RMS.
            - ``signal_rms``: Per-channel level-matched reference RMS.

            Shapes of per-channel tensors match the input rank: scalar for
            1-D input, ``(C,)`` for 2-D, ``(B, C)`` for 3-D.
        """
        ref, ndim = self._to_3d(reference)
        proc, _ = self._to_3d(processed)

        # Truncate to common length
        min_len = min(ref.shape[-1], proc.shape[-1])
        ref = ref[..., :min_len]
        proc = proc[..., :min_len]

        # Level-match per channel  — ref_scaled is (B, C, T)
        ref_scaled = self.match_levels(ref, proc, eps)
        residual = proc - ref_scaled

        # Power per channel — (B, C)
        residual_power = torch.mean(residual**2, dim=-1)
        signal_power = torch.mean(ref_scaled**2, dim=-1) + eps

        drl_ratio = residual_power / signal_power  # (B, C)
        channel_drl_db = 10 * torch.log10(drl_ratio + eps)
        channel_drl_pct = 100 * torch.sqrt(drl_ratio)

        # Scalar totals (mean over batch and channels)
        total_drl_db = channel_drl_db.mean()
        total_drl_pct = channel_drl_pct.mean()

        # Band analysis — scalars (mean over B, C)
        band_drl_db: Dict[str, torch.Tensor] = {}
        band_drl_percent: Dict[str, torch.Tensor] = {}

        for low, high in self.frequency_bands:
            res_band = self._apply_bandpass(residual, low, high)
            ref_band = self._apply_bandpass(ref_scaled, low, high)

            b_res_pow = torch.mean(res_band**2, dim=-1)  # (B, C)
            b_sig_pow = torch.mean(ref_band**2, dim=-1) + eps
            b_ratio = b_res_pow / b_sig_pow

            bname = f"{int(low)}_{int(high)}"
            band_drl_db[bname] = (10 * torch.log10(b_ratio + eps)).mean()
            band_drl_percent[bname] = (100 * torch.sqrt(b_ratio)).mean()

        # Collapse per-channel tensors to match input rank
        def _squeeze_bc(t: torch.Tensor) -> torch.Tensor:
            """Squeeze a (B, C) tensor back to match input ndim."""
            if ndim == 1:
                return t.squeeze()  # scalar
            if ndim == 2:
                return t.squeeze(0)  # (C,)
            return t  # (B, C)

        def _squeeze_bct(t: torch.Tensor) -> torch.Tensor:
            """Squeeze a (B, C, T) tensor back to match input ndim."""
            if ndim == 1:
                return t.squeeze(0).squeeze(0)  # (T,)
            if ndim == 2:
                return t.squeeze(0)  # (C, T)
            return t  # (B, C, T)

        return {
            "total_drl_db": total_drl_db,
            "total_drl_percent": total_drl_pct,
            "channel_drl_db": _squeeze_bc(channel_drl_db),
            "channel_drl_percent": _squeeze_bc(channel_drl_pct),
            "band_drl_db": band_drl_db,
            "band_drl_percent": band_drl_percent,
            "residual": _squeeze_bct(residual),
            "residual_rms": _squeeze_bc(torch.sqrt(residual_power)),
            "signal_rms": _squeeze_bc(torch.sqrt(signal_power)),
        }

    def forward(
        self,
        reference: torch.Tensor,
        processed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DRL between reference and processed signals.

        Args:
            reference: Original signal before processing.
            processed: Signal after processing.

        Returns:
            Dictionary with DRL measurements (see :meth:`compute_drl`).
        """
        return self.compute_drl(reference, processed)
