"""
Differentiable Distortion Residual Level (DRL) metric for PyTorch.

The DRL measures distortion introduced by nonlinear audio processing
using the nulling method:

1. Level-match: scale reference to match processed level (least squares)
2. Subtract: residual = processed - scaled_reference
3. Measure: DRL = 10*log10(||residual||^2 / ||scaled_reference||^2)

All operations maintain gradient flow for use as a loss function.
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

    Supports optional band-wise analysis via FIR bandpass filtering.

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

    def _apply_bandpass(
        self,
        signal: torch.Tensor,
        low: float,
        high: float,
    ) -> torch.Tensor:
        """Apply a pre-computed FIR bandpass filter to a signal."""
        band_name = self._filter_kernels[(low, high)]
        kernel = getattr(self, band_name)

        if signal.dim() == 1:
            signal = signal.unsqueeze(0).unsqueeze(0)
        elif signal.dim() == 2:
            signal = signal.unsqueeze(1)

        kernel = kernel.to(device=signal.device, dtype=signal.dtype)
        padding = self.num_filter_taps // 2
        filtered = F.conv1d(signal, kernel, padding=padding)
        return filtered.squeeze()

    @staticmethod
    def match_levels(
        reference: torch.Tensor,
        processed: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Scale reference to match processed level via least-squares projection.

        Computes ``g_hat = <x, y> / <x, x>`` and returns ``g_hat * x``.
        Fully differentiable w.r.t. both inputs.

        Args:
            reference: Reference (unprocessed) signal.
            processed: Processed signal.
            eps: Numerical stability constant.

        Returns:
            Level-matched reference signal.
        """
        numerator = torch.dot(reference.flatten(), processed.flatten())
        denominator = torch.dot(reference.flatten(), reference.flatten()) + eps
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
            reference: Original signal before processing.
            processed: Signal after processing.
            eps: Numerical stability constant.

        Returns:
            Dictionary with keys:

            - ``total_drl_db``: Broadband DRL in dB (scalar).
            - ``total_drl_percent``: DRL as percentage (scalar).
            - ``band_drl_db``: Per-band DRL in dB (dict, empty if no bands).
            - ``band_drl_percent``: Per-band DRL as percentage (dict).
            - ``residual``: The distortion residual signal.
            - ``residual_rms``: RMS of the residual.
            - ``signal_rms``: RMS of the level-matched reference.
        """
        if reference.dim() > 1:
            reference = reference.squeeze()
        if processed.dim() > 1:
            processed = processed.squeeze()

        min_len = min(len(reference), len(processed))
        reference = reference[:min_len]
        processed = processed[:min_len]

        reference_scaled = self.match_levels(reference, processed)
        residual = processed - reference_scaled

        residual_power = torch.mean(residual**2)
        signal_power = torch.mean(reference_scaled**2) + eps

        drl_ratio = residual_power / signal_power
        total_drl_db = 10 * torch.log10(drl_ratio + eps)
        total_drl_percent = 100 * torch.sqrt(drl_ratio)

        band_drl_db: Dict[str, torch.Tensor] = {}
        band_drl_percent: Dict[str, torch.Tensor] = {}

        for low, high in self.frequency_bands:
            residual_band = self._apply_bandpass(residual, low, high)
            reference_band = self._apply_bandpass(reference_scaled, low, high)

            band_residual_power = torch.mean(residual_band**2)
            band_signal_power = torch.mean(reference_band**2) + eps

            band_ratio = band_residual_power / band_signal_power
            band_name = f"{int(low)}_{int(high)}"
            band_drl_db[band_name] = 10 * torch.log10(band_ratio + eps)
            band_drl_percent[band_name] = 100 * torch.sqrt(band_ratio)

        return {
            "total_drl_db": total_drl_db,
            "total_drl_percent": total_drl_percent,
            "band_drl_db": band_drl_db,
            "band_drl_percent": band_drl_percent,
            "residual": residual,
            "residual_rms": torch.sqrt(residual_power),
            "signal_rms": torch.sqrt(signal_power),
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
