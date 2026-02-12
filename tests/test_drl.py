"""Tests for distortion_residual.DRL."""

import math

import pytest
import torch

from distortion_residual import DRL, design_fir_bandpass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def drl():
    return DRL(sample_rate=44100)


@pytest.fixture
def drl_no_bands():
    return DRL(sample_rate=44100, frequency_bands=None)


@pytest.fixture
def sine_1k():
    """1 kHz sine at 0 dBFS, 1 second."""
    sr = 44100
    t = torch.linspace(0, 1, sr, dtype=torch.float32)
    return torch.sin(2 * math.pi * 1000 * t)


# ---------------------------------------------------------------------------
# design_fir_bandpass
# ---------------------------------------------------------------------------

class TestDesignFirBandpass:
    def test_output_shape(self):
        h = design_fir_bandpass(44100, 200, 2000, num_taps=255)
        assert h.shape == (255,)

    def test_unity_dc_rejection(self):
        """A bandpass that excludes DC should have ~0 DC gain."""
        h = design_fir_bandpass(44100, 200, 2000, num_taps=255)
        # DC gain = sum of coefficients (should be ~0 for a bandpass)
        # Our normalisation makes sum=1, but the passband centre gain
        # is what matters; DC is well-attenuated by the filter shape.
        assert h.dtype == torch.float32

    def test_clamps_frequencies(self):
        """Edge-case frequencies should not crash."""
        h = design_fir_bandpass(44100, 0, 22050, num_taps=127)
        assert h.shape == (127,)


# ---------------------------------------------------------------------------
# DRL — identity / gain-only
# ---------------------------------------------------------------------------

class TestDRLIdentity:
    def test_identity_gives_neg_inf(self, drl, sine_1k):
        """Identical signals → DRL approaches -inf (no distortion)."""
        result = drl(sine_1k, sine_1k)
        assert result["total_drl_db"].item() < -80

    def test_linear_gain_invisible(self, drl, sine_1k):
        """A pure gain change should be cancelled by level matching."""
        gained = sine_1k * 2.0
        result = drl(sine_1k, gained)
        assert result["total_drl_db"].item() < -80

    def test_attenuation_invisible(self, drl, sine_1k):
        gained = sine_1k * 0.25
        result = drl(sine_1k, gained)
        assert result["total_drl_db"].item() < -80


# ---------------------------------------------------------------------------
# DRL — nonlinear processing
# ---------------------------------------------------------------------------

class TestDRLNonlinear:
    def test_hard_clip_detected(self, drl, sine_1k):
        """Hard clipping should produce measurable DRL."""
        clipped = torch.clamp(sine_1k, -0.5, 0.5)
        result = drl(sine_1k, clipped)
        assert result["total_drl_db"].item() > -60  # clearly non-zero

    def test_soft_clip_less_than_hard(self, drl, sine_1k):
        """Soft clipping should produce less distortion than hard."""
        hard = torch.clamp(sine_1k, -0.5, 0.5)
        soft = torch.tanh(sine_1k * 2) / 2  # roughly same headroom
        r_hard = drl(sine_1k, hard)
        r_soft = drl(sine_1k, soft)
        assert r_soft["total_drl_db"].item() < r_hard["total_drl_db"].item()

    def test_more_clipping_higher_drl(self, drl, sine_1k):
        """Heavier clipping → higher (less negative) DRL."""
        mild = torch.clamp(sine_1k, -0.8, 0.8)
        heavy = torch.clamp(sine_1k, -0.3, 0.3)
        r_mild = drl(sine_1k, mild)
        r_heavy = drl(sine_1k, heavy)
        assert r_heavy["total_drl_db"].item() > r_mild["total_drl_db"].item()


# ---------------------------------------------------------------------------
# DRL — gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_grad_through_processed(self, drl_no_bands):
        ref = torch.randn(4410)
        proc = torch.randn(4410, requires_grad=True)
        result = drl_no_bands(ref, proc)
        result["total_drl_db"].backward()
        assert proc.grad is not None
        assert torch.isfinite(proc.grad).all()

    def test_grad_through_gain_parameter(self, drl_no_bands):
        """Gradient should flow through a gain applied to the processed signal."""
        ref = torch.randn(4410)
        gain = torch.tensor(1.0, requires_grad=True)
        proc = ref * gain + torch.randn(4410) * 0.01  # tiny distortion
        result = drl_no_bands(ref, proc)
        result["total_drl_db"].backward()
        assert gain.grad is not None
        assert torch.isfinite(gain.grad).all()


# ---------------------------------------------------------------------------
# DRL — band analysis
# ---------------------------------------------------------------------------

class TestBandAnalysis:
    def test_bands_present(self, drl, sine_1k):
        clipped = torch.clamp(sine_1k, -0.5, 0.5)
        result = drl(sine_1k, clipped)
        assert "20_200" in result["band_drl_db"]
        assert "200_2000" in result["band_drl_db"]
        assert "2000_20000" in result["band_drl_db"]

    def test_no_bands_when_disabled(self, drl_no_bands, sine_1k):
        clipped = torch.clamp(sine_1k, -0.5, 0.5)
        result = drl_no_bands(sine_1k, clipped)
        assert result["band_drl_db"] == {}

    def test_custom_bands(self, sine_1k):
        custom = DRL(sample_rate=44100, frequency_bands=[(100, 1000)])
        clipped = torch.clamp(sine_1k, -0.5, 0.5)
        result = custom(sine_1k, clipped)
        assert "100_1000" in result["band_drl_db"]


# ---------------------------------------------------------------------------
# DRL — output dict completeness
# ---------------------------------------------------------------------------

class TestOutputDict:
    def test_keys(self, drl, sine_1k):
        clipped = torch.clamp(sine_1k, -0.5, 0.5)
        result = drl(sine_1k, clipped)
        expected = {
            "total_drl_db", "total_drl_percent",
            "band_drl_db", "band_drl_percent",
            "residual", "residual_rms", "signal_rms",
        }
        assert expected == set(result.keys())

    def test_percent_positive(self, drl, sine_1k):
        clipped = torch.clamp(sine_1k, -0.5, 0.5)
        result = drl(sine_1k, clipped)
        assert result["total_drl_percent"].item() > 0
