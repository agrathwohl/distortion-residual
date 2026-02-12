"""
Distortion Residual Level (DRL) -- a differentiable audio distortion metric.

>>> from distortion_residual import DRL
>>> drl = DRL(sample_rate=44100)
>>> result = drl(reference, processed)
>>> result['total_drl_db'].backward()
"""

from distortion_residual.drl import DRL, design_fir_bandpass

__all__ = ["DRL", "design_fir_bandpass"]
__version__ = "0.1.0"
