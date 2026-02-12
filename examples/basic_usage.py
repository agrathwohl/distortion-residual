#!/usr/bin/env python
"""Basic usage of distortion_residual.DRL."""

import math

import torch

from distortion_residual import DRL


def main():
    sr = 44100
    t = torch.linspace(0, 1, sr)
    reference = torch.sin(2 * math.pi * 1000 * t)

    # --- Example 1: hard clipping -------------------------------------------
    clipped = torch.clamp(reference, -0.5, 0.5)

    drl = DRL(sample_rate=sr)
    result = drl(reference, clipped)

    print("=== Hard clip at -6 dBFS ===")
    print(f"  DRL:  {result['total_drl_db'].item():.2f} dB")
    print(f"  DRL:  {result['total_drl_percent'].item():.2f} %")
    for band, val in result["band_drl_db"].items():
        print(f"  Band {band} Hz: {val.item():.2f} dB")

    # --- Example 2: use as a loss function -----------------------------------
    gain = torch.tensor(1.0, requires_grad=True)
    processed = torch.tanh(reference * gain)  # soft saturator

    result = drl(reference, processed)
    loss = result["total_drl_db"]
    loss.backward()

    print("\n=== Gradient through soft-clip ===")
    print(f"  DRL:      {loss.item():.2f} dB")
    print(f"  dL/dgain: {gain.grad.item():.6f}")

    # --- Example 3: broadband only (no band decomposition) -------------------
    drl_fast = DRL(sample_rate=sr, frequency_bands=None)
    result = drl_fast(reference, clipped)
    print(f"\n=== Broadband only ===")
    print(f"  DRL: {result['total_drl_db'].item():.2f} dB")


if __name__ == "__main__":
    main()
