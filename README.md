# distortion-residual

A differentiable **Distortion Residual Level (DRL)** metric for PyTorch.

DRL measures the nonlinear distortion introduced by audio processors
(limiters, compressors, saturators, etc.) using the **nulling method**:

1. **Level-match** the reference to the processed signal via least-squares
   projection, cancelling any linear gain difference.
2. **Subtract** the matched reference from the processed signal to isolate
   the distortion residual.
3. **Measure** the power ratio of the residual to the signal:

$$
\text{DRL} = 10 \log_{10} \frac{\lVert d \rVert^2}{\lVert \hat{g}\,x \rVert^2},
\qquad
\hat{g} = \frac{\langle x, y \rangle}{\langle x, x \rangle},
\qquad
d = y - \hat{g}\,x
$$

Every operation is differentiable, so DRL can be used directly as a **loss
function** for gradient-based optimisation of audio processing parameters.

## Installation

```bash
pip install distortion-residual
```

Or from source:

```bash
git clone https://github.com/agrathwohl/distortion-residual.git
cd distortion-residual
pip install -e .
```

### Optional: audio file I/O

```bash
pip install "distortion-residual[audio]"
```

## Quick start

```python
import torch
from distortion_residual import DRL

drl = DRL(sample_rate=44100)

reference = torch.randn(44100)              # 1 s of audio
processed = torch.clamp(reference, -0.5, 0.5)  # hard-clip at -6 dBFS

result = drl(reference, processed)
print(result["total_drl_db"])    # e.g. tensor(-18.42)
print(result["total_drl_percent"])  # e.g. tensor(12.0)
```

### As a loss function

```python
gain = torch.tensor(1.0, requires_grad=True)
processed = torch.tanh(reference * gain)

result = drl(reference, processed)
loss = result["total_drl_db"]
loss.backward()
print(gain.grad)  # gradient flows through
```

### Band-wise analysis

By default, DRL is decomposed into three frequency bands (20-200 Hz,
200-2000 Hz, 2000-20000 Hz). You can customise or disable this:

```python
# Custom bands
drl = DRL(sample_rate=44100, frequency_bands=[(100, 1000), (1000, 10000)])

# Broadband only (faster, no FIR filtering)
drl = DRL(sample_rate=44100, frequency_bands=None)
```

## Output dictionary

`DRL.forward()` returns a dict with:

| Key                 | Type                | Description                        |
| ------------------- | ------------------- | ---------------------------------- |
| `total_drl_db`      | `Tensor` (scalar)   | Broadband DRL in dB                |
| `total_drl_percent` | `Tensor` (scalar)   | DRL as a percentage                |
| `band_drl_db`       | `dict[str, Tensor]` | Per-band DRL in dB                 |
| `band_drl_percent`  | `dict[str, Tensor]` | Per-band DRL as percentage         |
| `residual`          | `Tensor`            | The distortion residual signal     |
| `residual_rms`      | `Tensor` (scalar)   | RMS of the residual                |
| `signal_rms`        | `Tensor` (scalar)   | RMS of the level-matched reference |

## How it works

The level-matching step (`g_hat = <x,y>/<x,x>`) projects out any linear gain component, so DRL is **invariant to makeup gain**. Only the nonlinear distortion component remains in the residual.

This makes DRL ideal for optimising dynamics processors: the loss function
measures what the processor _does to the waveform shape_, not how loud
it makes the output.

### Gradient properties

- **Through the residual**: linear subtraction, gradients pass directly.
- **Through level matching**: quotient rule on the inner-product ratio.
- **Through band filters**: FIR convolution is a linear operation.

All paths are fully differentiable. No straight-through estimators or
surrogate gradients required.

## Development

```bash
git clone https://github.com/agrathwohl/distortion-residual.git
cd distortion-residual
uv sync --extra dev
uv run pytest
```

## License

MIT
