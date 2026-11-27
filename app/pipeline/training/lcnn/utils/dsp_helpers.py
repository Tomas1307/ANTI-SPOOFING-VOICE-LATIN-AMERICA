"""
Signal-processing helpers the LFCC front end depends on.

Faithfully ported from ``sandbox/util_frontend.py`` and ``sandbox/util_dsp.py``
in project-NN-Pytorch-scripts (Xin Wang, NII), so a checkpoint trained against
that code produces identical features here.
"""
import sys

import torch
import torch.nn.functional as torch_nn_func


def stft_wrapper(
    x: torch.Tensor,
    fft_n: int,
    frame_shift: int,
    frame_length: int,
    window: torch.Tensor,
    pad_mode: str = "constant",
) -> torch.Tensor:
    """Compute a complex-valued STFT with a torch-version-appropriate call.

    Args:
        x: Waveform of shape (batch, length).
        fft_n: FFT length.
        frame_shift: Hop size in samples.
        frame_length: Frame length in samples.
        window: Window coefficients of shape (frame_length,).
        pad_mode: Padding mode forwarded to torch.stft.

    Returns:
        Complex-valued STFT of shape (batch, freq_bin, frame_num).
    """
    return torch.stft(
        x, fft_n, frame_shift, frame_length,
        window=window, onesided=True, pad_mode=pad_mode,
        return_complex=True,
    )


def trimf(x: torch.Tensor, params) -> torch.Tensor:
    """Triangular membership function, matching Matlab's ``trimf``.

    Args:
        x: Points to evaluate.
        params: Three increasing values [a, b, c] defining the triangle.

    Returns:
        Membership values in [0, 1], same shape as x.

    Raises:
        SystemExit: If params does not hold exactly three non-decreasing
            values. Preserved from the original rather than converted to an
            exception, so behaviour matches the checkpoint's training code.
    """
    if len(params) != 3:
        print("trimp requires params to be a list of 3 elements")
        sys.exit(1)
    a, b, c = params
    if a > b or b > c:
        print("trimp(x, [a, b, c]) requires a<=b<=c")
        sys.exit(1)

    y = torch.zeros_like(x)
    if a < b:
        index = torch.logical_and(a < x, x < b)
        y[index] = (x[index] - a) / (b - a)
    if b < c:
        index = torch.logical_and(b < x, x < c)
        y[index] = (c - x[index]) / (c - b)
    y[x == b] = 1
    return y


def delta(x: torch.Tensor) -> torch.Tensor:
    """Compute the first-order delta of a feature sequence.

    Args:
        x: Tensor of shape (batch, length, dim).

    Returns:
        Delta features of the same shape, computed along the length dimension
        with replicate padding at the edges.
    """
    length = x.shape[1]
    x_padded = torch_nn_func.pad(
        x.unsqueeze(1), (0, 0, 1, 1), "replicate"
    ).squeeze(1)
    return -1 * x_padded[:, 0:length] + x_padded[:, 2:]
