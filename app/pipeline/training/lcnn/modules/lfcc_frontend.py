"""
LFCC front end, faithfully ported from Xin Wang's LCNN baseline.
"""
import torch
from torch import nn

from app.pipeline.training.lcnn.modules.linear_dct import LinearDCT
from app.pipeline.training.lcnn.utils.dsp_helpers import delta, stft_wrapper, trimf


class LFCC(nn.Module):
    """Linear-frequency cepstral coefficients, with energy and deltas.

    Ported from ``sandbox/util_frontend.py`` in project-NN-Pytorch-scripts
    (Xin Wang, NII), so that a checkpoint trained against that code loads here
    without a key mismatch and reproduces the same features.

    One deviation from the literal source: ``torch.stft`` is called with
    ``return_complex=True`` unconditionally rather than the original's
    torch-version branch on the legacy ``[..., 2]`` layout. The amplitude is
    then read with ``.abs()`` instead of ``torch.norm(..., -1)``, which is the
    same quantity: the modulus of a complex number equals the L2 norm of its
    real/imaginary pair. Numerically identical, just not version-conditional.

    Attributes:
        lfcc_fb: Triangular filter bank, shape (num_fft_bins, filter_num).
        l_dct: The DCT-as-linear-layer, weight loaded from the checkpoint.
    """

    def __init__(
        self,
        frame_length: int,
        frame_shift: int,
        fft_n: int,
        sample_rate: int,
        filter_num: int,
        with_energy: bool = False,
        with_emphasis: bool = True,
        with_delta: bool = True,
        num_coef: int = None,
        min_freq: float = 0.0,
        max_freq: float = 1.0,
    ) -> None:
        """Initialize the front end.

        Args:
            frame_length: Frame length in waveform samples.
            frame_shift: Frame shift in waveform samples.
            fft_n: FFT length.
            sample_rate: Sample rate in Hz.
            filter_num: Number of triangular filters in the filter bank.
            with_energy: Whether to replace the first coefficient with
                log energy.
            with_emphasis: Whether to apply pre-emphasis to the waveform.
            with_delta: Whether to append delta and delta-delta coefficients.
            num_coef: Coefficients to keep from the filter bank. None keeps
                all of them.
            min_freq: Lower edge of the analysed band, as a fraction of the
                Nyquist frequency.
            max_freq: Upper edge of the analysed band, as a fraction of the
                Nyquist frequency.

        Raises:
            ValueError: If the frequency band is invalid.
        """
        super().__init__()
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fft_n = fft_n
        self.sample_rate = sample_rate
        self.filter_num = filter_num
        self.num_coef = num_coef if num_coef is not None else filter_num

        if not (0 <= min_freq < max_freq <= 1):
            raise ValueError(
                f"LFCC cannot work with min_freq {min_freq} and max_freq {max_freq}"
            )
        self.min_freq_bin = int(min_freq * (fft_n // 2 + 1))
        self.max_freq_bin = int(max_freq * (fft_n // 2 + 1))
        self.num_fft_bins = self.max_freq_bin - self.min_freq_bin

        f = (sample_rate / 2) * torch.linspace(
            min_freq, max_freq, self.num_fft_bins
        )
        filter_bands = torch.linspace(min(f), max(f), filter_num + 2)
        filter_bank = torch.zeros([self.num_fft_bins, filter_num])
        for index in range(filter_num):
            filter_bank[:, index] = trimf(
                f, [filter_bands[index], filter_bands[index + 1], filter_bands[index + 2]]
            )
        self.lfcc_fb = nn.Parameter(filter_bank, requires_grad=False)

        self.l_dct = LinearDCT(filter_num, bias=False)

        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.window_buf = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract LFCC features from a batch of waveforms.

        Args:
            x: Waveform of shape (batch, length).

        Returns:
            Features of shape (batch, frame_num, dim), where dim is
            ``num_coef`` alone, or ``num_coef * 3`` when deltas are appended.
        """
        if self.with_emphasis:
            x_copy = torch.zeros_like(x) + x
            x_copy[:, 1:] = x[:, 1:] - 0.97 * x[:, 0:-1]
        else:
            x_copy = x

        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.frame_length).to(x.device)

        x_stft = stft_wrapper(
            x_copy, self.fft_n, self.frame_shift, self.frame_length, self.window_buf
        )
        sp_amp = x_stft.abs().pow(2).permute(0, 2, 1).contiguous()

        if self.min_freq_bin > 0 or self.max_freq_bin < (self.fft_n // 2 + 1):
            sp_amp = sp_amp[:, :, self.min_freq_bin : self.max_freq_bin]

        fb_feature = torch.log10(
            torch.matmul(sp_amp, self.lfcc_fb) + torch.finfo(torch.float32).eps
        )

        lfcc = self.l_dct(fb_feature)
        if self.num_coef != self.filter_num:
            lfcc = lfcc[:, :, : self.num_coef]

        if self.with_energy:
            power_spec = sp_amp / self.fft_n
            energy = torch.log10(
                power_spec.sum(axis=2) + torch.finfo(torch.float32).eps
            )
            lfcc = lfcc.clone()
            lfcc[:, :, 0] = energy

        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            return torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        return lfcc
