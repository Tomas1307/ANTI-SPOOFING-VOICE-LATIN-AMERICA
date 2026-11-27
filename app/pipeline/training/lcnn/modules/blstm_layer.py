"""
Bidirectional LSTM wrapper, faithfully ported from Xin Wang's LCNN baseline.
"""
import sys

import torch
from torch import nn


class BLSTMLayer(nn.Module):
    """A bidirectional LSTM that preserves sequence length.

    Ported without modification from ``sandbox/block_nn.py`` in
    project-NN-Pytorch-scripts (Xin Wang, NII), so that a checkpoint trained
    against that code loads here without a key mismatch.

    Attributes:
        l_blstm: The underlying bidirectional LSTM module.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the layer.

        Args:
            input_dim: Feature dimension of the input.
            output_dim: Feature dimension of the output. Must be even, since
                the forward and backward directions each produce half of it.

        Raises:
            SystemExit: If output_dim is odd. Preserved from the original
                implementation so behaviour matches the checkpoint's training
                code exactly.
        """
        super().__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        self.l_blstm = nn.LSTM(input_dim, output_dim // 2, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bidirectional LSTM over a batch-first sequence.

        Args:
            x: Tensor of shape (batch, length, dim_in).

        Returns:
            Tensor of shape (batch, length, dim_out).
        """
        blstm_data, _hidden = self.l_blstm(x.permute(1, 0, 2))
        return blstm_data.permute(1, 0, 2)
