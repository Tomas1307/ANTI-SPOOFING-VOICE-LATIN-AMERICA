"""
Max Feature Map activation, faithfully ported from Xin Wang's LCNN baseline.
"""
import sys

import torch
from torch import nn


class MaxFeatureMap2D(nn.Module):
    """Max feature map activation over the channel dimension.

    Halves the channel count by taking an elementwise maximum between two
    equal splits of the input. This is the defining nonlinearity of the LCNN
    architecture; it has no learnable parameters, so this class exists purely
    to keep the network structure, and the checkpoint's module indices,
    identical to the published baseline.

    Ported without modification from ``sandbox/block_nn.py`` in
    project-NN-Pytorch-scripts (Xin Wang, NII), so that a checkpoint trained
    against that code loads here without a key mismatch.

    Attributes:
        max_dim: Dimension the maximum is taken over, after splitting it in two.
    """

    def __init__(self, max_dim: int = 1) -> None:
        """Initialize the activation.

        Args:
            max_dim: Dimension to split and maximize over. Index 1 (channel)
                for the convolutional stack this is used in.
        """
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the max feature map.

        Args:
            inputs: Tensor of shape (batch, channel, ...).

        Returns:
            Tensor of shape (batch, channel // 2, ...).

        Raises:
            SystemExit: If max_dim is out of range or the target dimension has
                an odd size. Preserved from the original implementation rather
                than converted to an exception, so behaviour matches the
                checkpoint's training code exactly.
        """
        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)

        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        maximum, _indices = inputs.view(*shape).max(self.max_dim)
        return maximum
