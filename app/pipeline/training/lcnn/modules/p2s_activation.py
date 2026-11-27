"""
P2SGrad output layer, faithfully ported from Xin Wang's LCNN baseline.
"""
import torch
from torch import nn
from torch.nn import Parameter


class P2SActivationLayer(nn.Module):
    """Cosine-similarity output layer used with P2SGrad training.

    Produces ``cos(theta)`` between an embedding vector and each class's
    weight vector, rather than an unbounded linear logit. Reference:
    Zhang, X. et al., "P2SGrad: Refined Gradients for Optimizing Deep Face
    Models", CVPR 2019.

    Ported without modification from ``core_modules/p2sgrad.py`` in
    project-NN-Pytorch-scripts (Xin Wang, NII), so that a checkpoint trained
    against that code loads here without a key mismatch. The output values
    are bounded in [-1, 1] rather than being conventional logits; they are
    still a valid ranking score for equal-error-rate computation, since a
    higher bonafide-class cosine similarity monotonically corresponds to a
    higher softmax probability of that class.

    Attributes:
        weight: Class weight vectors, shape (in_dim, out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the layer.

        Args:
            in_dim: Dimension of the input embedding.
            out_dim: Number of classes.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between the input and each class vector.

        Args:
            input_feat: Embedding of shape (batch, in_dim).

        Returns:
            Cosine similarities of shape (batch, out_dim), clamped to [-1, 1].
        """
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)

        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        inner_wx = input_feat.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        return cos_theta.clamp(-1, 1)
