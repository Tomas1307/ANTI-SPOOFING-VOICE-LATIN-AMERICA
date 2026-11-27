"""
Fixed DCT-as-linear-layer, structurally ported from Xin Wang's LCNN baseline.
"""
from torch import nn


class LinearDCT(nn.Linear):
    """A Discrete Cosine Transform expressed as a non-trainable linear layer.

    The published baseline (``sandbox/util_dsp.py`` in
    project-NN-Pytorch-scripts, Xin Wang, NII) initialises this layer's weight
    from a closed-form DCT-II matrix and freezes it. This class does not
    reimplement that derivation: since every use in this project loads a
    pretrained checkpoint, the weight arrives already computed as
    ``l_dct.weight`` in the state dict, and reconstructing the DCT matrix from
    scratch would be pure risk for zero benefit. The class exists so the
    parameter shape, name and non-trainability match the checkpoint exactly.

    Attributes:
        weight: The (in_features, in_features) DCT matrix, loaded from a
            checkpoint rather than computed here. Frozen, per the original.
    """

    def __init__(self, in_features: int, bias: bool = False) -> None:
        """Initialize the layer with a placeholder weight.

        The weight is meaningless until ``load_state_dict`` overwrites it;
        this constructor only fixes the shape and freezes the parameter.

        Args:
            in_features: Length of the signal the DCT is applied to.
            bias: Whether to add a bias term. The published baseline uses
                none.
        """
        super().__init__(in_features, in_features, bias=bias)
        self.weight.requires_grad = False
