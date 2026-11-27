"""
DF-Arena detector: a published end-to-end anti-spoofing model, wrapped.
"""
from typing import Any, Dict, List

import torch
from loguru import logger
from transformers import AutoConfig, AutoModel

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector


class DFArenaDetector(BaseSpoofDetector):
    """Adapter over the published DF-Arena anti-spoofing model.

    DF-Arena is a complete detector, not a backbone awaiting a classifier: a
    wav2vec2-XLS-R-1B encoder, an attention-weighted fusion over all 25 hidden
    layers, and a four-block conformer whose class token feeds a two-way
    output. Waveform in, two logits out.

    WHY THIS CLASS REIMPLEMENTS THE FORWARD PASS
    --------------------------------------------
    The published backbone hard-codes a batch size of one:

        def forward(self, x):
            out_ssl = self.ssl_model(x.unsqueeze(0))

    That ``unsqueeze(0)`` assumes ``x`` is a single waveform of shape (T,).
    Handed a real batch of shape (B, T) it produces (1, B, T), and wav2vec2's
    convolutional front end then rejects the resulting four-dimensional
    tensor. Every other operation in the published pipeline is already
    batch-general: the layer pooling reduces over ``dim=1`` of (B, T, D), and
    the conformer iterates the batch to prepend its class token.

    So this class calls the same submodules in the same order with genuinely
    batched input, skipping only that one line. The weights and the arithmetic
    are identical; scoring 30,000 clips one at a time is merely avoided.
    Equivalence is not assumed: ``app/scripts/verify_dfarena_batching.py``
    compares this path against the published per-clip path numerically.

    One real difference under fine-tuning: ``first_bn`` is a BatchNorm2d. In
    evaluation mode it uses running statistics, so batching cannot change a
    single score. In training mode with a batch larger than one it normalises
    across the batch rather than over a single sample, which is a departure
    from how the published weights were trained and belongs in the methods.

    The input contract comes from the published feature extractor: exactly
    64,600 samples, truncating longer clips and tiling shorter ones. The label
    order ``{1: bonafide, 0: spoof}`` matches the project convention and is
    asserted rather than assumed.

    Attributes:
        model: The published DF-Arena model.
        frozen: Whether the weights are excluded from optimisation.
    """

    REQUIRED_SAMPLES = 64600
    REQUIRED_BACKBONE_ATTRIBUTES = (
        "ssl_model",
        "get_attenF1D",
        "fc0",
        "sig",
        "first_bn",
        "selu",
        "conformer",
    )

    def __init__(self, model_id: str, freeze_backbone: bool = False) -> None:
        """Initialize the detector.

        Args:
            model_id: Hugging Face repository identifier of the model.
            freeze_backbone: Whether to freeze every published weight. There is
                no separate head to train afterwards, so freezing leaves
                nothing trainable; it is useful only for a pure inference run.

        Raises:
            ValueError: If the published label order is not the expected one,
                or if the backbone does not expose the submodules this adapter
                drives.
        """
        super().__init__()

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self._assert_label_order(config)

        self.model = AutoModel.from_pretrained(
            model_id, config=config, trust_remote_code=True
        )
        self._assert_backbone_shape()

        self.required_samples = self.REQUIRED_SAMPLES
        self.frozen = freeze_backbone

        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DFArenaDetector ready: {model_id}, "
            f"required input {self.REQUIRED_SAMPLES:,} samples "
            f"({self.REQUIRED_SAMPLES / 16000:.4f} s at 16 kHz), "
            f"trainable {trainable:,}/{total:,} parameters, frozen={freeze_backbone}"
        )

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Score a batch of waveforms.

        This mirrors ``DF_Arena_1B.forward`` operation for operation, differing
        only in passing the batch straight to the encoder instead of
        unsqueezing a single clip into one.

        Args:
            waveform: Waveforms of shape (batch, samples). Every clip carries
                exactly ``required_samples`` samples, which the dataset
                guarantees.
            lengths: True sample count per item. Unused: the fixed-length
                contract leaves no padding to mask, and the published model
                ignores its own attention mask. Accepted to satisfy the
                interface.

        Returns:
            Class logits of shape (batch, 2), spoof at index 0 and bonafide at
            index 1.
        """
        backbone = self.model.backbone

        encoded = backbone.ssl_model(waveform)
        pooled, layer_features = backbone.get_attenF1D(encoded.hidden_states)

        layer_weights = backbone.sig(backbone.fc0(pooled))
        layer_weights = layer_weights.view(
            layer_weights.shape[0], layer_weights.shape[1], layer_weights.shape[2], -1
        )

        fused = (layer_features * layer_weights).sum(dim=1).unsqueeze(dim=1)
        fused = backbone.selu(backbone.first_bn(fused))

        logits, _attention = backbone.conformer(fused.squeeze(1))
        return logits

    def parameter_groups(
        self, head_learning_rate: float, backbone_learning_rate: float
    ) -> List[Dict[str, Any]]:
        """Build optimiser parameter groups.

        DF-Arena publishes a trained output layer rather than exposing a fresh
        head, so no group warrants the larger head learning rate. Every
        trainable weight is pretrained and goes into a single group.

        Args:
            head_learning_rate: Ignored; retained to satisfy the interface.
            backbone_learning_rate: Peak learning rate for the whole model.

        Returns:
            One parameter group covering every trainable weight.

        Raises:
            ValueError: If the model is frozen, since nothing would train.
        """
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError(
                "DFArenaDetector is frozen, so there is nothing to optimise. "
                "Drop --freeze-backbone to fine-tune, or use --eval-only to "
                "score the published weights without training."
            )
        return [{"params": trainable, "lr": backbone_learning_rate, "name": "dfarena"}]

    def published_forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Score a batch through the published per-clip path, one clip at a time.

        This exists so the batched forward can be checked against the model as
        its authors published it. It is slow by construction and is not used
        during training or evaluation.

        Args:
            waveform: Waveforms of shape (batch, samples).

        Returns:
            Class logits of shape (batch, 2).
        """
        return torch.cat(
            [self.model(input_values=clip)["logits"] for clip in waveform], dim=0
        )

    def _assert_backbone_shape(self) -> None:
        """Verify the published backbone exposes what this adapter drives.

        The batched forward reaches into the backbone's submodules. If a future
        revision of the model renames or restructures them, failing loudly here
        beats failing obscurely inside a forward pass, or worse, silently
        producing wrong scores.

        Raises:
            ValueError: If the backbone is missing or lacks an expected member.
        """
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            raise ValueError(
                "The published model exposes no 'backbone' attribute. Its "
                "structure has changed; review the modeling source before "
                "using this adapter."
            )
        missing = [
            name
            for name in self.REQUIRED_BACKBONE_ATTRIBUTES
            if not hasattr(backbone, name)
        ]
        if missing:
            raise ValueError(
                f"The published backbone is missing {missing}. Its structure "
                "has changed; the batched forward in this adapter mirrors the "
                "published one and must be reviewed against the new source."
            )

    @staticmethod
    def _assert_label_order(config) -> None:
        """Verify the published label order matches the project convention.

        The project scores bonafide as class 1 so that a higher score means
        more genuine. Checking rather than assuming matters because a silent
        mismatch inverts every score, turning a good detector into an
        apparently terrible one, or the reverse.

        Args:
            config: The loaded model configuration.

        Raises:
            ValueError: If the mapping is present and does not match.
        """
        mapping = getattr(config, "id2label", None)
        if not mapping:
            logger.warning(
                "Model publishes no id2label mapping; assuming index 1 is "
                "bonafide, per the project convention."
            )
            return

        normalised = {int(key): str(value).lower() for key, value in mapping.items()}
        if normalised.get(1) != "bonafide" or normalised.get(0) != "spoof":
            raise ValueError(
                f"Unexpected label order {normalised}. This project scores "
                "bonafide as class 1; a mismatch would invert every score. "
                "Adapt the wrapper deliberately rather than proceeding."
            )
        logger.info(f"Label order verified: {normalised}")
