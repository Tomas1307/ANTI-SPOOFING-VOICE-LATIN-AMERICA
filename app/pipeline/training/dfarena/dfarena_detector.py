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

    DF-Arena is not a self-supervised backbone waiting for a classifier: it is
    a complete detector that consumes a raw waveform and emits two class
    logits. An earlier draft of this class bolted a freshly initialised pooled
    head onto it, which would have discarded a trained output layer and
    crashed besides, because the model returns a plain dict rather than a
    Hugging Face ModelOutput. Reading the published modeling source settled
    both points:

        def forward(self, input_values, attention_mask=None):
            logits = self.backbone(input_values)
            return {"logits": logits}

    Two consequences the wrapper has to respect. The returned object is a
    dict, so ``outputs["logits"]`` is the only correct access. And
    ``attention_mask`` is accepted but never forwarded to the backbone, so
    padding is not masked anywhere inside the model.

    The published feature extractor pins the input contract to exactly 64,600
    samples, truncating longer clips and tiling shorter ones. This class
    declares that through ``required_samples``, which makes the dataset crop
    every clip to precisely that length. Since every clip then has identical
    length, no padding exists and the unmasked attention is harmless.

    The model's own label order, ``{1: bonafide, 0: spoof}``, already matches
    the project convention, so no index swap is needed anywhere.

    Attributes:
        model: The published DF-Arena model.
        frozen: Whether the weights are excluded from optimisation.
    """

    REQUIRED_SAMPLES = 64600

    def __init__(
        self,
        model_id: str,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize the detector.

        Args:
            model_id: Hugging Face repository identifier of the model.
            freeze_backbone: Whether to freeze every published weight. Unlike
                a backbone-plus-head design there is no separate head to train
                afterwards, so freezing leaves nothing trainable; it is useful
                only for a pure inference run.

        Raises:
            ValueError: If the published label order is not the expected one,
                which would silently invert every score.
        """
        super().__init__()

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self._assert_label_order(config)

        self.model = AutoModel.from_pretrained(
            model_id, config=config, trust_remote_code=True
        )
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

        Args:
            waveform: Waveforms of shape (batch, samples). Every clip is
                expected to carry exactly ``required_samples`` samples, which
                the dataset guarantees.
            lengths: True sample count per item. Unused: the model ignores its
                own attention mask, and the fixed-length contract means there
                is no padding to mask. Accepted to satisfy the interface.

        Returns:
            Class logits of shape (batch, 2), spoof at index 0 and bonafide at
            index 1.
        """
        outputs = self.model(input_values=waveform)
        return outputs["logits"]

    def parameter_groups(
        self, head_learning_rate: float, backbone_learning_rate: float
    ) -> List[Dict[str, Any]]:
        """Build optimiser parameter groups.

        DF-Arena publishes a trained output layer rather than exposing a fresh
        head, so there is no group that warrants the larger head learning
        rate. Every trainable weight is pretrained and goes into a single
        group at the backbone rate.

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
        return [
            {
                "params": trainable,
                "lr": backbone_learning_rate,
                "name": "dfarena",
            }
        ]

    @staticmethod
    def _assert_label_order(config) -> None:
        """Verify the published label order matches the project convention.

        The project scores bonafide as class 1 so that a higher score means
        more genuine. DF-Arena publishes the same mapping. Checking rather
        than assuming matters because a silent mismatch inverts every score
        and turns a good detector into an apparently terrible one, or worse,
        the reverse.

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
