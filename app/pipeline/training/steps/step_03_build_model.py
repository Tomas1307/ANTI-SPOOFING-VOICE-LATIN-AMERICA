"""
Step 3: instantiate the detector backend named by the configuration.
"""
from typing import Callable, Dict

import torch
from loguru import logger

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector
from app.pipeline.training.dfarena.dfarena_detector import DFArenaDetector
from app.pipeline.training.dfarena.settings import settings as dfarena_settings
from app.pipeline.training.schemas.pipeline_config import DetectorTrainingConfig


class DetectorFactory:
    """Create detector backends from a registry keyed by name.

    Adding Nes2Net, HoliAntiSpoof or the LFCC-LCNN baseline means writing a
    subpackage beside dfarena with a subclass of BaseSpoofDetector, and adding
    one entry to the registry below. No other part of the pipeline changes,
    because every step downstream speaks only the abstract interface.

    Attributes:
        config: Run configuration naming the backend and its hyperparameters.
        device: Device the model is moved onto.
    """

    def __init__(self, config: DetectorTrainingConfig, device: torch.device) -> None:
        """Initialize the factory.

        Args:
            config: Run configuration.
            device: Device the model is moved onto.
        """
        self.config = config
        self.device = device
        self._registry: Dict[str, Callable[[], BaseSpoofDetector]] = {
            "dfarena": self._build_dfarena,
        }

    def execute(self) -> BaseSpoofDetector:
        """Build the configured detector.

        Returns:
            The detector, moved onto the target device.

        Raises:
            ValueError: If the configured backend is not registered.
        """
        logger.info(f"Step {self.__class__.__name__}: Starting")
        backend = self.config.detector_backend
        if backend not in self._registry:
            raise ValueError(
                f"Unknown detector backend '{backend}'; "
                f"registered backends: {sorted(self._registry)}"
            )

        model = self._registry[backend]().to(self.device)
        logger.info(f"Step {self.__class__.__name__}: Complete")
        return model

    def _build_dfarena(self) -> BaseSpoofDetector:
        """Construct the DF-Arena detector.

        The backbone identifier comes from the run configuration when one was
        supplied, and from the backend settings otherwise.

        Returns:
            An initialised DFArenaDetector.
        """
        return DFArenaDetector(
            model_id=self.config.model_id or dfarena_settings.MODEL_ID,
            hidden_dim=dfarena_settings.CLASSIFIER_HIDDEN_DIM,
            dropout=dfarena_settings.CLASSIFIER_DROPOUT,
            freeze_backbone=self.config.freeze_backbone,
            normalize_input=dfarena_settings.NORMALIZE_INPUT,
        )
