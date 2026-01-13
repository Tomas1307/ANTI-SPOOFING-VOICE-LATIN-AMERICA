from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import glob
import torch
import pandas as pd
import soundfile as sf
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from datasets import Dataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

class AugmentationType(Enum):
    """Enumeration of available augmentation types."""
    
    RIR_NOISE = "rir_noise"
    CODEC = "codec"
    RAWBOOST = "rawboost"
    DEVICE_IR = "device_ir"


class RoomSize(Enum):
    """Room size categories for RIR selection."""
    
    SMALL = "smallroom"
    MEDIUM = "mediumroom"
    LARGE = "largeroom"


class Distance(Enum):
    """Source-to-microphone distance categories."""
    
    NEAR = "near"      # 10-50 cm
    MID = "mid"        # 50-100 cm
    FAR = "far"        # 100-150 cm


class NoiseSource(Enum):
    """MUSAN noise source categories."""
    
    NOISE = "noise"
    SPEECH = "speech"
    MUSIC = "music"

@dataclass
class SNRDistribution:
    """
    Signal-to-Noise Ratio distribution configuration.
    
    Attributes:
        low_range: SNR range for high noise conditions (min, max, probability).
        mid_range: SNR range for moderate noise conditions.
        high_range: SNR range for low noise conditions.
    """
    
    low_range: Tuple[float, float, float] = (0.0, 5.0, 0.10)
    mid_range: Tuple[float, float, float] = (5.0, 30.0, 0.80)
    high_range: Tuple[float, float, float] = (30.0, 35.0, 0.10)
    
    def validate(self):
        """Validate that probabilities sum to 1.0."""
        total_prob = self.low_range[2] + self.mid_range[2] + self.high_range[2]
        assert abs(total_prob - 1.0) < 0.01, f"SNR probabilities must sum to 1.0, got {total_prob}"


@dataclass
class RIRNoiseConfig:
    """
    Configuration for RIR and noise augmentation.
    
    Attributes:
        snr_distribution: SNR ranges and their probabilities.
        noise_sources: Probability distribution across MUSAN categories.
        room_sizes: Probability distribution for room types.
        distances: List of distance categories to sample from.
        t60_range: Range of reverberation time T60 in seconds.
    """
    
    snr_distribution: SNRDistribution = field(default_factory=SNRDistribution)
    noise_sources: Dict[NoiseSource, float] = field(default_factory=lambda: {
        NoiseSource.NOISE: 0.50,
        NoiseSource.SPEECH: 0.30,
        NoiseSource.MUSIC: 0.20
    })
    room_sizes: Dict[RoomSize, float] = field(default_factory=lambda: {
        RoomSize.SMALL: 0.30,
        RoomSize.MEDIUM: 0.50,
        RoomSize.LARGE: 0.20
    })
    distances: list = field(default_factory=lambda: [Distance.NEAR, Distance.MID, Distance.FAR])
    t60_range: Tuple[float, float] = (0.2, 1.2)
    
    def validate(self):
        """Validate configuration consistency."""
        self.snr_distribution.validate()
        
        noise_prob_sum = sum(self.noise_sources.values())
        assert abs(noise_prob_sum - 1.0) < 0.01, f"Noise source probs must sum to 1.0, got {noise_prob_sum}"
        
        room_prob_sum = sum(self.room_sizes.values())
        assert abs(room_prob_sum - 1.0) < 0.01, f"Room size probs must sum to 1.0, got {room_prob_sum}"


@dataclass
class CodecConfig:
    """
    Configuration for channel and codec degradation.
    
    Attributes:
        target_sample_rates: List of target sample rates for downsampling.
        packet_loss_range: Min and max packet loss percentage.
        codec_types: List of codec types to simulate.
        apply_probability: Probability of applying codec degradation.
    """
    
    target_sample_rates: list = field(default_factory=lambda: [8000, 16000])
    packet_loss_range: Tuple[float, float] = (0.01, 0.05)
    codec_types: list = field(default_factory=lambda: ["g711", "amr", "opus"])
    apply_probability: float = 1.0


@dataclass
class RawBoostConfig:
    """
    Configuration for RawBoost augmentation.
    
    Attributes:
        clipping_threshold: Threshold for audio clipping (0.0-1.0).
        apply_linear_filtering: Enable linear convolutive noise.
        apply_nonlinear_filtering: Enable nonlinear distortion.
        apply_additive_noise: Enable signal-dependent additive noise.
    """
    
    clipping_threshold: float = 0.9
    apply_linear_filtering: bool = True
    apply_nonlinear_filtering: bool = True
    apply_additive_noise: bool = True


@dataclass
class AugmentationStrategy:
    """
    Overall augmentation distribution strategy.
    
    Attributes:
        augmentation_factor: Number of augmented copies per original (3x, 5x, 10x).
        type_distribution: Probability distribution across augmentation types.
        rir_noise_config: Configuration for RIR+Noise augmentation.
        codec_config: Configuration for codec degradation.
        rawboost_config: Configuration for RawBoost augmentation.
        include_original: Whether to include unaugmented original in output.
    """
    
    augmentation_factor: int = 3
    type_distribution: Dict[AugmentationType, float] = field(default_factory=lambda: {
        AugmentationType.RIR_NOISE: 0.60,
        AugmentationType.CODEC: 0.30,
        AugmentationType.RAWBOOST: 0.10
    })
    rir_noise_config: RIRNoiseConfig = field(default_factory=RIRNoiseConfig)
    codec_config: CodecConfig = field(default_factory=CodecConfig)
    rawboost_config: RawBoostConfig = field(default_factory=RawBoostConfig)
    include_original: bool = True
    
    def validate(self):
        """Validate strategy configuration."""
        type_prob_sum = sum(self.type_distribution.values())
        assert abs(type_prob_sum - 1.0) < 0.01, f"Type distribution must sum to 1.0, got {type_prob_sum}"
        
        self.rir_noise_config.validate()
    
    def get_augmentation_counts(self, n_originals: int) -> Dict[str, int]:
        """
        Calculate number of samples for each augmentation type.
        
        Args:
            n_originals: Number of original audio samples.
            
        Returns:
            Dictionary mapping augmentation types to sample counts.
        """
        total_augmented = n_originals * self.augmentation_factor
        
        return {
            "original": n_originals if self.include_original else 0,
            "rir_noise": int(total_augmented * self.type_distribution[AugmentationType.RIR_NOISE]),
            "codec": int(total_augmented * self.type_distribution[AugmentationType.CODEC]),
            "rawboost": int(total_augmented * self.type_distribution[AugmentationType.RAWBOOST]),
            "total": n_originals + total_augmented if self.include_original else total_augmented
        }

@dataclass
class TTSDataCollatorWithPadding:
    """
    Custom data collator for text-to-speech tasks that handles dynamic padding
    for both text inputs and audio spectrograms while managing speaker embeddings.
    
    This collator ensures compatibility with SpeechT5's reduction factor by 
    rounding down spectrogram lengths to valid multiples, preventing dimension
    mismatch errors during loss computation.
    
    Attributes:
        processor: SpeechT5Processor instance for tokenization and padding.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples with proper padding and alignment.
        
        Args:
            features: List of dictionaries containing input_ids, labels, and speaker_embeddings.
            
        Returns:
            Dictionary containing padded and batched tensors ready for model input.
        """
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        speaker_features = [torch.tensor(feature["speaker_embeddings"]) for feature in features]
        label_features = [torch.tensor(feature["labels"]) for feature in features]

        batch = self.processor.pad(input_ids=input_ids, return_tensors="pt")

        reduction_factor = 2
        
        target_lengths = torch.tensor([label.shape[0] for label in label_features])
        if reduction_factor > 1:
            target_lengths = target_lengths.new(
                [length - length % reduction_factor for length in target_lengths]
            )
        
        label_features = [label[:target_lengths[i]] for i, label in enumerate(label_features)]
        
        max_label_length = max(label.shape[0] for label in label_features)
        feature_dim = label_features[0].shape[1]
        
        batch_size = len(label_features)
        padded_labels = torch.zeros(batch_size, max_label_length, feature_dim)
        
        labels_attention_mask = torch.zeros(batch_size, max_label_length, dtype=torch.long)
        
        for i, label in enumerate(label_features):
            length = label.shape[0]
            padded_labels[i, :length, :] = label
            labels_attention_mask[i, :length] = 1
        
        labels = padded_labels.masked_fill(
            labels_attention_mask.unsqueeze(-1).ne(1), -100
        )

        batch["labels"] = labels
        batch["speaker_embeddings"] = torch.stack(speaker_features)

        return batch
    
@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    text: str
    speaker_id: str
    audio_path: str
    generation_time: float
    mel_loss: float
    audio_length: float
    sample_rate: int