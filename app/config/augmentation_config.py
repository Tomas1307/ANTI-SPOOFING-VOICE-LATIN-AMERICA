"""
Augmentation Configuration Manager

Centralized singleton for managing all augmentation hyperparameters and
distribution strategies. Ensures consistent configuration across the entire
augmentation pipeline.

Based on research from:
- Sánchez et al. (2024) - RIR and noise augmentation
- ASVspoof 2019 - Channel degradation strategies
- Tak et al. (2022) - RawBoost methodology
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from app.schema import AugmentationType, AugmentationStrategy

class AugmentationConfigManager:
    """
    Singleton configuration manager for augmentation pipeline.
    
    Provides centralized access to augmentation hyperparameters and ensures
    consistent configuration across all pipeline components.
    
    Usage:
        config = AugmentationConfigManager.get_instance()
        strategy = config.get_strategy("3x")
    """
    
    _instance: Optional['AugmentationConfigManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._strategies: Dict[str, AugmentationStrategy] = {}
            self._load_default_strategies()
            self._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'AugmentationConfigManager':
        """
        Get singleton instance of configuration manager.
        
        Returns:
            Singleton AugmentationConfigManager instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_default_strategies(self):
        """Load default augmentation strategies for 3x, 5x, and 10x."""
        self._strategies["3x"] = AugmentationStrategy(
            augmentation_factor=3,
            type_distribution={
                AugmentationType.RIR_NOISE: 0.60,
                AugmentationType.CODEC: 0.30,
                AugmentationType.RAWBOOST: 0.10
            }
        )
        
        self._strategies["5x"] = AugmentationStrategy(
            augmentation_factor=5,
            type_distribution={
                AugmentationType.RIR_NOISE: 0.60,
                AugmentationType.CODEC: 0.30,
                AugmentationType.RAWBOOST: 0.10
            }
        )
        
        self._strategies["10x"] = AugmentationStrategy(
            augmentation_factor=10,
            type_distribution={
                AugmentationType.RIR_NOISE: 0.60,
                AugmentationType.CODEC: 0.30,
                AugmentationType.RAWBOOST: 0.10
            }
        )
    
    def get_strategy(self, factor: str = "3x") -> AugmentationStrategy:
        """
        Retrieve augmentation strategy by factor.
        
        Args:
            factor: Augmentation factor (e.g., "3x", "5x", "7x", "10x", or any number).
            
        Returns:
            AugmentationStrategy configuration object.
            
        Raises:
            KeyError: If factor is not found in available strategies.
            ValueError: If factor format is invalid.
        """
        # Check if factor exists in pre-configured strategies
        if factor in self._strategies:
            return self._strategies[factor]
        
        # Try to parse custom factor (e.g., "7x", "15x")
        if factor.endswith('x'):
            try:
                factor_num = int(factor[:-1])
                # Create strategy on-the-fly for custom factor
                custom_strategy = AugmentationStrategy(
                    augmentation_factor=factor_num,
                    type_distribution={
                        AugmentationType.RIR_NOISE: 0.60,
                        AugmentationType.CODEC: 0.30,
                        AugmentationType.RAWBOOST: 0.10
                    }
                )
                return custom_strategy
            except ValueError:
                raise ValueError(f"Invalid factor format: '{factor}'. Expected format: '3x', '5x', etc.")
        
        raise KeyError(f"Strategy '{factor}' not found. Available: {list(self._strategies.keys())} or custom format like '7x'")
    
    def register_strategy(self, name: str, strategy: AugmentationStrategy):
        """
        Register a custom augmentation strategy.
        
        Args:
            name: Unique identifier for the strategy.
            strategy: AugmentationStrategy object to register.
        """
        strategy.validate()
        self._strategies[name] = strategy
    
    def get_all_strategies(self) -> Dict[str, AugmentationStrategy]:
        """
        Get all registered strategies.
        
        Returns:
            Dictionary mapping strategy names to AugmentationStrategy objects.
        """
        return self._strategies.copy()
    
    def print_strategy_summary(self, factor: str = "3x"):
        """
        Print human-readable summary of a strategy.
        
        Args:
            factor: Augmentation factor to summarize.
        """
        strategy = self.get_strategy(factor)
        
        print(f"\n{'='*70}")
        print(f"AUGMENTATION STRATEGY: {factor}")
        print(f"{'='*70}")
        
        print(f"\nAugmentation Factor: {strategy.augmentation_factor}x")
        print(f"Include Original: {strategy.include_original}")
        
        print(f"\nType Distribution:")
        for aug_type, prob in strategy.type_distribution.items():
            print(f"  - {aug_type.value}: {prob*100:.1f}%")
        
        print(f"\nRIR + Noise Configuration:")
        print(f"  SNR Distribution:")
        snr = strategy.rir_noise_config.snr_distribution
        print(f"    - Low (0-5 dB): {snr.low_range[2]*100:.1f}%")
        print(f"    - Mid (5-30 dB): {snr.mid_range[2]*100:.1f}%")
        print(f"    - High (30-35 dB): {snr.high_range[2]*100:.1f}%")
        
        print(f"  Noise Sources:")
        for source, prob in strategy.rir_noise_config.noise_sources.items():
            print(f"    - {source.value}: {prob*100:.1f}%")
        
        print(f"  Room Sizes:")
        for room, prob in strategy.rir_noise_config.room_sizes.items():
            print(f"    - {room.value}: {prob*100:.1f}%")
        
        print(f"  T60 Range: {strategy.rir_noise_config.t60_range[0]}-{strategy.rir_noise_config.t60_range[1]}s")
        
        print(f"\nCodec Configuration:")
        print(f"  Target Sample Rates: {strategy.codec_config.target_sample_rates}")
        print(f"  Packet Loss Range: {strategy.codec_config.packet_loss_range[0]*100:.1f}%-{strategy.codec_config.packet_loss_range[1]*100:.1f}%")
        
        print(f"\nRawBoost Configuration:")
        print(f"  Clipping Threshold: {strategy.rawboost_config.clipping_threshold}")
        print(f"  Linear Filtering: {strategy.rawboost_config.apply_linear_filtering}")
        print(f"  Nonlinear Filtering: {strategy.rawboost_config.apply_nonlinear_filtering}")
        
        print(f"\n{'='*70}\n")
    
    def calculate_dataset_sizes(self, n_train: int, factor: str = "3x") -> Dict[str, int]:
        """
        Calculate expected dataset sizes after augmentation.
        
        Args:
            n_train: Number of original training samples.
            factor: Augmentation factor to apply.
            
        Returns:
            Dictionary with sample counts per augmentation type.
        """
        strategy = self.get_strategy(factor)
        return strategy.get_augmentation_counts(n_train)


def main():
    """
    Demonstration of configuration manager usage.
    """
    config = AugmentationConfigManager.get_instance()
    
    for factor in ["3x", "5x", "7x", "10x", "15x"]:
        config.print_strategy_summary(factor)
        
        sizes = config.calculate_dataset_sizes(18204, factor)
        print(f"Dataset sizes for {factor} augmentation:")
        for key, value in sizes.items():
            print(f"  {key}: {value:,}")
        print()


if __name__ == "__main__":
    main()