"""
Augmentation Mode Calculator

Calculates optimal augmentation factors for simple and balanced modes.
Handles the mathematical logic for balancing bonafide and spoof classes.
"""

import math
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class AugmentationFactors:
    """
    Container for calculated augmentation factors.
    
    Attributes:
        bonafide_factor: Augmentation factor for bonafide files
        spoof_factor: Augmentation factor for spoof files
        total_factor: Overall dataset augmentation factor
        final_ratio: Actual achieved ratio (bonafide_pct, spoof_pct)
        target_ratio: Target ratio requested (bonafide_pct, spoof_pct)
    """
    bonafide_factor: int
    spoof_factor: int
    total_factor: float
    final_ratio: Tuple[float, float]
    target_ratio: Tuple[float, float]
    
    def __str__(self):
        return (
            f"AugmentationFactors(\n"
            f"  bonafide_factor={self.bonafide_factor}x,\n"
            f"  spoof_factor={self.spoof_factor}x,\n"
            f"  total_factor={self.total_factor:.2f}x,\n"
            f"  target_ratio={self.target_ratio[0]:.1f}/{self.target_ratio[1]:.1f},\n"
            f"  final_ratio={self.final_ratio[0]:.1f}/{self.final_ratio[1]:.1f}\n"
            f")"
        )


class AugmentationModeCalculator:
    """
    Calculator for augmentation factors in simple and balanced modes.
    
    Provides methods to compute optimal augmentation factors based on:
    - Mode (simple vs balanced)
    - Current dataset composition
    - Target balance ratio
    - Minimum augmentation factor
    """
    
    @staticmethod
    def calculate_simple_mode(
        n_bonafide: int,
        n_spoof: int,
        factor: int
    ) -> AugmentationFactors:
        """
        Calculate factors for simple mode (uniform augmentation).
        
        Args:
            n_bonafide: Number of bonafide files
            n_spoof: Number of spoof files
            factor: Augmentation factor to apply uniformly
            
        Returns:
            AugmentationFactors with uniform factors
        """
        n_total = n_bonafide + n_spoof
        
        # Both classes get same factor
        bonafide_factor = factor
        spoof_factor = factor
        
        # Calculate totals
        total_bonafide = n_bonafide * bonafide_factor
        total_spoof = n_spoof * spoof_factor
        total_files = total_bonafide + total_spoof
        
        # Calculate ratios (percentages)
        bonafide_pct = (total_bonafide / total_files) * 100
        spoof_pct = (total_spoof / total_files) * 100
        
        # Original ratio stays the same
        orig_bonafide_pct = (n_bonafide / n_total) * 100
        orig_spoof_pct = (n_spoof / n_total) * 100
        
        # Total factor
        total_factor = total_files / n_total
        
        return AugmentationFactors(
            bonafide_factor=bonafide_factor,
            spoof_factor=spoof_factor,
            total_factor=total_factor,
            final_ratio=(bonafide_pct, spoof_pct),
            target_ratio=(orig_bonafide_pct, orig_spoof_pct)  # No change in simple mode
        )
    
    @staticmethod
    def calculate_balanced_mode(
        n_bonafide: int,
        n_spoof: int,
        target_ratio: float = 0.50,
        min_total_factor: int = 3
    ) -> AugmentationFactors:
        """
        Calculate factors for balanced mode.
        
        Computes optimal augmentation factors to achieve target ratio
        while respecting minimum total augmentation factor.
        
        Args:
            n_bonafide: Number of bonafide files
            n_spoof: Number of spoof files
            target_ratio: Target proportion of bonafide (0.0-1.0)
            min_total_factor: Minimum overall augmentation factor
            
        Returns:
            AugmentationFactors with optimized factors for balance
        """
        n_total = n_bonafide + n_spoof
        
        # Calculate minimum total files needed
        total_target_minimum = n_total * min_total_factor
        
        # Calculate target counts for each class based on ratio
        bonafide_target = total_target_minimum * target_ratio
        spoof_target = total_target_minimum * (1 - target_ratio)
        
        # Calculate raw factors
        bonafide_factor_raw = bonafide_target / n_bonafide
        spoof_factor_raw = spoof_target / n_spoof
        
        # Round factors
        # Use ceiling to ensure we meet minimum
        bonafide_factor = max(1, math.ceil(bonafide_factor_raw))
        spoof_factor = max(1, math.ceil(spoof_factor_raw))
        
        # Calculate actual totals with rounded factors
        total_bonafide = n_bonafide * bonafide_factor
        total_spoof = n_spoof * spoof_factor
        total_files = total_bonafide + total_spoof
        
        # Calculate actual ratio achieved
        bonafide_pct = (total_bonafide / total_files) * 100
        spoof_pct = (total_spoof / total_files) * 100
        
        # Target ratio as percentages
        target_bonafide_pct = target_ratio * 100
        target_spoof_pct = (1 - target_ratio) * 100
        
        # Total factor
        total_factor = total_files / n_total
        
        return AugmentationFactors(
            bonafide_factor=bonafide_factor,
            spoof_factor=spoof_factor,
            total_factor=total_factor,
            final_ratio=(bonafide_pct, spoof_pct),
            target_ratio=(target_bonafide_pct, target_spoof_pct)
        )
    
    @staticmethod
    def get_calculation_summary(
        n_bonafide: int,
        n_spoof: int,
        factors: 'AugmentationFactors',
        mode: str = "simple"
    ) -> str:
        """
        Build detailed summary of augmentation calculation.

        Args:
            n_bonafide: Original number of bonafide files
            n_spoof: Original number of spoof files
            factors: Calculated AugmentationFactors
            mode: Mode used ("simple" or "balanced")

        Returns:
            Formatted summary string.
        """
        lines = []
        lines.append("\n" + "="*70)
        lines.append(f"AUGMENTATION CALCULATION SUMMARY - {mode.upper()} MODE")
        lines.append("="*70)

        n_total = n_bonafide + n_spoof
        orig_bonafide_pct = (n_bonafide / n_total) * 100
        orig_spoof_pct = (n_spoof / n_total) * 100

        lines.append(f"\nORIGINAL DATASET:")
        lines.append(f"  Bonafide: {n_bonafide:,} files ({orig_bonafide_pct:.1f}%)")
        lines.append(f"  Spoof:    {n_spoof:,} files ({orig_spoof_pct:.1f}%)")
        lines.append(f"  Total:    {n_total:,} files")

        if mode == "balanced":
            lines.append(f"\nTARGET CONFIGURATION:")
            lines.append(f"  Target ratio: {factors.target_ratio[0]:.1f}% / {factors.target_ratio[1]:.1f}%")

        lines.append(f"\nCALCULATED FACTORS:")
        lines.append(f"  Bonafide factor: {factors.bonafide_factor}x")
        lines.append(f"  Spoof factor:    {factors.spoof_factor}x")
        lines.append(f"  Total factor:    {factors.total_factor:.2f}x")

        total_bonafide = n_bonafide * factors.bonafide_factor
        total_spoof = n_spoof * factors.spoof_factor
        total_files = total_bonafide + total_spoof

        lines.append(f"\nAUGMENTED DATASET:")
        lines.append(f"  Bonafide: {total_bonafide:,} files ({factors.final_ratio[0]:.1f}%) [+{total_bonafide - n_bonafide:,}]")
        lines.append(f"  Spoof:    {total_spoof:,} files ({factors.final_ratio[1]:.1f}%) [+{total_spoof - n_spoof:,}]")
        lines.append(f"  Total:    {total_files:,} files [+{total_files - n_total:,}]")

        if mode == "balanced":
            deviation_bonafide = abs(factors.final_ratio[0] - factors.target_ratio[0])
            deviation_spoof = abs(factors.final_ratio[1] - factors.target_ratio[1])

            lines.append(f"\nBALANCE ACHIEVED:")
            lines.append(f"  Target:    {factors.target_ratio[0]:.1f}% / {factors.target_ratio[1]:.1f}%")
            lines.append(f"  Achieved:  {factors.final_ratio[0]:.1f}% / {factors.final_ratio[1]:.1f}%")
            lines.append(f"  Deviation: ±{max(deviation_bonafide, deviation_spoof):.1f}%")

        lines.append("="*70 + "\n")

        return "\n".join(lines)

    @staticmethod
    def print_calculation_summary(
        n_bonafide: int,
        n_spoof: int,
        factors: 'AugmentationFactors',
        mode: str = "simple"
    ):
        """
        Print detailed summary of augmentation calculation.

        Args:
            n_bonafide: Original number of bonafide files
            n_spoof: Original number of spoof files
            factors: Calculated AugmentationFactors
            mode: Mode used ("simple" or "balanced")
        """
        print(AugmentationModeCalculator.get_calculation_summary(
            n_bonafide, n_spoof, factors, mode
        ))


def test_calculator():
    """Test the augmentation calculator."""
    calculator = AugmentationModeCalculator()
    
    # Test data
    n_bonafide = 80
    n_spoof = 176
    
    print("\n" + "="*70)
    print("TESTING AUGMENTATION CALCULATOR")
    print("="*70)
    
    # Test simple mode
    print("\nTest 1: Simple mode with 3x factor")
    factors_simple = calculator.calculate_simple_mode(n_bonafide, n_spoof, factor=3)
    calculator.print_calculation_summary(n_bonafide, n_spoof, factors_simple, mode="simple")
    
    # Test balanced mode 50/50
    print("\nTest 2: Balanced mode 50/50 with 3x minimum")
    factors_balanced_50 = calculator.calculate_balanced_mode(
        n_bonafide, n_spoof, target_ratio=0.50, min_total_factor=3
    )
    calculator.print_calculation_summary(n_bonafide, n_spoof, factors_balanced_50, mode="balanced")
    
    # Test balanced mode 60/40
    print("\nTest 3: Balanced mode 60/40 with 5x minimum")
    factors_balanced_60 = calculator.calculate_balanced_mode(
        n_bonafide, n_spoof, target_ratio=0.60, min_total_factor=5
    )
    calculator.print_calculation_summary(n_bonafide, n_spoof, factors_balanced_60, mode="balanced")
    
    # Test balanced mode 70/30
    print("\nTest 4: Balanced mode 70/30 with 3x minimum")
    factors_balanced_70 = calculator.calculate_balanced_mode(
        n_bonafide, n_spoof, target_ratio=0.70, min_total_factor=3
    )
    calculator.print_calculation_summary(n_bonafide, n_spoof, factors_balanced_70, mode="balanced")


if __name__ == "__main__":
    test_calculator()