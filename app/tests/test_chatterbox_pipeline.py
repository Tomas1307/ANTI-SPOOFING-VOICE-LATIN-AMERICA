"""
Test script for Chatterbox Attack Pipeline (Validation Mode)

Runs the full 5-step pipeline on 3 speakers with 2 samples each (6 total).
"""
from app.pipeline.chatterbox_attack import ChatterboxAttackPipeline
from app.pipeline.chatterbox_attack.settings import settings

print("=" * 80)
print("CHATTERBOX ATTACK PIPELINE - VALIDATION MODE TEST")
print("=" * 80)
print()

# Configure validation mode
settings.VALIDATION_MODE = True
settings.SAMPLES_PER_SPEAKER = 2

print(f"Validation mode: {settings.VALIDATION_MODE}")
print(f"Speakers: {settings.VALIDATION_SPEAKERS}")
print(f"Samples per speaker: {settings.SAMPLES_PER_SPEAKER}")
print()

# Run pipeline
pipeline = ChatterboxAttackPipeline()
output_dir = pipeline.run()

print()
print("=" * 80)
print(f"PIPELINE COMPLETE!")
print(f"Output directory: {output_dir}")
print("=" * 80)
