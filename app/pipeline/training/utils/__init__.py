"""
Utilities for the detector training pipeline.

Nothing is re-exported here on purpose: protocol_io and metrics must stay
importable without torch or soundfile, which an eager re-export of
audio_dataset or training_checkpoint_manager would prevent.
"""
