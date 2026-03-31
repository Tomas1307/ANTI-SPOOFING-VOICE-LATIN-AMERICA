"""Partial Spoof Pipeline - Creates partially spoofed Latin American Spanish audio.

This pipeline replaces individual words in bonafide HABLA utterances with
voice-cloned versions from configurable attack systems, producing partial
spoof samples at controlled word-count tiers (W1, W2, W3).
"""
from app.pipeline.partial_spoof.pipeline_facade import PartialSpoofPipeline

__all__ = ["PartialSpoofPipeline"]
