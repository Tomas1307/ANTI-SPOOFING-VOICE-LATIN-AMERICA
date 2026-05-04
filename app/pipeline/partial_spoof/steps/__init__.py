"""
Step classes for Partial Spoof Pipeline.
"""
from app.pipeline.partial_spoof.steps.step_01_transcribe_bonafide import BonafideTranscriber
from app.pipeline.partial_spoof.steps.step_02_generate_cloned_speech import ClonedSpeechGenerator
from app.pipeline.partial_spoof.steps.step_03_forced_alignment import ForcedAligner
from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector
from app.pipeline.partial_spoof.steps.step_05_splice_audio import AudioSplicer
from app.pipeline.partial_spoof.steps.step_05b_apply_boundary_jitter import BoundaryJitterApplier
from app.pipeline.partial_spoof.steps.step_06_validate_splice import SpliceQualityValidator
from app.pipeline.partial_spoof.steps.step_07_format_output import OutputFormatter

__all__ = [
    "BonafideTranscriber",
    "ClonedSpeechGenerator",
    "ForcedAligner",
    "WordSelector",
    "AudioSplicer",
    "BoundaryJitterApplier",
    "SpliceQualityValidator",
    "OutputFormatter",
]
