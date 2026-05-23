"""
Abstract base class for per-attack voice cloning units.

Every attack pipeline exposes a concrete ``Cloner`` class in
``<attack>_attack/utils/cloner.py`` that subclasses ``BaseCloner`` and
implements its abstract methods using the attack's specific TTS SDK.
Both the standalone Step 3 (full-utterance attack pipeline) and the
partial_spoof Step 2 instantiate the same concrete Cloner via the
``cloner_dispatcher``, so the per-sample cloning logic exists in
exactly one place per attack.

This module lives under ``app/utils/`` so it stays venv-agnostic:
its dependencies are stdlib only (``abc``, ``pathlib``, ``typing``) and
importing it from any per-attack venv works cleanly. The concrete
Cloner subclasses pull in SDK-specific dependencies that only resolve
in their matching venv, so cross-venv imports fail at the SDK import
line (the intended isolation), not at the ABC import.

Contract enforcement notes:
    - Subclasses MUST override SYSTEM_ID (the uppercase attack identifier
      used to prefix output WAV filenames).
    - Subclasses MUST override NEEDS_REFERENCE_TRANSCRIPT to indicate
      whether their ``clone_single`` requires a non-empty ``reference_text``.
    - All four abstract methods (load, prepare_speaker, clone_single,
      cleanup) MUST be implemented. ``prepare_speaker`` is a no-op for
      attacks that do not need per-speaker state (Chatterbox, OmniVoice,
      FishGram) but must still exist on the subclass so the partial_spoof
      Step 2 loop can call it uniformly.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional, Tuple


class BaseCloner(ABC):
    """Abstract contract for per-attack voice cloning units.

    Subclasses live in ``<attack>_attack/utils/cloner.py`` and implement
    the abstract methods using the attack's specific TTS SDK. Both the
    standalone Step 3 (full-utterance attack pipeline) and partial_spoof
    Step 2 instantiate subclasses via ``cloner_dispatcher.get_cloner_class``
    and invoke the same lifecycle:

        cloner = CloneClass()
        cloner.load(device)
        for speaker_id, ref_path in ...:
            cloner.prepare_speaker(speaker_id, ref_path, reference_text)
            for text in ...:
                gen_time, duration = cloner.clone_single(
                    text=..., reference_audio_path=..., output_path=...,
                    reference_text=..., seed=...,
                )
        cloner.cleanup()

    Class attributes:
        SYSTEM_ID: Uppercase attack identifier (e.g. ``'OMNIVOICE'``).
            Used as the prefix for output WAV filenames in both
            pipelines. MUST be overridden by every concrete subclass.
        NEEDS_REFERENCE_TRANSCRIPT: True if ``clone_single`` requires a
            non-empty reference transcript (Qwen, OmniVoice). False
            otherwise. Callers consult this to decide whether to fetch
            the bonafide transcript before calling clone_single.
    """

    SYSTEM_ID: ClassVar[str] = ""
    NEEDS_REFERENCE_TRANSCRIPT: ClassVar[bool] = False

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model and any persistent state.

        Called once before any clone_single. Subclasses should load
        their TTS SDK model, apply any required patches (SDPA, watermark
        bypass, etc.), and cache device-bound resources. Raise on
        unrecoverable load failures (missing checkpoint, OOM, CUDA
        unavailable).

        Args:
            device: PyTorch device string (e.g. ``'cuda'``, ``'cuda:0'``,
                ``'cpu'``). Subclasses pass this through to their SDK's
                model loader.
        """

    @abstractmethod
    def prepare_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Path,
        reference_text: str = "",
    ) -> None:
        """Per-speaker setup. May be a no-op.

        Called once per speaker before the first clone_single call for
        that speaker. Implementations that cache per-speaker state
        (Qwen voice_clone_prompt, OpenVoice target_se, OuteTTS speaker
        profile) should populate their cache here keyed by
        ``reference_audio_path``. Implementations that don't need
        per-speaker state (Chatterbox, OmniVoice, FishGram) implement
        this as ``return None``.

        Args:
            speaker_id: HABLA speaker identifier. Used by some
                implementations only for log messages.
            reference_audio_path: Speaker reference audio path. The
                canonical cache key.
            reference_text: Optional reference transcript. Required by
                Qwen's ``create_voice_clone_prompt``; ignored by all
                other attacks. Default empty string is safe for the
                non-Qwen subclasses.
        """

    @abstractmethod
    def clone_single(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Generate ONE clone and write it to output_path.

        Args:
            text: Spanish text to synthesise.
            reference_audio_path: Speaker reference audio path. Cache
                key when the subclass uses per-speaker state from
                ``prepare_speaker``; passed directly to the SDK
                otherwise.
            output_path: Destination WAV path. Subclasses are
                responsible for resampling to ``settings.SAMPLE_RATE``
                (16 kHz) if they have a different native rate, except
                OmniVoice which writes at its native 24 kHz and lets
                downstream stages resample on load.
            reference_text: Optional reference transcript. Required by
                attacks where ``NEEDS_REFERENCE_TRANSCRIPT`` is True
                (Qwen, OmniVoice). Ignored otherwise.
            seed: Optional sampling seed. Accepted for interface
                uniformity; not every SDK honours it.

        Returns:
            Tuple of ``(generation_time_seconds, audio_duration_seconds)``.

        Raises:
            RuntimeError: On generation failure or if ``load()`` (or
                ``prepare_speaker()`` for stateful attacks) was not
                called first. Caller routes the exception through its
                recoverable-retry path.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release model state and clear CUDA memory.

        Called once after all clone_single calls complete. Subclasses
        should drop their model reference, clear per-speaker caches,
        and call ``torch.cuda.empty_cache()`` when applicable.
        """
