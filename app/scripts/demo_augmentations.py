"""
Augmentation Demo Generator (for presentation playback).

Produces, for a small set of representative bonafide and attack samples, the
original clip plus one isolated example of each augmentation family, so each
effect can be auditioned independently during the thesis presentation:

    original      - loudness-normalized source (reference)
    rir           - room reverberation only (RIRS_NOISES medium room)
    noise         - MUSAN ambient noise added at a fixed SNR
    babble        - MUSAN speech (background talkers) added at a fixed SNR
    codec_g711    - narrowband telephony codec (G.711 mu-law, 8 kHz)
    codec_opus    - broadband VoIP codec (Opus)
    rawboost      - RawBoost LnL+ISD+SSI (algo 4)

Each effect is applied in ISOLATION (not the stacked production pipeline) so the
audience can hear exactly what each augmentation does. Every output is passed
through the same uniform loudness normalization used by the production pipeline,
so A/B differences are timbre/channel, not level.

Outputs go to ``data/demo_augmentations/<label>/<variant>.wav`` plus a
``manifest.json``. The presentation references these paths relative to the repo
root.

Run on ml-server03 (needs torch/torchaudio + ffmpeg for the codec variants):
    python -m app.scripts.demo_augmentations
    python -m app.scripts.demo_augmentations --sample bonafide data/bonafide_dataset_by_speaker_v2/arf_00295/train/xxx.wav
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.augmenter import codec_backend
from app.augmenter.rawboost_reference import process_Rawboost_feature
from app.augmenter.schemas.codec_rawboost_config import RawBoostParams
import app.utils.utils as utils

SAMPLE_RATE = 16000


class DemoAugmentationGenerator:
    """
    Generate isolated per-augmentation demo clips for presentation playback.

    Attributes:
        rir_root: Root of the RIRS_NOISES dataset.
        musan_root: Root of the MUSAN dataset.
        output_root: Where demo clips are written.
        loudness_dbfs: Uniform loudness target applied to every output.
        snr_db: SNR used for the additive-noise demos.
    """

    def __init__(
        self,
        rir_root: str = "data/noise_dataset/RIR",
        musan_root: str = "data/noise_dataset/musan",
        output_root: str = "data/demo_augmentations",
        loudness_dbfs: float = -23.0,
        snr_db: float = 10.0,
        seed: int = 42,
    ):
        """
        Initialize the demo generator and index the RIR / MUSAN sources.

        Args:
            rir_root: RIRS_NOISES root directory.
            musan_root: MUSAN root directory.
            output_root: Output directory for demo clips.
            loudness_dbfs: Uniform RMS loudness target (dBFS).
            snr_db: SNR (dB) for the additive-noise and babble demos.
            seed: Random seed for reproducible RIR/noise picks.
        """
        random.seed(seed)
        np.random.seed(seed)

        self.rir_root = Path(rir_root)
        self.musan_root = Path(musan_root)
        self.output_root = Path(output_root)
        self.loudness_dbfs = loudness_dbfs
        self.snr_db = snr_db
        self.rawboost_params = RawBoostParams()

        self.rir_files = self._index(self.rir_root / "simulated_rirs" / "mediumroom")
        self.noise_files = self._index(self.musan_root / "noise")
        self.speech_files = self._index(self.musan_root / "speech")

        print("DemoAugmentationGenerator initialized:")
        print(f"  - Medium-room RIRs: {len(self.rir_files)}")
        print(f"  - MUSAN noise files: {len(self.noise_files)}")
        print(f"  - MUSAN speech files: {len(self.speech_files)}")

    def _index(self, root: Path) -> List[str]:
        """Recursively index .wav/.flac files under a directory."""
        if not root.exists():
            print(f"  Warning: source path not found: {root}")
            return []
        files: List[str] = []
        for ext in ("*.wav", "*.flac"):
            files.extend(str(p) for p in root.rglob(ext))
        return files

    def _norm(self, audio: np.ndarray) -> np.ndarray:
        """Apply the uniform loudness policy used by the production pipeline."""
        return utils.normalize_loudness(audio, self.loudness_dbfs)

    def _rir(self, audio: np.ndarray) -> np.ndarray:
        """Room reverberation only."""
        rir, _ = utils.load_audio(random.choice(self.rir_files), sr=SAMPLE_RATE)
        return utils.convolve_with_rir(audio, rir)

    def _add_noise(self, audio: np.ndarray, files: List[str]) -> np.ndarray:
        """Additive noise from the given MUSAN subset at the configured SNR."""
        noise, _ = utils.load_audio(random.choice(files), sr=SAMPLE_RATE)
        return utils.mix_audio_with_snr(audio, noise, self.snr_db)

    def _codec(self, audio: np.ndarray, codec_name: str) -> Optional[np.ndarray]:
        """Real codec round-trip for the named codec."""
        spec = codec_backend.DEFAULT_CODEC_REGISTRY[codec_name]
        bitrate = spec.bitrates[len(spec.bitrates) // 2] if spec.bitrates else None
        return codec_backend.apply_codec(audio, spec, bitrate)

    def _rawboost(self, audio: np.ndarray) -> np.ndarray:
        """RawBoost LnL+ISD+SSI (algo 4)."""
        out = process_Rawboost_feature(
            np.asarray(audio, dtype=np.float64), SAMPLE_RATE, self.rawboost_params, 4
        )
        return np.asarray(out, dtype=np.float32)

    def generate_for_sample(self, label: str, filepath: str) -> Dict[str, str]:
        """
        Produce all demo variants for one source clip and write them to disk.

        Args:
            label: Short identifier (e.g. "bonafide", "qwen").
            filepath: Path to the source audio.

        Returns:
            Mapping of variant name to the written relative path (skipped
            variants are omitted).
        """
        audio, _ = utils.load_audio(filepath, sr=SAMPLE_RATE)
        out_dir = self.output_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        variants: Dict[str, Optional[np.ndarray]] = {
            "original": audio,
            "rir": self._rir(audio) if self.rir_files else None,
            "noise": self._add_noise(audio, self.noise_files) if self.noise_files else None,
            "babble": self._add_noise(audio, self.speech_files) if self.speech_files else None,
            "codec_g711": self._codec(audio, "g711_ulaw"),
            "codec_opus": self._codec(audio, "opus"),
            "rawboost": self._rawboost(audio),
        }

        written: Dict[str, str] = {}
        for name, signal in variants.items():
            if signal is None:
                print(f"  [{label}] {name}: skipped (source/codec unavailable)")
                continue
            path = out_dir / f"{name}.wav"
            self._save_wav(self._norm(signal), path)
            written[name] = str(path).replace("\\", "/")
        print(f"  [{label}] wrote {len(written)} variants to {out_dir}")
        return written

    def _save_wav(self, audio: np.ndarray, path: Path) -> None:
        """Write a mono 16 kHz WAV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, format="WAV")

    def run(self, samples: List[Tuple[str, str]]) -> None:
        """
        Generate demo clips for every (label, filepath) sample and write a manifest.

        Args:
            samples: List of (label, filepath) pairs.
        """
        manifest: Dict[str, Dict[str, str]] = {}
        for label, filepath in samples:
            if not Path(filepath).exists():
                print(f"  [{label}] source not found, skipping: {filepath}")
                continue
            manifest[label] = self.generate_for_sample(label, filepath)

        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written: {manifest_path}")

    def autodiscover_samples(self) -> List[Tuple[str, str]]:
        """
        Best-effort discovery of one bonafide + one sample per attack pipeline.

        Returns:
            List of (label, filepath) pairs for sources that were found.
        """
        candidates: Dict[str, List[str]] = {
            "bonafide": [
                "data/bonafide_dataset_by_speaker_v2",
                "data/partition_dataset_by_speaker",
            ],
            "fishgram": ["data/fishgram_output"],
            "qwen": ["data/qwen_output"],
            "omnivoice": ["data/omnivoice_output"],
            "openvoice": ["data/openvoice_output"],
            "chatterbox": ["data/chatterbox_output"],
            "outetts": ["data/outetts_output"],
        }
        samples: List[Tuple[str, str]] = []
        for label, roots in candidates.items():
            for root in roots:
                found = self._index(Path(root))
                if found:
                    samples.append((label, sorted(found)[0]))
                    break
        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-augmentation demo clips for the presentation."
    )
    parser.add_argument(
        "--sample", nargs=2, action="append", metavar=("LABEL", "PATH"),
        help="A labelled source clip; repeatable. If omitted, sources are auto-discovered.",
    )
    parser.add_argument("--rir", default="data/noise_dataset/RIR")
    parser.add_argument("--musan", default="data/noise_dataset/musan")
    parser.add_argument("--output", default="data/demo_augmentations")
    parser.add_argument("--snr", type=float, default=10.0, help="SNR (dB) for noise/babble demos")
    parser.add_argument("--loudness_dbfs", type=float, default=-23.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generator = DemoAugmentationGenerator(
        rir_root=args.rir,
        musan_root=args.musan,
        output_root=args.output,
        loudness_dbfs=args.loudness_dbfs,
        snr_db=args.snr,
        seed=args.seed,
    )

    sample_list = (
        [(label, path) for label, path in args.sample]
        if args.sample else generator.autodiscover_samples()
    )
    if not sample_list:
        raise SystemExit(
            "No samples found. Pass them explicitly with --sample LABEL PATH."
        )

    generator.run(sample_list)
