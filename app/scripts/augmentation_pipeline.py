"""
Augmentation Pipeline for Anti-Spoofing Dataset
================================================

Creates augmented dataset in ASVspoof2019 LA format with state-of-the-art practices:

AUGMENTATION STRATEGY:
---------------------

1. SPEAKER-INDEPENDENT SPLITS:
   - Train/Val/Test use DIFFERENT speakers (no speaker overlap)
   - Prevents model from learning speaker-specific features
   - Ensures true generalization evaluation

2. UNIFORM FACTOR (no corpus-side rebalancing):
   Both classes receive the same augmentation factor, so the clean fraction
   (1 / factor) is identical for bonafide and spoof by construction; "is this
   clip augmented?" therefore carries no information about the label. The
   corpus's natural class imbalance is preserved and left to be corrected at
   training time (class weights or a balanced sampler), not baked into the
   released data.

3. CLEAN FRACTION (equal across classes, derived from the factor):
   - TRAIN: each original file emits one clean copy plus (factor - 1)
     augmented copies, for both classes alike.
   - VAL/TEST: 100% clean (NO augmentation).
   - LOUDNESS: every emitted clip (clean and all augmentation types, every
     split) is passed through one uniform loudness normalization so loudness
     reveals nothing about augmentation type or class.

4. AUGMENTATION TYPES (Applied to train only):
   - RIR + Noise (60%): Room acoustics + background noise (single) or first pass in stacked
   - Codec Degradation (40%): Telephone/VoIP artifacts (single) or second pass in stacked
   - Stacking gate (40% of aug clips): RIR_NOISE then CODEC applied in sequence;
     SYSTEM_ID uses "|" separator (e.g. "RIR_SMALL_NOI_SNR15|CODEC_8K_OPUS")
   - RawBoost: training-time on-the-fly only — NOT pre-baked in this offline corpus

OUTPUT STRUCTURE (ASVspoof2019 LA Format):
------------------------------------------

    data/augmented_{factor}/
    └── LA/
        ├── ASVspoof2019_LA_train/
        │   ├── flac/
        │   │   ├── LA_T_0000001.flac
        │   │   └── ...
        │   └── ASVspoof2019.LA.cm.train.trn.txt
        ├── ASVspoof2019_LA_dev/
        │   ├── flac/
        │   │   ├── LA_D_0000001.flac
        │   │   └── ...
        │   └── ASVspoof2019.LA.cm.dev.trl.txt
        └── ASVspoof2019_LA_eval/
            ├── flac/
            │   ├── LA_E_0000001.flac
            │   └── ...
            └── ASVspoof2019.LA.cm.eval.trl.txt

PROTOCOL FILE FORMAT:
--------------------

    SPEAKER_ID AUDIO_ID SYSTEM_ID KEY
    arf_00295 LA_T_0000001 - bonafide
    arf_00295 LA_T_0000002 RIR_SMALL_NOI_SNR15 bonafide
    clf_00123 LA_T_0000003 - spoof
    clf_00123 LA_T_0000004 CODEC_8K_LOSS2PCT spoof

    Where:
    - SPEAKER_ID: Speaker identifier
    - AUDIO_ID: Unique file ID (LA_T_* for train, LA_D_* for dev, LA_E_* for eval)
    - SYSTEM_ID: "-" for original/clean, or augmentation label for augmented
    - KEY: "bonafide" or "spoof"

USAGE:
------

    # Uniform factor 3x for both classes
    python run_augmentation.py --min_factor 3x --seed 42

    # Uniform factor 5x
    python run_augmentation.py --min_factor 5x

EXAMPLE:
--------

Input (train split): 20,000 bonafide originals, 80,000 spoof originals.
Factor 3x.

  - bonafide: 20,000 clean + 40,000 augmented = 60,000 total
  - spoof:    80,000 clean + 160,000 augmented = 240,000 total

Clean fraction is 1/3 (33.3%) for BOTH classes by construction, so
clean-vs-augmented carries no information about the label. The natural
80,000:20,000 spoof:bonafide imbalance is preserved; rebalancing is left to
training time (class weights or a balanced sampler).
"""

import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import augmenters
from app.augmenter.rir_augmenter import RIRAugmenter
from app.augmenter.codec_augmenter import CodecAugmenter

# Import utilities
from app.dataset_loader import DatasetLoader
from app.config.augmentation_config import AugmentationConfigManager
from app.schema import AugmentationType
import app.utils.utils as utils


class AugmentationPipeline:
    """
    State-of-the-art augmentation pipeline for anti-spoofing.

    Features:
    - Uniform augmentation factor for both classes (no corpus-side rebalancing)
    - Speaker-independent splits
    - ASVspoof2019 LA format output
    """

    def __init__(
        self,
        voices_root: str = "data/marsa_speaker_disjoint_partition",
        musan_root: str = "data/noise_dataset/musan",
        rir_root: str = "data/noise_dataset/RIR",
        output_root: str = "data/augmented",
        min_factor: str = "3x",
        loudness_target_dbfs: float = -23.0,
        loudness_peak_ceiling: float = 0.99,
        seed: int = 42
    ):
        """
        Initialize augmentation pipeline.

        Args:
            voices_root: Path to speaker-independent partitioned dataset
            musan_root: Path to MUSAN noise dataset
            rir_root: Path to RIR files
            output_root: Output directory
            min_factor: Augmentation factor applied uniformly to both classes
                (e.g., "3x", "5x")
            loudness_target_dbfs: Target RMS level applied to every emitted clip
            loudness_peak_ceiling: Peak ceiling applied after loudness normalization
            seed: Random seed for reproducibility
        """
        self.voices_root = Path(voices_root)
        self.musan_root = Path(musan_root)
        self.rir_root = Path(rir_root)
        self.output_root = Path(output_root)
        self.min_factor = min_factor
        self.loudness_target_dbfs = loudness_target_dbfs
        self.loudness_peak_ceiling = loudness_peak_ceiling
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Parse factor
        self.factor_num = int(min_factor.replace('x', ''))

        self.output_dir = self.output_root / f"augmented_{min_factor}"

        # Initialize components
        self.loader = DatasetLoader(
            voices_root=str(self.voices_root),
            musan_root=str(self.musan_root),
            rir_root=str(self.rir_root)
        )

        # Setup logger
        self.logger = logging.getLogger("augmentation_pipeline")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        # File handler (saved to output directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / f"{min_factor}.txt"
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.log_path = log_path

        # Load augmentation config
        config_manager = AugmentationConfigManager.get_instance()
        self.strategy = config_manager.get_strategy(min_factor)

        # Initialize augmenters
        self.logger.info("\nInitializing augmenters...")

        self.rir_augmenter = RIRAugmenter(
            config=self.strategy.rir_noise_config,
            rir_root=str(self.rir_root),
            noise_root=str(self.musan_root)
        )

        self.codec_augmenter = CodecAugmenter(
            config=self.strategy.codec_config
        )

        # RawBoost is not instantiated here — it runs training-time on-the-fly.

        # Statistics
        self.stats = {
            'train': {'bonafide': 0, 'spoof': 0, 'total': 0, 'clean': 0, 'augmented': 0,
                      'bonafide_clean': 0, 'spoof_clean': 0},
            'dev': {'bonafide': 0, 'spoof': 0, 'total': 0, 'clean': 0, 'augmented': 0,
                    'bonafide_clean': 0, 'spoof_clean': 0},
            'eval': {'bonafide': 0, 'spoof': 0, 'total': 0, 'clean': 0, 'augmented': 0,
                     'bonafide_clean': 0, 'spoof_clean': 0}
        }

        # Audio ID counters
        self.audio_id_counter = {'train': 1, 'dev': 1, 'eval': 1}

        # Protocol entries
        self.protocol_entries = {'train': [], 'dev': [], 'eval': []}

    def _select_augmentation_mode(self) -> List[AugmentationType]:
        """Select augmentation mode for one clip.

        With probability ``stacking_probability`` the clip receives both
        RIR_NOISE and CODEC in sequence (stacked); otherwise exactly one
        type is drawn according to ``type_distribution``.

        Returns:
            A list of AugmentationType values to apply in order.
        """
        if random.random() < self.strategy.stacking_probability:
            return [AugmentationType.RIR_NOISE, AugmentationType.CODEC]

        types = list(self.strategy.type_distribution.keys())
        probabilities = [self.strategy.type_distribution[t] for t in types]
        chosen = random.choices(types, weights=probabilities, k=1)[0]
        return [chosen]

    def _apply_single_augmentation(
        self,
        audio: np.ndarray,
        sr: int,
        aug_type: AugmentationType,
    ) -> Tuple[np.ndarray, str]:
        """Apply one augmentation type to audio.

        Args:
            audio: Input waveform as a numpy array.
            sr: Sample rate in Hz.
            aug_type: The augmentation to apply.

        Returns:
            Tuple of (augmented waveform, system_id string).
        """
        if aug_type == AugmentationType.RIR_NOISE:
            augmented, metadata = self.rir_augmenter.augment(audio, sr, return_metadata=True)
            system_id = self.rir_augmenter.get_augmentation_label(
                metadata['room_size'], metadata['noise_source'], metadata['snr_db']
            )
        elif aug_type == AugmentationType.CODEC:
            augmented, metadata = self.codec_augmenter.augment(audio, sr, return_metadata=True)
            system_id = self.codec_augmenter.get_augmentation_label(
                metadata['codec_sr'], metadata['packet_loss'],
                metadata['bandpass'], metadata['quantization_bits'],
                metadata.get('codec')
            )
        else:
            augmented = audio
            system_id = "-"

        return augmented, system_id

    def _apply_augmentation(
        self,
        audio: np.ndarray,
        sr: int,
        aug_types: List[AugmentationType],
    ) -> Tuple[np.ndarray, str]:
        """Apply a sequence of augmentation types and join their labels.

        Each type in ``aug_types`` is applied in order so that stacked clips
        accumulate both transformations. System IDs are joined with ``|``.

        Args:
            audio: Input waveform as a numpy array.
            sr: Sample rate in Hz.
            aug_types: Ordered list of augmentation types to apply.

        Returns:
            Tuple of (augmented waveform, composite system_id string).
        """
        system_ids: List[str] = []
        current = audio

        for aug_type in aug_types:
            current, part_id = self._apply_single_augmentation(current, sr, aug_type)
            if part_id and part_id != "-":
                system_ids.append(part_id)

        composite_id = "|".join(system_ids) if system_ids else "-"
        return current, composite_id

    def _generate_audio_id(self, split: str) -> str:
        """Generate ASVspoof-style audio ID."""
        prefix_map = {'train': 'LA_T', 'dev': 'LA_D', 'eval': 'LA_E'}
        prefix = prefix_map[split]
        audio_id = f"{prefix}_{self.audio_id_counter[split]:07d}"
        self.audio_id_counter[split] += 1
        return audio_id

    def _save_audio_and_protocol(
        self, audio: np.ndarray, sr: int, split: str,
        speaker_id: str, system_id: str, key: str
    ):
        """Save audio as FLAC and add protocol entry."""
        audio_id = self._generate_audio_id(split)

        flac_dir = self.output_dir / "LA" / f"ASVspoof2019_LA_{split}" / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)

        audio_path = flac_dir / f"{audio_id}.flac"
        # Single, uniform loudness policy for every emitted clip (clean and all
        # augmentation types, every split) so loudness leaks nothing about class.
        audio = utils.normalize_loudness(
            audio, self.loudness_target_dbfs, self.loudness_peak_ceiling
        )
        utils.save_audio_flac(audio, str(audio_path), sr=sr)

        protocol_entry = f"{speaker_id} {audio_id} {system_id} {key}"
        self.protocol_entries[split].append(protocol_entry)

        self.stats[split][key] += 1
        self.stats[split]['total'] += 1

        if system_id == "-":
            self.stats[split]['clean'] += 1
            self.stats[split][f'{key}_clean'] += 1
        else:
            self.stats[split]['augmented'] += 1

    def _emit_class(
        self, files: List[Dict], split: str, key: str,
        clean_count: int, aug_count: int
    ):
        """Emit clean + augmented copies for one class.

        Args:
            files: List of file-info dicts for this class/split.
            split: One of "train"/"dev"/"eval".
            key: "bonafide" or "spoof".
            clean_count: Number of clean copies to emit. ``None`` means "one clean
                copy per original" (used for the unaugmented dev/eval splits).
            aug_count: Number of augmented copies to emit.
        """
        if not files:
            self.logger.warning(f"  No {key} files for {split}; skipping.")
            return

        if clean_count is None:
            clean_count = len(files)

        audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}

        def _load(idx: int) -> Tuple[str, np.ndarray, int]:
            info = files[idx % len(files)]
            path = info['filepath']
            if path not in audio_cache:
                audio_cache[path] = utils.load_audio(path)
            audio, sr = audio_cache[path]
            return info['speaker_id'], audio, sr

        for i in tqdm(range(clean_count), desc=f"  {split} {key} clean"):
            speaker_id, audio, sr = _load(i)
            self._save_audio_and_protocol(audio, sr, split, speaker_id, "-", key)

        for i in tqdm(range(aug_count), desc=f"  {split} {key} aug"):
            speaker_id, audio, sr = _load(i)
            aug_types = self._select_augmentation_mode()
            augmented, system_id = self._apply_augmentation(audio, sr, aug_types)
            self._save_audio_and_protocol(augmented, sr, split, speaker_id, system_id, key)

    def _process_split(self, split: str, counts: Dict[str, Tuple]):
        """Process a complete split from per-class (clean, aug) counts.

        Args:
            split: One of "train"/"dev"/"eval".
            counts: Mapping ``{'bonafide': (clean, aug), 'spoof': (clean, aug)}``.
                A ``clean`` value of ``None`` means "one clean copy per original".
        """
        self.logger.info(f"\nProcessing {split} split...")

        if split == 'train':
            bonafide_files = self.loader.load_bonafide_train_files()
            spoof_files = self.loader.load_spoof_train_files()
        elif split == 'dev':
            bonafide_files = self.loader.load_bonafide_dev_files()
            spoof_files = self.loader.load_spoof_dev_files()
        elif split == 'eval':
            bonafide_files = self.loader.load_bonafide_eval_files()
            spoof_files = self.loader.load_spoof_eval_files()

        b_clean, b_aug = counts['bonafide']
        s_clean, s_aug = counts['spoof']
        self.logger.info(
            f"  Bonafide: {len(bonafide_files)} originals -> "
            f"{b_clean if b_clean is not None else len(bonafide_files)} clean + {b_aug} aug"
        )
        self.logger.info(
            f"  Spoof:    {len(spoof_files)} originals -> "
            f"{s_clean if s_clean is not None else len(spoof_files)} clean + {s_aug} aug"
        )

        self._emit_class(bonafide_files, split, 'bonafide', b_clean, b_aug)
        self._emit_class(spoof_files, split, 'spoof', s_clean, s_aug)

    def _write_protocol_files(self):
        """Write protocol files for all splits."""
        self.logger.info("\nWriting protocol files...")

        protocol_map = {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'
        }

        for split, filename in protocol_map.items():
            if self.protocol_entries[split]:
                protocol_dir = self.output_dir / "LA" / f"ASVspoof2019_LA_{split}"
                protocol_path = protocol_dir / filename

                with open(protocol_path, 'w') as f:
                    for entry in self.protocol_entries[split]:
                        f.write(entry + '\n')

                self.logger.info(f"  Written: {filename} ({len(self.protocol_entries[split])} entries)")

    def _print_final_report(self):
        """Print and log the final report, including the per-class clean-fraction
        check that proves shortcut #1 is gone (bonafide and spoof clean % equal)."""
        self.logger.info("\n" + "="*70)
        self.logger.info("AUGMENTATION COMPLETE")
        self.logger.info("="*70)

        self.logger.info(f"\nConfiguration:")
        self.logger.info(f"  Min factor:   {self.min_factor}")
        self.logger.info(f"  Clean frac:   {1.0 / self.factor_num:.0%} (uniform, both classes)")
        self.logger.info(f"  Seed:         {self.seed}")
        self.logger.info(f"  Output:       {self.output_dir}")

        self.logger.info(f"\nFinal Dataset Statistics:")
        for split in ['train', 'dev', 'eval']:
            if self.stats[split]['total'] > 0:
                s = self.stats[split]
                bonafide_pct = (s['bonafide'] / s['total']) * 100
                spoof_pct = (s['spoof'] / s['total']) * 100
                clean_pct = (s['clean'] / s['total']) * 100

                # Per-class clean fraction: these MUST match (leak #1 check).
                b_clean_pct = (s['bonafide_clean'] / s['bonafide'] * 100) if s['bonafide'] else 0.0
                s_clean_pct = (s['spoof_clean'] / s['spoof'] * 100) if s['spoof'] else 0.0

                self.logger.info(f"\n  {split.upper()}:")
                self.logger.info(f"    Total:    {s['total']:,} files")
                self.logger.info(f"    Bonafide: {s['bonafide']:,} ({bonafide_pct:.1f}%)")
                self.logger.info(f"    Spoof:    {s['spoof']:,} ({spoof_pct:.1f}%)")
                self.logger.info(f"    Clean:    {s['clean']:,} ({clean_pct:.1f}%)")
                self.logger.info(f"    Augmented: {s['augmented']:,} ({100-clean_pct:.1f}%)")
                self.logger.info(
                    f"    Clean fraction per class (must match): "
                    f"bonafide {b_clean_pct:.1f}% vs spoof {s_clean_pct:.1f}%"
                )

        self.logger.info("="*70 + "\n")
        self.logger.info(f"Log saved to: {self.log_path}")

    def _preflight_check(self) -> Dict[str, Dict[str, int]]:
        """Validate the input partition before any audio is written.

        Runs in seconds and fails loudly rather than letting a multi-hour run
        produce a silently incomplete corpus. Three conditions are enforced
        for every split:

        1. The split is non-empty.
        2. No file present on disk was ignored during discovery, so the
           released corpus cannot be missing audio that exists in the input.
        3. One sample file of each distinct extension actually decodes, which
           catches a missing codec backend before any work is done instead of
           part-way through the run.

        The full discovery report for each split is written to the pipeline
        log so that any later discrepancy can be audited from the log alone.

        Returns:
            Per-split statistics mapping as returned by the loader.

        Raises:
            RuntimeError: If any split is empty, any file was ignored, or any
                sampled file fails to decode.
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PREFLIGHT CHECK")
        self.logger.info("="*70)

        statistics = self.loader.get_dataset_statistics()
        split_loaders = {
            'train': self.loader.load_train_files,
            'dev': self.loader.load_dev_files,
            'eval': self.loader.load_eval_files,
        }
        problems: List[str] = []

        for split, load_files in split_loaders.items():
            report = self.loader.reports[split]
            self.logger.info(
                f"\n  {split.upper()}: {report.audio_file_count:,} files from "
                f"{report.speaker_count:,} speakers ({report.structure})"
            )
            self.logger.info(
                f"    bonafide {report.bonafide_count:,} / "
                f"spoof {report.spoof_count:,}"
            )

            if report.audio_file_count == 0:
                problems.append(
                    f"{split}: no audio files discovered under "
                    f"{self.voices_root / split}"
                )
                continue

            if report.skipped_total:
                self.logger.info(
                    f"    IGNORED: {report.skipped_by_extension}"
                )
                problems.append(
                    f"{split}: {report.skipped_total} file(s) on disk were "
                    f"ignored by extension {report.skipped_by_extension}; "
                    f"they would be missing from the corpus"
                )

            if report.unknown_prefix_count:
                problems.append(
                    f"{split}: {report.unknown_prefix_count} file(s) match "
                    f"neither the bonafide nor the spoof naming convention"
                )

            probes: Dict[str, str] = {}
            for info in load_files():
                extension = Path(info['filepath']).suffix.lower()
                if extension not in probes:
                    probes[extension] = info['filepath']

            for extension, filepath in sorted(probes.items()):
                try:
                    audio, sample_rate = utils.load_audio(filepath)
                except Exception as error:
                    problems.append(
                        f"{split}: cannot decode '{extension}' audio "
                        f"({Path(filepath).name}): {error}"
                    )
                    continue

                if audio.size == 0:
                    problems.append(
                        f"{split}: decoded '{extension}' sample "
                        f"{Path(filepath).name} is empty"
                    )
                    continue

                self.logger.info(
                    f"    decode OK {extension}: {audio.size / sample_rate:.2f}s "
                    f"at {sample_rate} Hz"
                )

        if problems:
            self.logger.error("\nPREFLIGHT FAILED:")
            for problem in problems:
                self.logger.error(f"  - {problem}")
            raise RuntimeError(
                f"Preflight check failed with {len(problems)} problem(s); "
                f"see the log above. No audio was written."
            )

        self.logger.info("\nPreflight check passed.")
        return statistics

    def run(self):
        """Execute the complete augmentation pipeline.

        Raises:
            RuntimeError: If the preflight check rejects the input partition.
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("ANTI-SPOOFING AUGMENTATION PIPELINE")
        self.logger.info("="*70)

        self.logger.info(f"\nConfiguration:")
        self.logger.info(f"  Min factor:   {self.min_factor}")
        self.logger.info(f"  Clean frac:   {1.0 / self.factor_num:.0%} (uniform, both classes)")
        self.logger.info(f"  Loudness:     {self.loudness_target_dbfs} dBFS")
        self.logger.info(f"  Seed:         {self.seed}")

        # Validate the whole partition before writing anything.
        stats = self._preflight_check()

        n_bonafide = stats['train']['bonafide']
        n_spoof = stats['train']['spoof']

        self.logger.info(f"  Train bonafide: {n_bonafide:,}")
        self.logger.info(f"  Train spoof:    {n_spoof:,}")

        # Uniform factor for both classes: one clean copy per original plus
        # (factor - 1) augmented copies, natural class ratio preserved.
        f = self.factor_num
        train_counts = {
            'bonafide': (n_bonafide, n_bonafide * (f - 1)),
            'spoof': (n_spoof, n_spoof * (f - 1)),
        }
        self.logger.info(
            f"\nUniform factor {f}x for both classes "
            f"(clean fraction {1.0 / f:.0%}), natural ratio preserved."
        )

        # Process splits (dev/eval are clean only: one clean copy per original).
        self._process_split('train', train_counts)
        self._process_split('dev', {'bonafide': (None, 0), 'spoof': (None, 0)})
        self._process_split('eval', {'bonafide': (None, 0), 'spoof': (None, 0)})

        # Write protocols
        self._write_protocol_files()

        # Final report
        self._print_final_report()


if __name__ == "__main__":
    pipeline = AugmentationPipeline(min_factor="3x")
    print("Pipeline initialized successfully!")
