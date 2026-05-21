"""
Pre-flight manifest generator for the HABLA-Spoof corpus composition plan.

Given a transcribed bonafide corpus (sample_key -> entry with word count
and audio path), produces the dispatch CSV that assigns every eligible
file to exactly one attack system and one partition (not_jittered /
jittered), with pre-computed tier eligibility. The manifest is the
single source of truth for which file belongs to which pipeline run.
"""
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from app.pipeline.partial_spoof.schemas.manifest_entry import ManifestEntry
from app.pipeline.partial_spoof.schemas.manifest_summary import ManifestSummary
from app.pipeline.partial_spoof.utils.tier_eligibility import TierEligibilityComputer


class ManifestGenerator:
    """Build the partial spoof dispatch manifest from a transcribed corpus.

    Two-stage deterministic plan per speaker:
      Stage 1: partition the speaker's files 50/50 into not_jittered and
        jittered using BONAFIDE_PARTITION_SEED + sha256(speaker_id).
      Stage 2: assign each file in each partition to one attack via a
        multinomial draw weighted by attack_weights, seeded with
        ATTACK_ASSIGNMENT_SEED + sha256(speaker_id).

    Probabilistic per-speaker assignment preserves the corpus-wide
    weight marginal under all speaker file-count distributions. Small
    speakers may not see every attack (mathematical inevitability with
    a 40/20/10/10/10/10 distribution and few files), but the global
    fraction per attack converges to the target weights.

    Attributes:
        attack_weights: Mapping from attack name to its probability share.
        attack_assignment_seed: Base seed for per-file attack draws.
        bonafide_partition_seed: Base seed for the per-speaker 50/50 split.
        tier_computer: TierEligibilityComputer instance.
    """

    def __init__(
        self,
        attack_weights: Dict[str, float],
        attack_assignment_seed: int,
        bonafide_partition_seed: int,
        tier_computer: TierEligibilityComputer,
    ) -> None:
        """Initialise the manifest generator.

        Args:
            attack_weights: Dict mapping attack name to probability
                weight. Must sum to 1.0 within 1e-6 tolerance.
            attack_assignment_seed: Seed for the attack-assignment RNG.
            bonafide_partition_seed: Seed for the not_jittered/jittered
                partition RNG. Should match the value used by Step 1.
            tier_computer: Initialised TierEligibilityComputer.

        Raises:
            ValueError: If weights do not sum to ~1.0.
        """
        total = sum(attack_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"attack_weights must sum to 1.0, got {total}"
            )
        self.attack_weights = attack_weights
        self.attack_assignment_seed = attack_assignment_seed
        self.bonafide_partition_seed = bonafide_partition_seed
        self.tier_computer = tier_computer

    def generate(
        self,
        transcripts: Dict[str, Dict],
    ) -> Tuple[List[ManifestEntry], ManifestSummary]:
        """Produce the full manifest plus its aggregate summary.

        Iterates speakers (sorted for determinism), filters tier-ineligible
        files, applies the per-speaker not_jittered/jittered split, then
        assigns attacks per file via probabilistic draw. Builds one
        ManifestEntry per emitted file and one ManifestSummary at the
        end with corpus marginals, per-speaker coverage stats, and the
        realised attack distribution.

        Args:
            transcripts: Dict from sample_key (e.g. 'arf_00295_TEXT_00001')
                to entry dict with keys speaker_id, split, audio_path,
                transcript, word_count, word_timestamps.

        Returns:
            Tuple of (entries, summary). Entries are ordered by
            (speaker_id, partition, sample_key).
        """
        by_speaker = self._group_by_speaker(transcripts)
        attacks = list(self.attack_weights.keys())
        probs = np.array([self.attack_weights[a] for a in attacks], dtype=float)

        entries: List[ManifestEntry] = []
        per_attack_per_partition: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        speaker_attacks: Dict[str, set] = defaultdict(set)

        for speaker_id in sorted(by_speaker):
            speaker_entries = by_speaker[speaker_id]
            eligible = [
                e for e in speaker_entries
                if self.tier_computer.is_eligible(e["word_count"])
            ]
            if not eligible:
                logger.warning(
                    f"Speaker {speaker_id}: no tier-eligible files; skipped."
                )
                continue

            eligible.sort(key=lambda e: e["sample_key"])
            not_jittered, jittered = self._split_partition(
                eligible, speaker_id
            )

            for partition_label, partition_files in (
                ("not_jittered", not_jittered),
                ("jittered", jittered),
            ):
                if not partition_files:
                    continue
                assignment_rng = self._make_rng(
                    self.attack_assignment_seed, speaker_id
                )
                draws = assignment_rng.choice(
                    len(attacks),
                    size=len(partition_files),
                    p=probs,
                )
                for file_entry, attack_idx in zip(partition_files, draws):
                    attack = attacks[int(attack_idx)]
                    planned = self.tier_computer.compute(file_entry["word_count"])
                    manifest_entry = ManifestEntry(
                        speaker_id=speaker_id,
                        sample_key=file_entry["sample_key"],
                        audio_path=Path(file_entry["audio_path"]),
                        split=file_entry["split"],
                        partition=partition_label,
                        attack=attack,
                        planned_tiers=planned,
                        word_count=file_entry["word_count"],
                        bonafide_transcript=file_entry.get("transcript"),
                    )
                    entries.append(manifest_entry)
                    per_attack_per_partition[attack][partition_label] += 1
                    speaker_attacks[speaker_id].add(attack)

        summary = self._build_summary(
            entries=entries,
            per_attack_per_partition=per_attack_per_partition,
            speaker_attacks=speaker_attacks,
        )
        return entries, summary

    def _group_by_speaker(
        self,
        transcripts: Dict[str, Dict],
    ) -> Dict[str, List[Dict]]:
        """Re-key transcripts as speaker_id -> list of file entries.

        Args:
            transcripts: Original sample_key -> entry mapping.

        Returns:
            Dict mapping speaker_id to a list of entry dicts; each entry
            gets the sample_key folded back into it for downstream use.
        """
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for sample_key, entry in transcripts.items():
            speaker_id = entry["speaker_id"]
            file_entry = dict(entry)
            file_entry["sample_key"] = sample_key
            grouped[speaker_id].append(file_entry)
        return grouped

    def _split_partition(
        self,
        eligible: List[Dict],
        speaker_id: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Apply the deterministic per-speaker 50/50 partition.

        Same algorithm as step_01 _apply_partition so the manifest
        matches Step 1's filter exactly.

        Args:
            eligible: Speaker's tier-eligible files, sorted by sample_key.
            speaker_id: Used to seed the shuffle.

        Returns:
            Tuple of (not_jittered_files, jittered_files).
        """
        rng = self._make_rng(self.bonafide_partition_seed, speaker_id)
        shuffled_indices = rng.permutation(len(eligible))
        half = len(shuffled_indices) // 2
        not_jittered_indices = sorted(shuffled_indices[:half].tolist())
        jittered_indices = sorted(shuffled_indices[half:].tolist())
        not_jittered = [eligible[i] for i in not_jittered_indices]
        jittered = [eligible[i] for i in jittered_indices]
        return not_jittered, jittered

    def _make_rng(
        self,
        base_seed: int,
        speaker_id: str,
    ) -> np.random.RandomState:
        """Build a per-speaker RNG from a base seed and a stable speaker hash.

        Uses the same sha256(speaker_id)[:4] little-endian formula as
        step_01 _apply_partition so both stages key off the same speaker
        identity.

        Args:
            base_seed: Stage-specific base seed.
            speaker_id: HABLA speaker identifier.

        Returns:
            Seeded np.random.RandomState ready for use.
        """
        speaker_hash = int.from_bytes(
            hashlib.sha256(speaker_id.encode("utf-8")).digest()[:4],
            byteorder="little",
        )
        return np.random.RandomState(base_seed + speaker_hash)

    def _build_summary(
        self,
        entries: List[ManifestEntry],
        per_attack_per_partition: Dict[str, Dict[str, int]],
        speaker_attacks: Dict[str, set],
    ) -> ManifestSummary:
        """Aggregate the manifest into a ManifestSummary for auditing.

        Args:
            entries: All ManifestEntry rows emitted.
            per_attack_per_partition: Running cross-tabulation.
            speaker_attacks: speaker_id -> set of attacks they hit.

        Returns:
            Fully-populated ManifestSummary.
        """
        total_entries = len(entries)
        per_attack_count: Dict[str, int] = defaultdict(int)
        per_partition_count: Dict[str, int] = defaultdict(int)
        per_tier_count: Dict[str, int] = defaultdict(int)

        for entry in entries:
            per_attack_count[entry.attack] += 1
            per_partition_count[entry.partition] += 1
            for tier in entry.planned_tiers:
                per_tier_count[tier] += 1

        speakers_total = len(speaker_attacks)
        speakers_with_all_attacks = sum(
            1 for hit in speaker_attacks.values()
            if len(hit) == len(self.attack_weights)
        )
        speakers_with_single_attack = sum(
            1 for hit in speaker_attacks.values() if len(hit) == 1
        )

        actual_fraction: Dict[str, float] = {}
        if total_entries > 0:
            for attack in self.attack_weights:
                actual_fraction[attack] = (
                    per_attack_count.get(attack, 0) / total_entries
                )

        nested = {
            attack: dict(parts)
            for attack, parts in per_attack_per_partition.items()
        }

        return ManifestSummary(
            total_entries=total_entries,
            per_attack_count=dict(per_attack_count),
            per_partition_count=dict(per_partition_count),
            per_attack_per_partition_count=nested,
            per_tier_potential_count=dict(per_tier_count),
            speakers_total=speakers_total,
            speakers_with_all_attacks=speakers_with_all_attacks,
            speakers_with_single_attack=speakers_with_single_attack,
            attack_weights_target=dict(self.attack_weights),
            attack_weights_actual=actual_fraction,
        )
