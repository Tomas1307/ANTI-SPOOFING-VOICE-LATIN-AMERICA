"""
Build the strict (sentence-disjoint) dev/eval filter list for MARSA.

The canonical partition is speaker-disjoint but not sentence-disjoint:
66.4 percent of eval source utterances share their sentence with a training
utterance (Usage Notes). This script emits, for every dev/eval clip in the
speaker-disjoint partition, whether its underlying SOURCE SENTENCE is unseen
in training ("strict"). The output joins onto every augmented tier through
the metadata CSV's source_file column (which stores the partition basename).

Sentence resolution per clip class:
  - bonafide_<spk>_<n>: symlink target basename is the original bonafide
    file, looked up in the partial-spoof manifest (audio_path column) for
    its Parakeet transcript.
  - spoof_<system>-ps<tier>_<spk>_<n> (partial spoof): the spliced filename
    embeds the SOURCE bonafide utterance id (<SYSTEM>_PSW<k>_<spk>_<origid>),
    resolved through the manifest like bonafide. The host utterance, not the
    spliced transcript, defines the sentence identity.
  - spoof_<system>_<spk>_<n> (full spoof): the symlink target is an LA flac
    (LA_T_XXXXXXX.flac). No system writes an explicit LA-id-to-sample map,
    so the mapping is RECONSTRUCTED: protocol line order is zipped against
    validated_samples.json entries ordered by (speaker_id, text_id), after
    verifying per-speaker counts agree, and the reconstruction is then
    VERIFIED acoustically by comparing FLAC durations against the recorded
    per-sample durations on a random sample. If verification fails for a
    system, its clips are marked unresolved (conservatively non-strict)
    rather than silently mislabeled.

The training sentence set is the union of (a) manifest transcripts of all
train-speaker bonafide utterances and (b) every attack system's
generation_metadata.json target texts for train speakers, so a sentence
seen in ANY training clip, genuine or synthetic, disqualifies an eval clip
from the strict subset.

Usage on ml-server03 (fishgram_env has soundfile/pydantic/loguru):
    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
    python -m app.scripts.build_strict_eval_filter
    deactivate
"""
import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
from loguru import logger

from app.schemas.strict_eval_filter_report import StrictEvalFilterReport

DEFAULT_PARTITION_DIR = Path("data/marsa_speaker_disjoint_partition")
DEFAULT_MANIFEST_CSV = Path("data/manifests/partial_spoof_plan.csv")
DEFAULT_OUTPUT_CSV = Path("data/marsa_speaker_disjoint_partition/strict_eval_filter.csv")

FULLSPOOF_SYSTEMS: List[str] = [
    "fishgram", "qwen", "openvoice", "chatterbox", "outetts", "omnivoice",
]
FULLSPOOF_OUTPUT_DIRS: Dict[str, Path] = {
    system: Path(f"data/{system}_output") for system in FULLSPOOF_SYSTEMS
}

DURATION_CHECK_SAMPLES = 40
DURATION_TOLERANCE_S = 0.06
DURATION_MAX_FAILURES = 2


class StrictEvalFilterBuilder:
    """Builds the sentence-disjoint strict filter for dev/eval clips.

    Attributes:
        partition_dir: Root of the speaker-disjoint symlink partition.
        manifest_csv: Partial-spoof manifest with per-utterance transcripts.
        fullspoof_dirs: Mapping of system name to its output directory.
        output_csv: Where the per-clip filter table is written.
        seed: RNG seed for the duration-verification sampling.
    """

    def __init__(
        self,
        partition_dir: Optional[Path] = None,
        manifest_csv: Optional[Path] = None,
        fullspoof_dirs: Optional[Dict[str, Path]] = None,
        output_csv: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the builder.

        Args:
            partition_dir: Override for the partition root.
            manifest_csv: Override for the manifest CSV path.
            fullspoof_dirs: Override mapping of system name to output dir.
            output_csv: Override for the output table path.
            seed: Seed for duration-verification sampling.
        """
        self.partition_dir = partition_dir or DEFAULT_PARTITION_DIR
        self.manifest_csv = manifest_csv or DEFAULT_MANIFEST_CSV
        self.fullspoof_dirs = fullspoof_dirs or FULLSPOOF_OUTPUT_DIRS
        self.output_csv = output_csv or DEFAULT_OUTPUT_CSV
        self.seed = seed

    @staticmethod
    def _norm(text: str) -> str:
        """Normalize a sentence for identity comparison.

        Lowercases, strips accents (via NFKD, since combining marks fall
        outside the word character class), removes punctuation, and
        collapses whitespace. Identical to the normalization used for the
        published overlap measurements.

        Args:
            text: Raw sentence.

        Returns:
            Normalized sentence string.
        """
        text = unicodedata.normalize("NFKD", text.lower())
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()

    def run(self) -> StrictEvalFilterReport:
        """Build the filter table and return a summary report.

        Returns:
            StrictEvalFilterReport with per-split counts.
        """
        spk2split = self._speaker_split_map()
        train_speakers = {s for s, sp in spk2split.items() if sp == "train"}

        logger.info("Building training sentence reference set...")
        manifest_by_basename, train_sentences = self._load_manifest(train_speakers)
        for system, out_dir in self.fullspoof_dirs.items():
            added = self._add_generation_texts(out_dir, train_speakers, train_sentences)
            logger.info(f"  {system}: +{added} train target texts.")
        logger.info(f"  {len(train_sentences):,} distinct normalized train sentences.")

        logger.info("Reconstructing and verifying full-spoof LA-id maps...")
        la_maps: Dict[str, Dict[str, str]] = {}
        verification: Dict[str, str] = {}
        for system, out_dir in self.fullspoof_dirs.items():
            la_map, verdict = self._map_system_la_ids(system, out_dir)
            verification[system] = verdict
            if la_map is not None:
                la_maps[system] = la_map
            logger.info(f"  {system}: {verdict}")

        logger.info("Classifying dev/eval clips...")
        rows: List[Tuple[str, str, str, str, str, int]] = []
        totals: Dict[str, int] = {}
        stricts: Dict[str, int] = {}
        unresolved: Dict[str, int] = {}

        for split in ("dev", "eval"):
            split_dir = self.partition_dir / split
            totals[split] = stricts[split] = unresolved[split] = 0
            for speaker_dir in sorted(split_dir.iterdir()):
                if not speaker_dir.is_dir():
                    continue
                for link in sorted(speaker_dir.iterdir()):
                    sentence = self._sentence_for_clip(
                        link, manifest_by_basename, la_maps
                    )
                    totals[split] += 1
                    if sentence is None:
                        unresolved[split] += 1
                        strict = 0
                    else:
                        strict = int(sentence not in train_sentences)
                        stricts[split] += strict
                    attack_id = self._attack_from_name(link.name)
                    rows.append(
                        (link.name, split, speaker_dir.name, attack_id,
                         "unresolved" if sentence is None else "resolved",
                         strict)
                    )

        # Per-class breakdown (bonafide / full-spoof / partial-spoof) so the
        # composition of the strict subset is interpretable at a glance.
        breakdown: Dict[Tuple[str, str], List[int]] = {}
        for _name, split, _spk, attack_id, resolution, strict in rows:
            if attack_id == "-":
                group = "bonafide"
            elif "-ps" in attack_id:
                group = "partial"
            else:
                group = "full"
            cell = breakdown.setdefault((split, group), [0, 0, 0])
            cell[0] += 1
            cell[1] += strict
            cell[2] += int(resolution == "unresolved")
        for (split, group), (total, strict, unres) in sorted(breakdown.items()):
            resolved = total - unres
            pct = strict / resolved * 100 if resolved else 0.0
            logger.info(
                f"  {split:5s} {group:9s} total={total:6,}  "
                f"strict={strict:6,} ({pct:.1f}% of resolved)  "
                f"unresolved={unres:,}"
            )

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["source_file", "split", "speaker_id", "attack_id",
                 "resolution", "strict"]
            )
            writer.writerows(rows)

        report = StrictEvalFilterReport(
            train_sentence_count=len(train_sentences),
            per_split_total=totals,
            per_split_strict=stricts,
            per_split_unresolved=unresolved,
            mapping_verification=verification,
            notes=[
                "strict=1 means the clip's source sentence never appears in "
                "any training clip (bonafide or synthetic).",
                "Unresolved clips are conservatively marked strict=0.",
                "Join onto augmented tiers via metadata CSV source_file.",
            ],
        )
        self._log_summary(report)
        return report

    def _speaker_split_map(self) -> Dict[str, str]:
        """Map every partition speaker to its split.

        Returns:
            Mapping speaker_id -> split name.
        """
        return {
            d.name: split
            for split in ("train", "dev", "eval")
            for d in (self.partition_dir / split).iterdir()
            if d.is_dir()
        }

    def _load_manifest(
        self, train_speakers: set
    ) -> Tuple[Dict[str, str], set]:
        """Load the manifest into a basename lookup and train sentence set.

        Args:
            train_speakers: Speaker IDs assigned to train.

        Returns:
            Tuple of (bonafide basename -> normalized sentence,
            set of normalized train sentences).
        """
        by_basename: Dict[str, str] = {}
        train_sentences: set = set()
        with open(self.manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sentence = self._norm(row["bonafide_transcript"] or "")
                if not sentence:
                    continue
                basename = Path(row["audio_path"]).stem
                by_basename[basename] = sentence
                if row["speaker_id"] in train_speakers:
                    train_sentences.add(sentence)
        return by_basename, train_sentences

    def _add_generation_texts(
        self, output_dir: Path, train_speakers: set, train_sentences: set
    ) -> int:
        """Add one system's train-speaker target texts to the train set.

        Args:
            output_dir: The system's output directory.
            train_speakers: Speaker IDs assigned to train.
            train_sentences: Mutable set updated in place.

        Returns:
            Number of sentences added.
        """
        metadata_path = output_dir / "generation_metadata.json"
        if not metadata_path.exists():
            logger.warning(f"No generation_metadata.json under {output_dir}.")
            return 0
        added = 0
        data = json.load(open(metadata_path, encoding="utf-8"))
        for entry in data.values():
            if entry.get("speaker_id") in train_speakers:
                sentence = self._norm(entry.get("text", ""))
                if sentence and sentence not in train_sentences:
                    train_sentences.add(sentence)
                    added += 1
        return added

    def _map_system_la_ids(
        self, system: str, output_dir: Path
    ) -> Tuple[Optional[Dict[str, str]], str]:
        """Reconstruct one system's LA-id to sentence mapping and verify it.

        Protocol lines (in file order) are zipped against the system's
        validated samples ordered by (speaker_id, text_id). The zip is only
        trusted if per-speaker counts agree exactly AND a random sample of
        FLAC durations matches the recorded per-sample durations.

        Args:
            system: System name.
            output_dir: The system's output directory.

        Returns:
            Tuple of (LA audio_id -> normalized sentence, verdict string);
            the mapping is None when verification fails.
        """
        validated_path = output_dir / "validated_samples.json"
        la_dir = output_dir / "LA"
        if not validated_path.exists() or not la_dir.exists():
            return None, "missing validated_samples.json or LA dir"

        validated = json.load(open(validated_path, encoding="utf-8"))

        # Some pipelines wrote everything into one LA split dir, others into
        # three (train/dev/eval, with the sample's split recorded per entry).
        # Reconstruction is therefore performed PER LA SPLIT DIR, matching
        # protocol lines against the validated entries belonging to that
        # split; within each split several candidate orderings are tried and
        # accepted only after speaker-sequence AND duration verification.
        insertion_all = list(validated.values())
        split_dirs = sorted(la_dir.glob("ASVspoof2019_LA_*"))
        mapping: Dict[str, str] = {}
        labels_used: List[str] = []

        for split_dir in split_dirs:
            split_token = split_dir.name.rsplit("_", 1)[-1]
            flac_dir = split_dir / "flac"
            protocol_entries: List[Tuple[str, str, Path]] = []
            for protocol in sorted(split_dir.glob("ASVspoof2019.LA.cm.*.txt")):
                for line in open(protocol, encoding="utf-8"):
                    parts = line.split()
                    if len(parts) >= 2:
                        protocol_entries.append((parts[0], parts[1], flac_dir))
            if not protocol_entries:
                continue

            if len(split_dirs) > 1:
                insertion = [
                    e for e in insertion_all if e.get("split") == split_token
                ]
            else:
                insertion = insertion_all

            if len(protocol_entries) != len(insertion):
                return None, (
                    f"{split_dir.name}: {len(protocol_entries)} protocol "
                    f"lines vs {len(insertion)} validated samples"
                )

            # Per-speaker matching: protocol lines are consumed in order
            # within each speaker's own sequence and zipped against that
            # speaker's validated entries, so the GLOBAL speaker order of the
            # protocol is irrelevant. Only the within-speaker order is
            # assumed (two candidates), and every candidate must still pass
            # the acoustic duration verification.
            per_speaker_entries: Dict[str, List[dict]] = {}
            for entry in insertion:
                per_speaker_entries.setdefault(entry["speaker_id"], []).append(entry)

            per_speaker_protocol: Dict[str, List[Tuple[str, str, Path]]] = {}
            for item in protocol_entries:
                per_speaker_protocol.setdefault(item[0], []).append(item)

            if set(per_speaker_protocol) != set(per_speaker_entries) or any(
                len(per_speaker_protocol[s]) != len(per_speaker_entries[s])
                for s in per_speaker_protocol
            ):
                return None, f"{split_dir.name}: per-speaker count mismatch"

            matched = False
            last_reason = "no candidate ordering matched"
            for label, keyfn in [
                ("text_id order", lambda e: e["text_id"]),
                ("within-speaker insertion", None),
            ]:
                pairs = []
                for speaker_id, proto_items in per_speaker_protocol.items():
                    entries = per_speaker_entries[speaker_id]
                    if keyfn is not None:
                        entries = sorted(entries, key=keyfn)
                    pairs.extend(zip(proto_items, entries))
                verdict = self._verify_durations(pairs)
                if verdict is not None:
                    last_reason = f"{split_dir.name}/{label}: {verdict}"
                    continue
                for (_spk, audio_id, _fd), entry in pairs:
                    mapping[audio_id] = self._norm(entry["text"])
                labels_used.append(f"{split_token}:{label}")
                matched = True
                break

            if not matched:
                # Order-free fallback: within each speaker, pair protocol
                # flacs against entries by DURATION (measured vs recorded).
                # No ordering assumption remains; ambiguous or unmatched
                # clips are simply left out of the mapping (they surface as
                # unresolved downstream instead of being mislabeled).
                added, dropped = self._match_by_duration(
                    per_speaker_protocol, per_speaker_entries, mapping
                )
                total = added + dropped
                if total and added / total >= 0.95:
                    labels_used.append(
                        f"{split_token}:duration matching "
                        f"({added:,} matched, {dropped:,} dropped)"
                    )
                    matched = True
            if not matched:
                return None, last_reason

        if not mapping:
            return None, "no protocol entries found"
        return mapping, f"OK via {'; '.join(labels_used)} ({len(mapping):,} ids)"

    def _match_by_duration(
        self,
        per_speaker_protocol: Dict[str, List[Tuple[str, str, Path]]],
        per_speaker_entries: Dict[str, List[dict]],
        mapping: Dict[str, str],
    ) -> Tuple[int, int]:
        """Pair protocol flacs with entries by duration, per speaker.

        Both sides are sorted by duration (measured for flacs, recorded for
        entries) and paired positionally; a pair is accepted only when the
        two durations agree within DURATION_TOLERANCE_S. When two entries of
        one speaker have near-identical durations AND different sentences the
        positional pairing could swap them, so such ambiguous pairs are also
        dropped rather than risked.

        Args:
            per_speaker_protocol: Speaker to ordered protocol items.
            per_speaker_entries: Speaker to validated entries.
            mapping: Output mapping updated in place (audio_id to sentence).

        Returns:
            Tuple of (accepted pair count, dropped pair count).
        """
        added = 0
        dropped = 0
        for speaker_id, proto_items in per_speaker_protocol.items():
            measured: List[Tuple[float, str]] = []
            for _spk, audio_id, flac_dir in proto_items:
                flac_path = flac_dir / f"{audio_id}.flac"
                if not flac_path.exists():
                    dropped += 1
                    continue
                measured.append((sf.info(str(flac_path)).duration, audio_id))
            entries = [
                (float(e["duration_seconds"]), self._norm(e["text"]))
                for e in per_speaker_entries[speaker_id]
                if e.get("duration_seconds") is not None
            ]
            measured.sort()
            entries.sort()

            for i, ((m_dur, audio_id), (r_dur, sentence)) in enumerate(
                zip(measured, entries)
            ):
                if abs(m_dur - r_dur) > DURATION_TOLERANCE_S:
                    dropped += 1
                    continue
                ambiguous = False
                for j in (i - 1, i + 1):
                    if 0 <= j < len(entries):
                        n_dur, n_sentence = entries[j]
                        if (abs(n_dur - r_dur) <= DURATION_TOLERANCE_S
                                and n_sentence != sentence):
                            ambiguous = True
                if ambiguous:
                    dropped += 1
                    continue
                mapping[audio_id] = sentence
                added += 1
        return added, dropped

    def _verify_durations(
        self, pairs: List[Tuple[Tuple[str, str, Path], dict]]
    ) -> Optional[str]:
        """Verify a candidate mapping acoustically on a random sample.

        Args:
            pairs: Zipped (protocol entry, validated sample) pairs.

        Returns:
            None when verification passes, otherwise a failure description.
        """
        rng = random.Random(self.seed)
        check = rng.sample(pairs, min(DURATION_CHECK_SAMPLES, len(pairs)))
        failures = 0
        checked = 0
        for (_speaker_id, audio_id, flac_dir), entry in check:
            recorded = entry.get("duration_seconds")
            flac_path = flac_dir / f"{audio_id}.flac"
            if recorded is None or not flac_path.exists():
                continue
            checked += 1
            actual = sf.info(str(flac_path)).duration
            if abs(actual - float(recorded)) > DURATION_TOLERANCE_S:
                failures += 1
        if checked == 0:
            return "duration verification impossible (no durations recorded)"
        if failures > DURATION_MAX_FAILURES:
            return (
                f"duration check failed ({failures}/{checked} beyond "
                f"{DURATION_TOLERANCE_S}s)"
            )
        return None

    def _sentence_for_clip(
        self,
        link: Path,
        manifest_by_basename: Dict[str, str],
        la_maps: Dict[str, Dict[str, str]],
    ) -> Optional[str]:
        """Resolve the normalized source sentence of one partition clip.

        Args:
            link: Partition symlink path.
            manifest_by_basename: Bonafide basename -> normalized sentence.
            la_maps: Per-system LA audio_id -> normalized sentence.

        Returns:
            The normalized sentence, or None when unresolvable.
        """
        name = link.name
        target = link.resolve() if link.is_symlink() else link

        if name.startswith("bonafide_"):
            return manifest_by_basename.get(target.stem)

        if name.startswith("spoof_"):
            token = name.split("_")[1]
            if "-ps" in token:
                # Spliced filenames embed the source bonafide basename after
                # SYSTEM_PSW<k>[J]_<speaker>_. HABLA sources are named
                # <speaker>_<digits> (speaker prefix repeated), Common Voice
                # sources are named common_voice_es_<digits> (no speaker
                # prefix), so the manifest key differs per provenance.
                match = re.match(
                    r"^[A-Za-z0-9]+_PSW\d+J?_([a-z]{3}_\d{5})_(.+)$",
                    target.stem,
                )
                if match:
                    speaker, rest = match.group(1), match.group(2)
                    key = rest if rest.startswith("common_voice") \
                        else f"{speaker}_{rest}"
                    return manifest_by_basename.get(key)
                return None
            la_map = la_maps.get(token)
            if la_map is not None:
                return la_map.get(target.stem)
        return None

    @staticmethod
    def _attack_from_name(name: str) -> str:
        """Extract the attack token from a partition filename.

        Args:
            name: Partition basename.

        Returns:
            Attack slug, or '-' for bonafide.
        """
        if name.startswith("bonafide_"):
            return "-"
        parts = name.split("_")
        return parts[1] if len(parts) >= 2 else "unknown"

    def _log_summary(self, report: StrictEvalFilterReport) -> None:
        """Log a human-readable summary.

        Args:
            report: The computed report.
        """
        logger.info("=" * 70)
        logger.info("STRICT EVAL FILTER COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Train sentence set: {report.train_sentence_count:,}")
        for split in ("dev", "eval"):
            total = report.per_split_total.get(split, 0)
            strict = report.per_split_strict.get(split, 0)
            unres = report.per_split_unresolved.get(split, 0)
            pct = strict / total * 100 if total else 0.0
            logger.info(
                f"  {split:5s} total={total:,}  strict={strict:,} ({pct:.1f}%)  "
                f"unresolved={unres:,}"
            )
        for system, verdict in report.mapping_verification.items():
            logger.info(f"  LA map {system}: {verdict}")
        logger.info(f"Output: {self.output_csv}")
        logger.info("=" * 70)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build the sentence-disjoint strict dev/eval filter list."
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override the output CSV path.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for duration-verification sampling (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    StrictEvalFilterBuilder(output_csv=args.output, seed=args.seed).run()
