"""Verify speaker disjointness across the train, validation and test partitions.

The ICAI paper claims that HABLA speakers are assigned exclusively to a single
partition and are never shared across splits. That claim is the paper's defence
against the overfitting concern raised in review, so it must be demonstrated
rather than asserted. This script checks it directly against the file lists that
were actually used for training.

Speaker identity is recovered in two ways:

* Spanish (HABLA) utterance identifiers embed the speaker, e.g. ``vem_03397``
  decodes as Venezuela / male / speaker 03397. Voice-conversion utterances name
  both the source and the target speaker, and both are counted as involved,
  which is the strict criterion.
* English (ASVspoof 2019 LA) identifiers do not embed the speaker, so it is
  looked up in ``protocol.txt``, whose first column is the speaker label.

The script also reports whether the same utterance identifier appears verbatim
in two partitions, which would indicate a corpus construction error rather than
a design decision.

Run on ml-server03:

    python3 scripts/ICAI/habla_v1/check_speaker_leakage.py

An alternative corpus root may be given as the first argument.

Exit status is 0 when no leakage is found and 1 otherwise, so the script can be
used as a gate in a larger workflow.
"""

import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Set, Tuple

DEFAULT_ROOT = Path(
    "/home/jahurtado905/notebooks/anti-spoofing/anti-spoof-eval/03-asvspoof-mega"
)

PARTITIONS: Dict[str, str] = {
    "ENGLISH": "DATA_EN",
    "SPANISH": "DATA_LAT",
    "COMBINED": "DATA_UN",
}

SPLITS: Tuple[str, ...] = ("train", "val", "test")

SPEAKER_TOKEN = re.compile(r"(?:ar|cl|co|pe|ve)[fm]_\d+")


def load_english_speaker_map(data_dir: Path) -> Dict[str, str]:
    """Map ASVspoof utterance identifiers to their speaker label.

    Args:
        data_dir: Partition directory containing ``asvspoof2019_LA/protocol.txt``.

    Returns:
        Mapping from utterance identifier to speaker label. Empty when the
        protocol file is absent.
    """
    protocol = data_dir / "asvspoof2019_LA" / "protocol.txt"
    if not protocol.exists():
        return {}

    mapping: Dict[str, str] = {}
    for line in protocol.read_text(errors="ignore").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            mapping[fields[1]] = fields[0]
    return mapping


def speakers_of(utterance_id: str, english_map: Dict[str, str]) -> Set[str]:
    """Return every speaker involved in one utterance.

    Args:
        utterance_id: Identifier as it appears in the ``.lst`` file.
        english_map: Utterance to speaker mapping for the English corpus.

    Returns:
        Set of speaker labels. For voice-conversion items this contains both the
        source and the target speaker. Empty when the speaker cannot be resolved.
    """
    if utterance_id.startswith("LA_"):
        speaker = english_map.get(utterance_id)
        return {speaker} if speaker else set()
    return set(SPEAKER_TOKEN.findall(utterance_id))


def read_list(path: Path) -> list:
    """Read a file list, dropping blank lines."""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def audit_partition(label: str, data_dir: Path) -> bool:
    """Audit one corpus partition and print its report.

    Args:
        label: Human readable partition name used in the report.
        data_dir: Directory holding ``asvspoof2019_LA/scp/*.lst``.

    Returns:
        True when no speaker or utterance overlap was detected.
    """
    scp_dir = data_dir / "asvspoof2019_LA" / "scp"
    print("=" * 70)
    print(f"{label}  ({data_dir.name})")

    if not scp_dir.exists():
        print(f"  directory not found: {scp_dir}")
        return True

    english_map = load_english_speaker_map(data_dir)

    utterances: Dict[str, list] = {}
    speakers: Dict[str, Set[str]] = {}
    for split in SPLITS:
        items = read_list(scp_dir / f"{split}.lst")
        if not items:
            continue
        utterances[split] = items
        found: Set[str] = set()
        unresolved = 0
        for item in items:
            resolved = speakers_of(item, english_map)
            if resolved:
                found |= resolved
            else:
                unresolved += 1
        speakers[split] = found
        note = f"   [{unresolved} utterances with no resolvable speaker]" if unresolved else ""
        print(f"  {split:6s} {len(items):7d} utterances   {len(found):5d} speakers{note}")

    clean = True

    print("  speaker overlap")
    for first, second in combinations(speakers, 2):
        shared = speakers[first] & speakers[second]
        if shared:
            clean = False
            sample = ", ".join(sorted(shared)[:8])
            print(f"    LEAK  {first} & {second}: {len(shared)} shared -> {sample}")
        else:
            print(f"    ok    {first} & {second}: none shared")

    print("  identical utterances")
    for first, second in combinations(utterances, 2):
        duplicated = set(utterances[first]) & set(utterances[second])
        if duplicated:
            clean = False
            print(f"    LEAK  {first} & {second}: {len(duplicated)} identical identifiers")
        else:
            print(f"    ok    {first} & {second}: none identical")

    print(f"  => {'NO LEAKAGE' if clean else 'REVIEW REQUIRED'}\n")
    return clean


def main() -> int:
    """Audit every partition and return a process exit status."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    print(f"corpus root: {root}\n")

    all_clean = True
    for label, folder in PARTITIONS.items():
        all_clean &= audit_partition(label, root / folder)

    print("=" * 70)
    print("RESULT:", "no leakage detected" if all_clean else "leakage detected, see above")
    return 0 if all_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
