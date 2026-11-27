"""Determine whether the HABLA speaker overlap is real leakage or a naming artefact.

The companion script ``check_speaker_leakage.py`` counts every speaker named in
an utterance identifier, including both endpoints of a voice conversion. That
strict criterion cannot distinguish two very different situations:

1. The partitions genuinely share speakers, which would mean the reported
   Spanish error rates are optimistic.
2. The partitions were defined on the target speaker of each conversion, and the
   apparent overlap comes only from source speakers being reused.

Bonafide utterances settle the question because they name exactly one speaker
and carry no source or target ambiguity. If bonafide speakers overlap between
train and test, the leakage is real and no alternative reading exists.

This script reports, per partition:

* speaker overlap restricted to bonafide utterances, which is the decisive test;
* speaker overlap for conversion sources and conversion targets separately;
* how much of the test material involves a speaker also seen in training.

Run on ml-server03:

    python3 scripts/ICAI/habla_v1/analyze_leakage_detail.py
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DEFAULT_ROOT = Path(
    "/home/jahurtado905/notebooks/anti-spoofing/anti-spoof-eval/03-asvspoof-mega"
)

PARTITIONS: Dict[str, str] = {"SPANISH": "DATA_LAT", "COMBINED": "DATA_UN"}
SPLITS: Tuple[str, ...] = ("train", "val", "test")
ATTACK_PREFIXES: Tuple[str, ...] = ("CycleGAN", "Diff", "StarGAN", "TTS")

SPEAKER_TOKEN = re.compile(r"(?:ar|cl|co|pe|ve)[fm]_\d+")


def classify(utterance_id: str) -> str:
    """Return the attack family of an utterance, or ``bonafide``."""
    for prefix in ATTACK_PREFIXES:
        if utterance_id.startswith(prefix + "-"):
            return prefix
    return "bonafide"


def endpoints(utterance_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return the first and second speaker named in an identifier.

    For bonafide utterances only the first is present. For conversions the two
    correspond to the endpoints of the transformation; their order follows the
    corpus naming convention and is reported without assuming which is which.
    """
    found = SPEAKER_TOKEN.findall(utterance_id)
    first = found[0] if found else None
    second = found[1] if len(found) > 1 else None
    return first, second


def read_list(path: Path) -> List[str]:
    """Read a file list, dropping blank lines."""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def report(label: str, data_dir: Path) -> None:
    """Print the detailed leakage breakdown for one partition."""
    scp_dir = data_dir / "asvspoof2019_LA" / "scp"
    print("=" * 72)
    print(f"{label}  ({data_dir.name})")

    if not scp_dir.exists():
        print(f"  directory not found: {scp_dir}\n")
        return

    bonafide: Dict[str, Set[str]] = {}
    first_side: Dict[str, Set[str]] = {}
    second_side: Dict[str, Set[str]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    spanish_items: Dict[str, List[str]] = {}

    for split in SPLITS:
        items = [u for u in read_list(scp_dir / f"{split}.lst") if not u.startswith("LA_")]
        if not items:
            continue
        spanish_items[split] = items
        bonafide[split] = set()
        first_side[split] = set()
        second_side[split] = set()
        counts[split] = {}
        for item in items:
            family = classify(item)
            counts[split][family] = counts[split].get(family, 0) + 1
            first, second = endpoints(item)
            if family == "bonafide":
                if first:
                    bonafide[split].add(first)
            else:
                if first:
                    first_side[split].add(first)
                if second:
                    second_side[split].add(second)

    print("  composition of the Spanish material")
    for split in spanish_items:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts[split].items()))
        print(f"    {split:6s} {len(spanish_items[split]):6d} utterances   {breakdown}")

    def overlap(name: str, sets: Dict[str, Set[str]]) -> None:
        print(f"  {name}")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if a not in sets or b not in sets:
                continue
            shared = sets[a] & sets[b]
            verdict = f"{len(shared)} shared" if shared else "none shared"
            sizes = f"({len(sets[a])} vs {len(sets[b])})"
            print(f"    {a:5s} & {b:5s} {sizes:18s} {verdict}")

    overlap("BONAFIDE speakers  <-- decisive test", bonafide)
    overlap("conversion, first speaker named", first_side)
    overlap("conversion, second speaker named", second_side)

    if "train" in spanish_items and "test" in spanish_items:
        train_all = bonafide["train"] | first_side["train"] | second_side["train"]
        exposed = 0
        for item in spanish_items["test"]:
            first, second = endpoints(item)
            if (first and first in train_all) or (second and second in train_all):
                exposed += 1
        total = len(spanish_items["test"])
        share = 100.0 * exposed / total if total else 0.0
        print("  exposure")
        print(f"    {exposed} of {total} Spanish test utterances ({share:.1f}%) "
              f"involve a speaker also present in training")

    decisive = bonafide.get("train", set()) & bonafide.get("test", set())
    print("  =>", "REAL LEAKAGE: bonafide speakers shared between train and test"
          if decisive else "bonafide speakers are disjoint between train and test")
    print()


def main() -> int:
    """Run the detailed audit over the Spanish and combined partitions."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    print(f"corpus root: {root}\n")
    for label, folder in PARTITIONS.items():
        report(label, root / folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
