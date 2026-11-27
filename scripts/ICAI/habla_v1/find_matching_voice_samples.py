"""Find one speaker present across TTS, StarGAN, CycleGAN and DiffVC alike.

WHY
---
For an ICAI presentation slide contrasting attack families (TTS vs. voice
conversion), the reference audio samples only make the intended point if
they all target the SAME voice. Grabbing the first match from each attack
family independently (as an earlier ad hoc `grep`/`find` pass did) picks
four different speakers, which defeats the comparison.

This script scans every real audio file under DATA_LAT/asvspoof2019_LA's
data/, eval/ and train_dev/ directories, classifies each by attack family
from its filename prefix (TTS-, StarGAN-, CycleGAN-, Diff-; chained attacks
such as TTS-StarGAN- or TTS-Diff- are excluded, since they are a different,
compound condition), and extracts every speaker token embedded in the
filename (source and target both, for voice-conversion files, since the
corpus naming convention does not by itself disambiguate which is which --
see analyze_leakage_detail.py). It then reports every speaker token that
appears in all four families, with one example file path per family, so a
single voice identity can be used for every reference sample.

Read-only with respect to the corpus.

Usage on ml-server03:

    python3 scripts/ICAI/habla_v1/find_matching_voice_samples.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

DEFAULT_ROOT = Path(
    "/home/jahurtado905/notebooks/anti-spoofing/anti-spoof-eval/03-asvspoof-mega"
)

SEARCH_DIRS = ("data", "eval", "train_dev")

# Exact single-attack prefixes only; chained attacks (TTS-StarGAN-,
# TTS-Diff-) are deliberately excluded by the trailing '-' plus a
# non-uppercase-letter lookahead-equivalent check below.
ATTACK_PREFIXES: Dict[str, str] = {
    "TTS-": "TTS",
    "StarGAN-": "StarGAN",
    "CycleGAN-": "CycleGAN",
    "Diff-": "DiffVC",
}
CHAINED_MARKERS = ("TTS-StarGAN-", "TTS-Diff-", "TTS-CycleGAN-")

SPEAKER_TOKEN = re.compile(r"(?:ar|cl|co|pe|ve)[fm]_\d+")


def classify(filename: str) -> str:
    """Return the attack family for one filename, or '' if unrecognized.

    Args:
        filename: Basename of the audio file (with or without extension).

    Returns:
        One of 'TTS', 'StarGAN', 'CycleGAN', 'DiffVC', or '' for anything
        that does not match a single, non-chained attack prefix (including
        bonafide files, which carry no attack prefix at all).
    """
    if any(filename.startswith(marker) for marker in CHAINED_MARKERS):
        return ""
    for prefix, family in ATTACK_PREFIXES.items():
        if filename.startswith(prefix):
            return family
    return ""


def main() -> int:
    """Scan the corpus and print speakers shared by all four attack families."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    la_dir = root / "DATA_LAT" / "asvspoof2019_LA"
    print(f"corpus root: {la_dir}\n")

    # family -> speaker token -> list of example file paths
    by_family: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))

    for sub in SEARCH_DIRS:
        d = la_dir / sub
        if not d.exists():
            continue
        for path in d.glob("*.wav"):
            family = classify(path.name)
            if not family:
                continue
            for speaker in set(SPEAKER_TOKEN.findall(path.name)):
                by_family[family][speaker].append(path)

    for family in ATTACK_PREFIXES.values():
        n_files = sum(len(v) for v in by_family[family].values())
        print(f"  {family:10s} {len(by_family[family]):5d} distinct speakers, "
              f"{n_files} speaker-file associations")

    speaker_sets: List[Set[str]] = [
        set(by_family[family].keys()) for family in ATTACK_PREFIXES.values()
    ]
    shared = set.intersection(*speaker_sets) if all(speaker_sets) else set()

    print(f"\nSpeakers present in all four families: {len(shared)}\n")
    for speaker in sorted(shared):
        print(f"  {speaker}")
        for family in ATTACK_PREFIXES.values():
            example = by_family[family][speaker][0]
            print(f"    {family:10s} {example}")
        print()

    if not shared:
        print("No single speaker appears in all four families. Re-run allowing "
              "any three families, or treat source vs. target separately.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
