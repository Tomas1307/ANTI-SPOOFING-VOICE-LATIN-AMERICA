"""
Print a chronological timeline of every transformation in a partial-spoof
jitter utterance. Reads directly from the events_timeline field that
Step 5b writes into boundary_jitter_metadata.json -- no recomputation.

Usage on ml-server03:
    python -m app.scripts.describe_jitter_sample <base_name>

Example output:
    Utterance: arf_00295_clip01_W2
    Spoof words      : 2
    Jitter ops       : truncate=2  overlap=3  bleed=1  natural=4
    Length drift     : -832 samples (-52.0 ms)
    -----------------------------------------------------------------
    Time(s)    Type           Detail
    -----------------------------------------------------------------
    0.456      natural        boundary[0]
    0.890      truncate       boundary[1] left_tail 23.4 ms (delta -376)
    1.200      spoof_start    'huyo' (splice: ola_hanning, 45.0 ms crossfade)
    1.200      overlap        boundary[2] 52.1 ms hanning (delta -832)
    1.550      spoof_end      'huyo' (splice: ola_hanning, 45.0 ms crossfade)

The same data is in:
    data/qwen_partial_spoof_jitter/boundary_jitter_metadata.json -> .[<base_name>].events_timeline
You can also query it with jq directly.
"""
import json
import sys
from pathlib import Path
from typing import Dict


OUTPUT_DIR = Path("data/qwen_partial_spoof_jitter")


def _load_json(path: Path) -> dict:
    """Read a JSON file from disk and return its parsed contents.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_event_detail(event: dict) -> str:
    """Render a single event's detail string for the timeline display.

    Args:
        event: Event dict from boundary_jitter_metadata.json events_timeline.

    Returns:
        Human-readable detail string for the event.
    """
    op = event.get("type", "?")

    if op in ("spoof_start", "spoof_end"):
        word = event.get("word", "?")
        method = event.get("splice_method", "?")
        cf_ms = event.get("crossfade_ms", 0.0) or 0.0
        return f"'{word}' (splice: {method}, {cf_ms:.1f} ms crossfade)"

    boundary = event.get("boundary_index")
    bidx = f"boundary[{boundary}]" if boundary is not None else "boundary[?]"

    if op == "natural":
        return bidx
    if op == "truncate":
        return (
            f"{bidx} {event.get('side', '?')} "
            f"{event.get('duration_ms', 0.0):.1f} ms "
            f"(delta {int(event.get('delta_samples', 0)):+d} samples)"
        )
    if op == "overlap":
        return (
            f"{bidx} "
            f"{event.get('duration_ms', 0.0):.1f} ms "
            f"{event.get('fade', '?')} "
            f"(delta {int(event.get('delta_samples', 0)):+d} samples)"
        )
    if op == "bleed":
        return (
            f"{bidx} {event.get('direction', '?')} "
            f"{event.get('duration_ms', 0.0):.1f} ms "
            f"(delta {int(event.get('delta_samples', 0)):+d} samples)"
        )
    return bidx


def main() -> int:
    """Pretty-print the chronological event timeline for one utterance.

    Returns:
        Process exit code (0 on success, 1 on missing input).
    """
    if len(sys.argv) < 2:
        print("Usage: python -m app.scripts.describe_jitter_sample <base_name>")
        print("Example: python -m app.scripts.describe_jitter_sample arf_00295_clip01_W2")
        return 1

    base_name = sys.argv[1]

    jitter_meta = _load_json(OUTPUT_DIR / "boundary_jitter_metadata.json")
    splice_meta = _load_json(OUTPUT_DIR / "splice_metadata.json")

    if base_name not in jitter_meta:
        print(f"Base name '{base_name}' not found in boundary_jitter_metadata.json")
        print(f"Available keys (first 10): {list(jitter_meta.keys())[:10]}")
        return 1

    jitter_entry = jitter_meta[base_name]
    splice_entry = splice_meta.get(base_name, {})
    events = jitter_entry.get("events_timeline", [])
    drift_samples = int(jitter_entry.get("drift_samples", 0))
    drift_ms = drift_samples / 16000.0 * 1000.0

    counts: Dict[str, int] = {
        "spoof_start": 0,
        "spoof_end": 0,
        "natural": 0,
        "truncate": 0,
        "overlap": 0,
        "bleed": 0,
    }
    for e in events:
        t = e.get("type", "?")
        counts[t] = counts.get(t, 0) + 1

    print(f"Utterance: {base_name}")
    print(f"Total duration   : {splice_entry.get('total_duration_s', 0.0):.2f} s")
    print(f"Spoof words      : {len(splice_entry.get('spoofed_words', []))}")
    print(
        f"Jitter ops       : truncate={counts['truncate']}  overlap={counts['overlap']}  "
        f"bleed={counts['bleed']}  natural={counts['natural']}"
    )
    print(f"Length drift     : {drift_samples:+d} samples ({drift_ms:+.1f} ms)")
    print("-" * 80)
    print(f"{'Time(s)':<11}{'Type':<15}Detail")
    print("-" * 80)
    for e in events:
        print(f"{e.get('time_s', 0.0):<11.3f}{e.get('type', '?'):<15}{_format_event_detail(e)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
