"""
Test splice engine with REAL audio from the partial spoof test run.

Uses actual bonafide + cloned WAV files and real alignment timestamps
from data/attacks/qwen_partial_spoof/. Produces spliced output files
that can be listened to for quality assessment.
"""
import json
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

for mod in ["torch", "jiwer", "nemo", "nemo.collections", "nemo.collections.asr",
            "torchaudio", "speechbrain", "torchmetrics"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import librosa
import soundfile as sf
import numpy as np

importlib.import_module("app.pipeline.partial_spoof.utils.crossfade")
splice_mod = importlib.import_module("app.pipeline.partial_spoof.utils.splice_engine")
splice_words = splice_mod.splice_words

SAMPLE_RATE = 16000
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "attacks" / "qwen_partial_spoof"
OUTPUT_DIR = DATA_DIR / "respliced"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    align_path = DATA_DIR / "alignment_metadata.json"
    with open(align_path, "r", encoding="utf-8") as f:
        alignment = json.load(f)

    select_path = DATA_DIR / "word_selection_metadata.json"
    with open(select_path, "r", encoding="utf-8") as f:
        selections = json.load(f)

    print(f"Loaded {len(alignment)} aligned samples, {len(selections)} selections")
    print(f"Output dir: {OUTPUT_DIR}\n")

    for sample_key, entry in alignment.items():
        project_root = Path(__file__).parent.parent.parent

        bf_rel = entry["bonafide_audio_path"].replace(
            "bonafide_dataset_by_speaker_v2", "bonafide_dataset_by_speaker"
        )
        bonafide_path = project_root / bf_rel

        cl_rel = entry["cloned_audio_path"].replace(
            "data/qwen_partial_spoof/", "data/attacks/qwen_partial_spoof/"
        )
        cloned_path = project_root / cl_rel

        if not bonafide_path.exists():
            print(f"SKIP {sample_key}: bonafide not found at {bonafide_path}")
            continue
        if not cloned_path.exists():
            print(f"SKIP {sample_key}: cloned not found at {cloned_path}")
            continue

        bonafide_audio, _ = librosa.load(str(bonafide_path), sr=SAMPLE_RATE, mono=True)
        cloned_audio, _ = librosa.load(str(cloned_path), sr=SAMPLE_RATE, mono=True)

        print(f"{'=' * 70}")
        print(f"Sample: {sample_key}")
        print(f"  Transcript: {entry['transcript']}")
        print(f"  Bonafide: {len(bonafide_audio)/SAMPLE_RATE:.2f}s  |  Clone: {len(cloned_audio)/SAMPLE_RATE:.2f}s")
        print(f"  Bonafide words: {[w['word'] for w in entry['bonafide_words']]}")
        print(f"  Cloned words:   {[w['word'] for w in entry['cloned_words']]}")

        if sample_key not in selections:
            print(f"  No selections for this sample")
            continue

        for sel in selections[sample_key]["selections"]:
            tier = sel["tier"]
            indices = sel["selected_indices"]
            target_words = [entry["bonafide_words"][i]["word"] for i in indices]

            print(f"\n  --- {tier}: replacing {target_words} (indices {indices}) ---")

            result, details = splice_words(
                bonafide_audio=bonafide_audio,
                cloned_audio=cloned_audio,
                bonafide_words=entry["bonafide_words"],
                cloned_words=entry["cloned_words"],
                selected_indices=indices,
                sample_rate=SAMPLE_RATE,
                crossfade_ms=20.0,
                max_silence_steal_ms=50.0,
                max_stretch_ratio=1.1,
            )

            out_name = f"RESPLICED_{tier}_{sample_key}.wav"
            out_path = OUTPUT_DIR / out_name
            sf.write(str(out_path), result, SAMPLE_RATE)

            spliced_words = [d["word"] for d in details]
            print(f"  Spliced: {spliced_words} ({len(details)}/{len(indices)})")
            for d in details:
                print(f"    [{d['word_index']}] '{d['word']}' "
                      f"bf=[{d['bonafide_start_s']:.2f}-{d['bonafide_end_s']:.2f}] "
                      f"cl=[{d['cloned_start_s']:.2f}-{d['cloned_end_s']:.2f}] "
                      f"ratio={d['duration_ratio']:.3f}")
            print(f"  Saved: {out_path}")

    print(f"\n{'=' * 70}")
    print(f"All respliced files saved to: {OUTPUT_DIR}")
    print(f"Compare with originals in: {DATA_DIR / 'spliced'}")
    print("Listen and compare the boundary artifacts!")


if __name__ == "__main__":
    main()
