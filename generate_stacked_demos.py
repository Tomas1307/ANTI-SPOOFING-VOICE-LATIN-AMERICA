"""
Generate stacked augmentation demo samples.

Takes each speaker's existing rir.wav (RIR+Noise already applied) and runs a
codec round-trip on top, producing samples that represent the new stacking gate:
    RIR+Noise -> Codec

Three codec variants are generated per speaker:
    stacked_rir_g711.wav    G.711 u-law  (8 kHz telephony)
    stacked_rir_opus.wav    Opus         (16 kHz VoIP / WebRTC)
    stacked_rir_amr.wav     AMR-NB       (8 kHz mobile voice, if available)

Updates data/demo_augmentations/manifest.json with the new entries.

Requirements:
    ffmpeg in PATH (same as the augmentation pipeline)
    soundfile, numpy  (pip install soundfile numpy)

Usage:
    python generate_stacked_demos.py
    python generate_stacked_demos.py --dry-run   # print commands only
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


MANIFEST_PATH = Path("data/demo_augmentations/manifest.json")
DEMO_ROOT     = Path("data/demo_augmentations")
TARGET_SR     = 16000

SPEAKERS = ["bonafide", "fishgram", "qwen", "chatterbox", "openvoice", "outetts", "omnivoice"]

# Codec definitions: (output_name, ffmpeg_encode_args, ffmpeg_encode_ext)
# Each entry encodes the source wav into a lossy format then decodes it back.
CODECS = [
    {
        "name":    "stacked_rir_g711",
        "label":   "Stacked: RIR+Noise -> G.711 u-law (8 kHz telephony)",
        "encode":  ["-ar", "8000", "-acodec", "pcm_mulaw"],
        "ext":     ".wav",
    },
    {
        "name":    "stacked_rir_opus",
        "label":   "Stacked: RIR+Noise -> Opus (16 kHz VoIP)",
        "encode":  ["-c:a", "libopus", "-ar", "16000", "-b:a", "16000"],
        "ext":     ".ogg",
    },
    {
        "name":    "stacked_rir_amr",
        "label":   "Stacked: RIR+Noise -> AMR-NB (8 kHz mobile)",
        "encode":  ["-ar", "8000", "-b:a", "7950"],
        "ext":     ".amr",
    },
]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_codec(ext: str, encode_args: list, tmp_dir: Path, sample_sr: int = TARGET_SR) -> bool:
    """Check whether a codec is usable by encoding 0.1 s of silence."""
    silence = np.zeros(int(sample_sr * 0.1), dtype=np.float32)
    src = tmp_dir / "probe_src.wav"
    mid = tmp_dir / f"probe_mid{ext}"
    sf.write(str(src), silence, sample_sr)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)] + encode_args + [str(mid)]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def apply_codec_roundtrip(
    src_wav: Path,
    dst_wav: Path,
    encode_args: list,
    ext: str,
    tmp_dir: Path,
    dry_run: bool = False,
) -> bool:
    """
    Encode src_wav through a lossy codec then decode back to dst_wav at TARGET_SR.

    Returns True on success.
    """
    mid = tmp_dir / f"mid{ext}"

    encode_cmd = (
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_wav)]
        + encode_args
        + [str(mid)]
    )
    decode_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mid),
        "-ar", str(TARGET_SR),
        "-ac", "1",
        str(dst_wav),
    ]

    if dry_run:
        print("  ENCODE:", " ".join(encode_cmd))
        print("  DECODE:", " ".join(decode_cmd))
        return True

    result = subprocess.run(encode_cmd, capture_output=True)
    if result.returncode != 0:
        print(f"    [WARN] encode failed: {result.stderr.decode(errors='replace').strip()}")
        return False

    result = subprocess.run(decode_cmd, capture_output=True)
    if result.returncode != 0:
        print(f"    [WARN] decode failed: {result.stderr.decode(errors='replace').strip()}")
        return False

    return True


def run(dry_run: bool):
    if not ffmpeg_available():
        print("ERROR: ffmpeg not found in PATH.")
        sys.exit(1)

    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)

        # Probe which codecs are available once
        available_codecs = []
        print("Probing codec availability...")
        for codec in CODECS:
            ok = probe_codec(codec["ext"], codec["encode"], tmp_dir)
            status = "OK" if ok else "SKIP (not available in ffmpeg build)"
            print(f"  {codec['name']:<25} {status}")
            if ok:
                available_codecs.append(codec)
        print()

        if not available_codecs:
            print("ERROR: no codecs available. Check your ffmpeg build.")
            sys.exit(1)

        for speaker in SPEAKERS:
            rir_path = DEMO_ROOT / speaker / "rir.wav"
            if not rir_path.exists():
                print(f"[SKIP] {speaker}: rir.wav not found")
                continue

            print(f"Processing: {speaker}")
            if speaker not in manifest:
                manifest[speaker] = {}

            for codec in available_codecs:
                out_name = f"{codec['name']}.wav"
                out_path = DEMO_ROOT / speaker / out_name
                rel_path = str(out_path).replace("\\", "/")

                print(f"  -> {out_name}  ({codec['label']})")
                success = apply_codec_roundtrip(
                    rir_path, out_path, codec["encode"], codec["ext"], tmp_dir, dry_run
                )

                if success and not dry_run:
                    manifest[speaker][codec["name"]] = rel_path
                    print(f"     saved: {out_path}")

            print()

    if not dry_run:
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"manifest.json updated: {MANIFEST_PATH}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Generate stacked augmentation demo samples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print ffmpeg commands without executing them")
    args = parser.parse_args()
    run(args.dry_run)


if __name__ == "__main__":
    main()
