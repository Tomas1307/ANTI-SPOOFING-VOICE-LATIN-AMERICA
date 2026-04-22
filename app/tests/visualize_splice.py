"""
Visualize splice boundaries to diagnose artifacts.
Generates an HTML page with waveform plots around each splice point.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SAMPLE_RATE = 16000
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "attacks" / "qwen_partial_spoof"


def load_audio(rel_path_str):
    """Load audio with fallback path resolution."""
    p = PROJECT_ROOT / rel_path_str
    if not p.exists():
        p = PROJECT_ROOT / rel_path_str.replace(
            "bonafide_dataset_by_speaker_v2", "bonafide_dataset_by_speaker"
        )
    if not p.exists():
        p = PROJECT_ROOT / rel_path_str.replace(
            "data/qwen_partial_spoof/", "data/attacks/qwen_partial_spoof/"
        )
    audio, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
    return audio


def samples_to_svg_polyline(audio, start_s, end_s, width=800, height=120):
    """Convert audio samples to SVG polyline points."""
    start_sample = int(start_s * SAMPLE_RATE)
    end_sample = int(end_s * SAMPLE_RATE)
    segment = audio[max(0, start_sample):min(len(audio), end_sample)]

    if len(segment) == 0:
        return ""

    step = max(1, len(segment) // width)
    downsampled = segment[::step][:width]

    points = []
    for i, val in enumerate(downsampled):
        x = i * (width / len(downsampled))
        y = height / 2 - val * (height / 2) * 0.9
        points.append(f"{x:.1f},{y:.1f}")

    return " ".join(points)


def main():
    splice_path = DATA_DIR / "splice_metadata.json"
    with open(splice_path, "r", encoding="utf-8") as f:
        splice_meta = json.load(f)

    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Splice Boundary Visualization</title>
<style>
body { background: #0a0a0a; color: #fff; font-family: 'Segoe UI', sans-serif; padding: 40px; }
h1 { font-size: 28px; margin-bottom: 30px; }
h2 { font-size: 18px; color: #aaa; margin: 30px 0 10px; }
.sample { background: #111; border: 1px solid #333; border-radius: 8px; padding: 24px; margin-bottom: 24px; }
.label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.word-info { font-size: 14px; color: #f5a623; margin-bottom: 12px; }
svg { background: #0d0d0d; border: 1px solid #222; border-radius: 4px; display: block; margin: 8px 0; }
.row { display: flex; gap: 20px; margin-bottom: 16px; }
.col { flex: 1; }
.splice-line { stroke: #ff4d4d; stroke-width: 2; stroke-dasharray: 4,4; }
.bonafide-wave { stroke: #3eff8b; stroke-width: 1; fill: none; }
.cloned-wave { stroke: #4d9fff; stroke-width: 1; fill: none; }
.spliced-wave { stroke: #f5a623; stroke-width: 1; fill: none; }
.old-spliced-wave { stroke: #ff4d4d; stroke-width: 1; fill: none; opacity: 0.6; }
audio { width: 100%; height: 36px; filter: invert(1) hue-rotate(180deg) brightness(0.7); margin-top: 8px; }
</style></head><body>
<h1>Splice Boundary Visualization</h1>
<p style="color:#888; margin-bottom:30px;">Comparing bonafide, cloned, old spliced (5ms linear), and new respliced (20ms cosine + zero-crossing + energy norm) at each splice boundary.</p>
"""]

    for splice_key, entry in splice_meta.items():
        if not entry["spoofed_words"]:
            continue

        bonafide_audio = load_audio(entry["bonafide_audio_path"])
        cloned_audio = load_audio(entry["cloned_audio_path"])

        old_spliced_path = DATA_DIR / "spliced" / Path(entry["spliced_audio_path"]).name
        new_spliced_path = DATA_DIR / "respliced" / f"RESPLICED_{entry['tier']}_{splice_key.rsplit('_' + entry['tier'].replace('W','W'), 1)[0]}.wav"

        # Try to find the respliced file
        respliced_name = f"RESPLICED_{entry['tier']}_{splice_key.replace('_' + entry['tier'], '')}.wav"
        new_spliced_path = DATA_DIR / "respliced" / respliced_name

        old_spliced = load_audio(str(old_spliced_path)) if old_spliced_path.exists() else None
        new_spliced = load_audio(str(new_spliced_path)) if new_spliced_path.exists() else None

        html_parts.append(f'<div class="sample">')
        html_parts.append(f'<h2>{splice_key}</h2>')
        html_parts.append(f'<div class="word-info">Transcript: {entry["transcript"]}</div>')

        for word_info in entry["spoofed_words"]:
            word = word_info["word"]
            bf_start = word_info["bonafide_start_s"]
            bf_end = word_info["bonafide_end_s"]
            cl_start = word_info["cloned_start_s"]
            cl_end = word_info["cloned_end_s"]

            # Show 300ms context around the splice
            margin = 0.3
            view_start = max(0, bf_start - margin)
            view_end = min(len(bonafide_audio) / SAMPLE_RATE, bf_end + margin)
            view_dur = view_end - view_start

            w = 800
            h = 120

            # Splice boundary positions in SVG coordinates
            left_x = (bf_start - view_start) / view_dur * w
            right_x = (bf_end - view_start) / view_dur * w

            html_parts.append(f'<div class="word-info">Spliced word: "<strong>{word}</strong>" [{bf_start:.2f}s - {bf_end:.2f}s] ratio={word_info["duration_ratio"]:.3f}</div>')

            # Bonafide waveform
            html_parts.append('<div class="label">Bonafide (original)</div>')
            bf_points = samples_to_svg_polyline(bonafide_audio, view_start, view_end, w, h)
            html_parts.append(f'<svg width="{w}" height="{h}"><polyline class="bonafide-wave" points="{bf_points}"/>'
                            f'<line x1="{left_x}" y1="0" x2="{left_x}" y2="{h}" class="splice-line"/>'
                            f'<line x1="{right_x}" y1="0" x2="{right_x}" y2="{h}" class="splice-line"/></svg>')

            # Cloned waveform (at cloned timestamps)
            cl_view_start = max(0, cl_start - margin)
            cl_view_end = cl_end + margin
            html_parts.append('<div class="label">Cloned (Qwen TTS)</div>')
            cl_points = samples_to_svg_polyline(cloned_audio, cl_view_start, cl_view_end, w, h)
            cl_left_x = (cl_start - cl_view_start) / (cl_view_end - cl_view_start) * w
            cl_right_x = (cl_end - cl_view_start) / (cl_view_end - cl_view_start) * w
            html_parts.append(f'<svg width="{w}" height="{h}"><polyline class="cloned-wave" points="{cl_points}"/>'
                            f'<line x1="{cl_left_x}" y1="0" x2="{cl_left_x}" y2="{h}" class="splice-line"/>'
                            f'<line x1="{cl_right_x}" y1="0" x2="{cl_right_x}" y2="{h}" class="splice-line"/></svg>')

            # Old spliced (5ms linear)
            if old_spliced is not None:
                html_parts.append('<div class="label">Old spliced (5ms linear crossfade)</div>')
                old_points = samples_to_svg_polyline(old_spliced, view_start, view_end, w, h)
                html_parts.append(f'<svg width="{w}" height="{h}"><polyline class="old-spliced-wave" points="{old_points}"/>'
                                f'<line x1="{left_x}" y1="0" x2="{left_x}" y2="{h}" class="splice-line"/>'
                                f'<line x1="{right_x}" y1="0" x2="{right_x}" y2="{h}" class="splice-line"/></svg>')

            # New respliced (20ms cosine + zero-crossing + energy norm)
            if new_spliced is not None:
                html_parts.append('<div class="label">New respliced (20ms cosine + zero-crossing + energy norm)</div>')
                new_points = samples_to_svg_polyline(new_spliced, view_start, view_end, w, h)
                html_parts.append(f'<svg width="{w}" height="{h}"><polyline class="spliced-wave" points="{new_points}"/>'
                                f'<line x1="{left_x}" y1="0" x2="{left_x}" y2="{h}" class="splice-line"/>'
                                f'<line x1="{right_x}" y1="0" x2="{right_x}" y2="{h}" class="splice-line"/></svg>')

            # Audio players
            html_parts.append('<div class="row">')
            html_parts.append('<div class="col"><div class="label">Bonafide</div>')
            html_parts.append(f'<audio controls src="{entry["bonafide_audio_path"].replace("bonafide_dataset_by_speaker_v2","bonafide_dataset_by_speaker")}"></audio></div>')
            if old_spliced is not None:
                html_parts.append(f'<div class="col"><div class="label">Old (5ms)</div>')
                html_parts.append(f'<audio controls src="data/attacks/qwen_partial_spoof/spliced/{Path(entry["spliced_audio_path"]).name}"></audio></div>')
            if new_spliced is not None:
                html_parts.append(f'<div class="col"><div class="label">New (20ms)</div>')
                html_parts.append(f'<audio controls src="data/attacks/qwen_partial_spoof/respliced/{respliced_name}"></audio></div>')
            html_parts.append('</div>')

        html_parts.append('</div>')

    html_parts.append('</body></html>')

    out_path = PROJECT_ROOT / "splice_debug.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Visualization saved to: {out_path}")
    print("Open in browser to compare waveforms and listen side by side.")


if __name__ == "__main__":
    main()
