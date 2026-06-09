"""
Presentation builder - assembles individual slide files into presentation.html.

Usage:
    python presentation/build.py

Reads header.html, all slides/*.html in sorted order, and footer.html.
Auto-assigns sequential slide IDs (s1, s2, ...) and injects the slide count.
Output: presentation.html in the project root.
"""
import re
from pathlib import Path


def build():
    """Assemble presentation.html from components."""
    base_dir = Path(__file__).parent
    project_root = base_dir.parent

    header = (base_dir / "header.html").read_text(encoding="utf-8")
    footer = (base_dir / "footer.html").read_text(encoding="utf-8")

    slides_dir = base_dir / "slides"
    slide_files = sorted(slides_dir.glob("*.html"))

    if not slide_files:
        raise FileNotFoundError(f"No slide files found in {slides_dir}")

    slide_count = len(slide_files)
    assembled_slides = []

    for idx, slide_file in enumerate(slide_files, start=1):
        content = slide_file.read_text(encoding="utf-8").strip()
        content = re.sub(
            r'<section\s+class="([^"]*)"(?:\s+id="[^"]*")?',
            rf'<section class="\1" id="s{idx}"',
            content,
            count=1,
        )
        slide_name = slide_file.stem
        comment = f"\n<!-- {'=' * 50}\n     {idx:02d} - {slide_name}\n{'=' * 50} -->\n"
        assembled_slides.append(comment + content)

    footer = footer.replace("__SLIDE_COUNT__", str(slide_count))

    output = header + "\n".join(assembled_slides) + "\n\n" + footer
    output_path = project_root / "presentation.html"
    output_path.write_text(output, encoding="utf-8")

    print(f"Built presentation.html: {slide_count} slides")
    for i, sf in enumerate(slide_files, 1):
        print(f"  s{i:2d} = {sf.name}")


if __name__ == "__main__":
    build()
