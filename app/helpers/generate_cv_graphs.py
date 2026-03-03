"""
Common Voice Spanish Geographic Analysis - Graph Generator.

Generates presentation-ready graphs comparing HABLA dataset (18k samples)
against Mozilla Common Voice Spanish v24.0 (436k validated samples).

Analyzes geographic distribution, speaker balance, per-speaker contribution
distributions, and dataset growth potential for Latin American anti-spoofing
research.

Usage:
    python -m app.helpers.generate_cv_graphs

Output:
    data/cv_metadata/graphs/*.png (13 graphs)
"""
import csv
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


TSV_PATH = os.path.join("data", "cv_metadata", "validated.tsv")
OUTPUT_DIR = os.path.join("data", "cv_metadata", "graphs")

LATAM_MAP = {
    "Mexico": ["mexico", "méxico", "cdmx", "ciudad de m", "mexicano"],
    "Andino-Pacifico\n(CO/PE/EC)": [
        "andino", "colombia", "bogot", "peru", "lima",
        "ecuador", "santander", "valle del cauca", "cucuta",
    ],
    "Rioplatense\n(AR/UY/PY)": [
        "rioplatense", "argentin", "paragua", "uruguay", "cordob",
    ],
    "Caribe\n(CU/VE/PR/DO/PA)": [
        "caribe", "cuba", "venezuel", "panam", "puerto rico", "dominican",
    ],
    "Chile": ["chile", "chileno"],
    "America Central": ["central", "guatemala", "costa rica"],
}

SPAIN_KEYWORDS = [
    "espa", "canaria", "peninsular", "madrid", "castilla",
    "andaluc", "catalu", "galicia", "baleares", "valencia",
    "nativo, from spain",
]

LATAM_REGIONS = [
    "Mexico", "Andino-Pacifico\n(CO/PE/EC)", "Rioplatense\n(AR/UY/PY)",
    "America Central", "Chile", "Caribe\n(CU/VE/PR/DO/PA)",
]

COLORS_LATAM = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653"]
COLORS_MACRO = ["#2A9D8F", "#E63946", "#A8DADC", "#6C757D"]
HABLA_COLOR = "#1D3557"
CV_COLOR = "#E63946"

HABLA_SAMPLES = 18000
HABLA_SPEAKERS_EST = 500


def extract_data():
    """Parse validated.tsv and categorize samples by geographic region.

    Returns:
        Tuple containing:
            - region_speakers: dict mapping region to set of speaker IDs
            - region_samples: Counter mapping region to sample count
            - total: int total number of validated samples
            - speaker_counts_by_region: dict mapping region to dict of
              speaker_id -> sample_count for per-speaker distribution analysis
    """
    region_speakers = defaultdict(set)
    region_samples = Counter()
    speaker_counts_by_region = defaultdict(lambda: Counter())
    total = 0

    with open(TSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        accent_idx = header.index("accents")
        client_idx = header.index("client_id")

        for row in reader:
            total += 1
            accent = row[accent_idx].strip().lower()
            speaker = row[client_idx]

            if accent == "":
                region_speakers["Not Provided"].add(speaker)
                region_samples["Not Provided"] += 1
                speaker_counts_by_region["Not Provided"][speaker] += 1
                continue

            matched = False
            for region, keywords in LATAM_MAP.items():
                if any(kw in accent for kw in keywords):
                    region_speakers[region].add(speaker)
                    region_samples[region] += 1
                    speaker_counts_by_region[region][speaker] += 1
                    matched = True
                    break

            if not matched:
                if any(kw in accent for kw in SPAIN_KEYWORDS):
                    region_speakers["Spain"].add(speaker)
                    region_samples["Spain"] += 1
                    speaker_counts_by_region["Spain"][speaker] += 1
                else:
                    region_speakers["Other"].add(speaker)
                    region_samples["Other"] += 1
                    speaker_counts_by_region["Other"][speaker] += 1

    return region_speakers, region_samples, total, speaker_counts_by_region


def apply_style():
    """Apply consistent matplotlib styling for all graphs."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def graph_01_macro_pie(region_samples, total):
    """Graph 1: Macro geographic distribution pie chart."""
    fig, ax = plt.subplots(figsize=(10, 8))

    latam_smp = sum(region_samples[r] for r in LATAM_REGIONS)
    spain_smp = region_samples["Spain"]
    notprov_smp = region_samples["Not Provided"]

    labels = ["Latin America\n(45.1%)", "Spain\n(22.5%)", "Not Provided\n(32.3%)"]
    sizes = [latam_smp, spain_smp, notprov_smp]
    explode = (0.05, 0, 0)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=COLORS_MACRO[:3],
        autopct=lambda pct: f"{int(pct / 100 * total):,}",
        shadow=False, startangle=90, textprops={"fontsize": 13},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")

    ax.set_title(
        "Common Voice Spanish (v24.0)\n"
        "Geographic Distribution - 436,590 Validated Samples",
        fontsize=18, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "01_macro_geographic_pie.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 1: Macro geographic pie DONE")


def graph_02_latam_pie(region_samples):
    """Graph 2: Latin America detail pie chart."""
    fig, ax = plt.subplots(figsize=(10, 8))

    latam_smp = sum(region_samples[r] for r in LATAM_REGIONS)
    latam_labels = []
    latam_sizes = []
    for r in LATAM_REGIONS:
        s = region_samples[r]
        pct = s / latam_smp * 100
        latam_labels.append(f"{r}\n({pct:.1f}%)")
        latam_sizes.append(s)

    explode_la = (0.05, 0.05, 0, 0, 0, 0.1)
    wedges, texts, autotexts = ax.pie(
        latam_sizes, explode=explode_la, labels=latam_labels, colors=COLORS_LATAM,
        autopct=lambda pct: f"{int(pct / 100 * latam_smp):,}" if pct > 1 else "",
        shadow=False, startangle=90, textprops={"fontsize": 12},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")

    ax.set_title(
        "Latin American Samples Breakdown\n"
        "197,082 Samples from 3,175 Speakers",
        fontsize=18, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "02_latam_detail_pie.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 2: LatAm detail pie DONE")


def graph_03_speakers_bar(region_speakers):
    """Graph 3: Unique speakers per region horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

    all_regions = LATAM_REGIONS + ["Spain", "Not Provided"]
    spk_counts = [len(region_speakers[r]) for r in all_regions]
    bar_colors = COLORS_LATAM + ["#E9C46A", "#A8DADC"]

    bars = ax.barh(all_regions, spk_counts, color=bar_colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, spk_counts):
        ax.text(
            bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", fontsize=12, fontweight="bold",
        )

    ax.set_xlabel("Number of Unique Speakers", fontsize=14)
    ax.set_title(
        "Unique Speakers per Region - Common Voice Spanish v24.0",
        fontsize=18, fontweight="bold",
    )
    ax.set_xlim(0, max(spk_counts) * 1.2)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "03_speakers_per_region_bar.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 3: Speakers per region bar DONE")


def graph_04_samples_bar(region_samples):
    """Graph 4: Validated samples per region horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

    all_regions = LATAM_REGIONS + ["Spain", "Not Provided"]
    smp_counts = [region_samples[r] for r in all_regions]
    bar_colors = COLORS_LATAM + ["#E9C46A", "#A8DADC"]

    bars = ax.barh(all_regions, smp_counts, color=bar_colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, smp_counts):
        ax.text(
            bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", fontsize=12, fontweight="bold",
        )

    ax.set_xlabel("Number of Validated Samples", fontsize=14)
    ax.set_title(
        "Validated Samples per Region - Common Voice Spanish v24.0",
        fontsize=18, fontweight="bold",
    )
    ax.set_xlim(0, max(smp_counts) * 1.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "04_samples_per_region_bar.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 4: Samples per region bar DONE")


def graph_05_ratio_bar(region_speakers, region_samples):
    """Graph 5: Average samples per speaker ratio bar chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

    all_regions = LATAM_REGIONS + ["Spain", "Not Provided"]
    bar_colors = COLORS_LATAM + ["#E9C46A", "#A8DADC"]

    ratios = []
    ratio_regions = []
    ratio_colors = []
    for i, r in enumerate(all_regions):
        spk = len(region_speakers[r])
        smp = region_samples[r]
        if spk > 0:
            ratios.append(smp / spk)
            ratio_regions.append(r)
            ratio_colors.append(bar_colors[i])

    bars = ax.barh(ratio_regions, ratios, color=ratio_colors, edgecolor="white", linewidth=1.5)

    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{ratio:.1f}", va="center", fontsize=12, fontweight="bold",
        )

    ax.set_xlabel("Average Samples per Speaker", fontsize=14)
    ax.set_title(
        "Contribution Intensity - Average Samples per Speaker",
        fontsize=18, fontweight="bold",
    )
    ax.axvline(
        x=np.mean(ratios), color="red", linestyle="--", linewidth=2,
        label=f"Mean: {np.mean(ratios):.1f}",
    )
    ax.legend(fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "05_samples_per_speaker_ratio.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 5: Samples/speaker ratio DONE")


def graph_06_habla_vs_cv(region_samples, total):
    """Graph 6: HABLA vs Common Voice side-by-side comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    latam_smp = sum(region_samples[r] for r in LATAM_REGIONS)
    andino_smp = region_samples["Andino-Pacifico\n(CO/PE/EC)"]

    # Left: Sample count
    ax1 = axes[0]
    categories = ["HABLA\n(Your Dataset)", "Common Voice\n(LatAm Only)", "Common Voice\n(All Spanish)"]
    values = [HABLA_SAMPLES, latam_smp, total]
    bar_cols = [HABLA_COLOR, CV_COLOR, "#A8DADC"]
    bars = ax1.bar(categories, values, color=bar_cols, edgecolor="white", linewidth=2, width=0.6)
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
            f"{val:,}", ha="center", fontsize=13, fontweight="bold",
        )
    ax1.set_ylabel("Number of Samples", fontsize=14)
    ax1.set_title("Sample Count Comparison", fontsize=16, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: Growth factor
    ax2 = axes[1]
    growth_cats = ["+ Andino Only\n(CO/PE/EC)", "+ All LatAm", "+ All Spanish"]
    growth_vals = [
        (HABLA_SAMPLES + andino_smp) / HABLA_SAMPLES,
        (HABLA_SAMPLES + latam_smp) / HABLA_SAMPLES,
        (HABLA_SAMPLES + total) / HABLA_SAMPLES,
    ]
    growth_cols = ["#457B9D", CV_COLOR, "#A8DADC"]
    bars = ax2.bar(growth_cats, growth_vals, color=growth_cols, edgecolor="white", linewidth=2, width=0.6)
    for bar, val in zip(bars, growth_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}x", ha="center", fontsize=15, fontweight="bold",
        )
    ax2.set_ylabel("Growth Factor (vs. 18k baseline)", fontsize=14)
    ax2.set_title("Dataset Growth Potential", fontsize=16, fontweight="bold")
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    fig.suptitle(
        "HABLA (18k) vs Common Voice Spanish (436k)",
        fontsize=20, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "06_habla_vs_commonvoice.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 6: HABLA vs CV comparison DONE")


def graph_07_imbalance_log(region_speakers):
    """Graph 7: Latin America speaker imbalance on log scale."""
    fig, ax = plt.subplots(figsize=(14, 7))

    latam_spk = [
        (r, len(region_speakers[r]))
        for r in LATAM_REGIONS if len(region_speakers[r]) > 0
    ]
    latam_spk.sort(key=lambda x: -x[1])

    regions_sorted = [x[0] for x in latam_spk]
    counts_sorted = [x[1] for x in latam_spk]

    bars = ax.bar(
        range(len(regions_sorted)), counts_sorted,
        color=COLORS_LATAM[:len(regions_sorted)], edgecolor="white", linewidth=2,
    )

    for bar, count in zip(bars, counts_sorted):
        ypos = bar.get_height() * 1.3 if count > 10 else bar.get_height() + 2
        ax.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{count:,}", ha="center", fontsize=13, fontweight="bold",
        )

    ax.set_xticks(range(len(regions_sorted)))
    ax.set_xticklabels(regions_sorted, fontsize=11)
    ax.set_ylabel("Number of Unique Speakers (log scale)", fontsize=14)
    ax.set_yscale("log")
    ax.set_title(
        "Latin America Speaker Imbalance (Log Scale)\n"
        "382:1 ratio between Mexico and Caribbean",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "07_latam_imbalance_log.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 7: Imbalance log scale DONE")


def graph_08_andino_focus(region_samples):
    """Graph 8: Andino-Pacifico focus impact for Colombian research."""
    fig, ax = plt.subplots(figsize=(12, 7))

    latam_smp = sum(region_samples[r] for r in LATAM_REGIONS)
    andino_smp = region_samples["Andino-Pacifico\n(CO/PE/EC)"]

    categories = [
        "Your HABLA\nDataset", "Common Voice\nAndino Only",
        "Combined\n(HABLA + Andino)", "Common Voice\nAll LatAm",
    ]
    values = [HABLA_SAMPLES, andino_smp, HABLA_SAMPLES + andino_smp, latam_smp]
    bar_cols = [HABLA_COLOR, "#2A9D8F", "#E9C46A", CV_COLOR]

    bars = ax.bar(categories, values, color=bar_cols, edgecolor="white", linewidth=2, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
            f"{val:,}", ha="center", fontsize=14, fontweight="bold",
        )

    ax.set_ylabel("Number of Samples", fontsize=14)
    ax.set_title(
        "Impact for Colombian/Andean Research Focus\n"
        "HABLA + Common Voice Andino-Pacifico",
        fontsize=18, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "08_andino_focus_impact.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 8: Andino focus impact DONE")


def graph_09_speaker_diversity(region_speakers):
    """Graph 9: Speaker diversity comparison HABLA vs Common Voice."""
    fig, ax = plt.subplots(figsize=(12, 7))

    all_regions = LATAM_REGIONS + ["Spain", "Not Provided"]
    all_cv_speakers = sum(len(region_speakers[r]) for r in all_regions)
    andino_spk = len(region_speakers["Andino-Pacifico\n(CO/PE/EC)"])

    categories = [
        "HABLA\n(Estimated)", "CV Andino\nSpeakers",
        "CV All LatAm\nSpeakers", "CV All Spanish\nSpeakers",
    ]
    values = [HABLA_SPEAKERS_EST, andino_spk, 3175, all_cv_speakers]
    bar_cols = [HABLA_COLOR, "#2A9D8F", CV_COLOR, "#A8DADC"]

    bars = ax.bar(categories, values, color=bar_cols, edgecolor="white", linewidth=2, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}", ha="center", fontsize=14, fontweight="bold",
        )

    ax.set_ylabel("Number of Unique Speakers", fontsize=14)
    ax.set_title(
        "Speaker Diversity Comparison\nHABLA vs Common Voice",
        fontsize=18, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "09_speaker_diversity.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 9: Speaker diversity DONE")


def graph_10_undeclared(region_speakers):
    """Graph 10: Undeclared accent speakers pie chart."""
    fig, ax = plt.subplots(figsize=(10, 8))

    declared_spk = (
        sum(len(region_speakers[r]) for r in LATAM_REGIONS)
        + len(region_speakers["Spain"])
    )
    undeclared_spk = len(region_speakers["Not Provided"])
    total_spk = declared_spk + undeclared_spk

    labels = [
        f"Declared Accent\n{declared_spk:,} speakers\n({declared_spk / total_spk * 100:.1f}%)",
        f"NOT Declared\n{undeclared_spk:,} speakers\n({undeclared_spk / total_spk * 100:.1f}%)",
    ]
    sizes = [declared_spk, undeclared_spk]
    colors_decl = ["#2A9D8F", "#D62828"]
    explode = (0, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors_decl,
        autopct=lambda pct: f"{int(pct / 100 * total_spk):,}",
        shadow=False, startangle=90, textprops={"fontsize": 14},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight("bold")
        t.set_color("white")

    ax.set_title(
        "The Undeclared Speaker Problem\n"
        "75.2% of Speakers Did NOT Declare Their Accent",
        fontsize=18, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "10_undeclared_black_hole.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 10: Undeclared black hole DONE")


def graph_11_boxplot_samples_per_speaker(speaker_counts_by_region):
    """Graph 11: Box plot of samples-per-speaker distribution per region.

    Reveals how evenly samples are distributed across speakers within
    each region. Wide boxes indicate high variance (power users vs
    casual contributors).
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    all_regions = LATAM_REGIONS + ["Spain", "Not Provided"]
    bar_colors = COLORS_LATAM + ["#E9C46A", "#A8DADC"]

    plot_data = []
    plot_labels = []
    valid_colors = []
    stats_text = []

    for i, region in enumerate(all_regions):
        counts = list(speaker_counts_by_region[region].values())
        if len(counts) < 2:
            continue
        plot_data.append(counts)
        plot_labels.append(region)
        valid_colors.append(bar_colors[i])

        arr = np.array(counts)
        median = np.median(arr)
        mean = np.mean(arr)
        std = np.std(arr)
        max_val = np.max(arr)
        min_val = np.min(arr)
        stats_text.append(
            f"Mean:{mean:.0f} Med:{median:.0f}\n"
            f"Std:{std:.0f} Max:{max_val:,}"
        )

    bp = ax.boxplot(
        plot_data, vert=True, patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )

    for patch, color in zip(bp["boxes"], valid_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, txt in enumerate(stats_text):
        q3 = np.percentile(plot_data[i], 75)
        ax.text(
            i + 1, q3 + (q3 * 0.15), txt,
            ha="center", fontsize=8, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    ax.set_xticklabels(plot_labels, fontsize=10)
    ax.set_ylabel("Samples per Speaker", fontsize=14)
    ax.set_title(
        "Distribution of Samples per Speaker by Region (Outliers Hidden)\n"
        "Wide boxes = high inequality among contributors",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "11_boxplot_samples_per_speaker.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 11: Boxplot samples per speaker DONE")


def graph_12_mean_median_comparison(speaker_counts_by_region):
    """Graph 12: Mean vs Median samples per speaker per region.

    A large gap between mean and median indicates that a small number of
    power users are inflating the average while most speakers contribute
    very few samples. This is a classic sign of skewed distributions.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    all_regions = LATAM_REGIONS + ["Spain", "Not Provided"]
    bar_colors = COLORS_LATAM + ["#E9C46A", "#A8DADC"]

    means = []
    medians = []
    valid_regions = []
    valid_colors = []
    skew_ratios = []

    for i, region in enumerate(all_regions):
        counts = list(speaker_counts_by_region[region].values())
        if len(counts) < 2:
            continue
        arr = np.array(counts)
        m = np.mean(arr)
        md = np.median(arr)
        means.append(m)
        medians.append(md)
        valid_regions.append(region)
        valid_colors.append(bar_colors[i])
        skew_ratios.append(m / md if md > 0 else 0)

    x = np.arange(len(valid_regions))
    width = 0.35

    bars_mean = ax.bar(
        x - width / 2, means, width, label="Mean",
        color=[c for c in valid_colors], edgecolor="white", linewidth=1.5,
    )
    bars_median = ax.bar(
        x + width / 2, medians, width, label="Median",
        color=[c for c in valid_colors], edgecolor="black", linewidth=1.5,
        alpha=0.5,
    )

    for bar, val in zip(bars_mean, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", fontsize=10, fontweight="bold",
        )
    for bar, val in zip(bars_median, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", fontsize=10, fontweight="bold",
        )

    for i, ratio in enumerate(skew_ratios):
        color = "#D62828" if ratio > 5 else "#E9C46A" if ratio > 2 else "#2A9D8F"
        label = "SEVERE" if ratio > 5 else "MODERATE" if ratio > 2 else "OK"
        ax.text(
            x[i], max(means[i], medians[i]) + max(means) * 0.12,
            f"Skew: {ratio:.1f}x\n({label})",
            ha="center", fontsize=9, fontweight="bold", color=color,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(valid_regions, fontsize=10)
    ax.set_ylabel("Samples per Speaker", fontsize=14)
    ax.set_title(
        "Mean vs Median Samples per Speaker\n"
        "Large gap = few power users inflate the average (skewed distribution)",
        fontsize=16, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="upper right")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "12_mean_vs_median_per_region.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 12: Mean vs Median comparison DONE")


def graph_13_power_users_analysis(speaker_counts_by_region):
    """Graph 13: Power user concentration analysis per region.

    Shows the percentage of total samples contributed by the top 10%
    of speakers in each region. High concentration means a few speakers
    dominate the dataset, reducing effective speaker diversity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    all_regions = LATAM_REGIONS + ["Spain", "Not Provided"]
    bar_colors = COLORS_LATAM + ["#E9C46A", "#A8DADC"]

    top10_pcts = []
    top1_pcts = []
    valid_regions = []
    valid_colors = []
    top_speaker_counts = []
    total_speaker_counts = []

    for i, region in enumerate(all_regions):
        counts = sorted(speaker_counts_by_region[region].values(), reverse=True)
        if len(counts) < 10:
            continue

        total_smp = sum(counts)
        n_speakers = len(counts)

        top10_n = max(1, int(n_speakers * 0.10))
        top10_smp = sum(counts[:top10_n])
        top10_pct = top10_smp / total_smp * 100

        top1_n = max(1, int(n_speakers * 0.01))
        top1_smp = sum(counts[:top1_n])
        top1_pct = top1_smp / total_smp * 100

        top10_pcts.append(top10_pct)
        top1_pcts.append(top1_pct)
        valid_regions.append(region)
        valid_colors.append(bar_colors[i])
        top_speaker_counts.append(top10_n)
        total_speaker_counts.append(n_speakers)

    # Left: Top 10% concentration
    ax1 = axes[0]
    bars = ax1.barh(
        valid_regions, top10_pcts,
        color=valid_colors, edgecolor="white", linewidth=1.5,
    )
    for bar, pct, top_n, total_n in zip(
        bars, top10_pcts, top_speaker_counts, total_speaker_counts
    ):
        color = "#D62828" if pct > 70 else "#E9C46A" if pct > 50 else "#2A9D8F"
        ax1.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% ({top_n} of {total_n} speakers)",
            va="center", fontsize=10, fontweight="bold", color=color,
        )
    ax1.set_xlabel("% of Total Samples from Top 10% Speakers", fontsize=13)
    ax1.set_title("Top 10% Speaker Concentration", fontsize=16, fontweight="bold")
    ax1.axvline(x=50, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="50% threshold")
    ax1.set_xlim(0, 100)
    ax1.legend(fontsize=10)
    ax1.invert_yaxis()

    # Right: Top 1% concentration
    ax2 = axes[1]
    bars = ax2.barh(
        valid_regions, top1_pcts,
        color=valid_colors, edgecolor="white", linewidth=1.5,
    )
    for bar, pct in zip(bars, top1_pcts):
        color = "#D62828" if pct > 30 else "#E9C46A" if pct > 15 else "#2A9D8F"
        ax2.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=11, fontweight="bold", color=color,
        )
    ax2.set_xlabel("% of Total Samples from Top 1% Speakers", fontsize=13)
    ax2.set_title("Top 1% Speaker Concentration", fontsize=16, fontweight="bold")
    ax2.axvline(x=20, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="20% threshold")
    ax2.set_xlim(0, max(top1_pcts) * 1.3 if top1_pcts else 100)
    ax2.legend(fontsize=10)
    ax2.invert_yaxis()

    fig.suptitle(
        "Power User Concentration Analysis\n"
        "High % = few speakers dominate the dataset, reducing effective diversity",
        fontsize=18, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "13_power_users_concentration.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()
    print("Graph 13: Power users concentration DONE")


def main():
    """Generate all 13 presentation graphs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apply_style()

    print("Parsing validated.tsv...")
    region_speakers, region_samples, total, speaker_counts_by_region = extract_data()
    print(f"Total samples: {total:,}")
    print()

    graph_01_macro_pie(region_samples, total)
    graph_02_latam_pie(region_samples)
    graph_03_speakers_bar(region_speakers)
    graph_04_samples_bar(region_samples)
    graph_05_ratio_bar(region_speakers, region_samples)
    graph_06_habla_vs_cv(region_samples, total)
    graph_07_imbalance_log(region_speakers)
    graph_08_andino_focus(region_samples)
    graph_09_speaker_diversity(region_speakers)
    graph_10_undeclared(region_speakers)
    graph_11_boxplot_samples_per_speaker(speaker_counts_by_region)
    graph_12_mean_median_comparison(speaker_counts_by_region)
    graph_13_power_users_analysis(speaker_counts_by_region)

    print()
    print("=" * 60)
    print(f"ALL 13 GRAPHS SAVED to: {OUTPUT_DIR}")
    print("=" * 60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
