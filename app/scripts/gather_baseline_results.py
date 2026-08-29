"""
Gather every training-run result into one paper-ready summary.

WHY
----
Each `run_detector_training` invocation writes its own `config.json` and
`result.json` under `data/training_runs/<run-name>/`, one run at a time. For a
paper table -- e.g. "EER of previously published detectors evaluated
zero-shot on MARSA" -- those numbers need collecting across every run,
labelled with which model and which checkpoint each row came from, since a
bare backend name like `lcnn` is ambiguous between the original and
fine-tuned LSTM-sum weights.

This script reads every run directory once, joins its config against its
result, and prints a compact markdown table plus (optionally) a full JSON and
CSV dump. The markdown table is meant to be pasted directly into a chat
message; the JSON keeps every per-attack figure and low-sample-count flag for
when a deeper table is needed later.

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate

    # Print the summary table
    python -m app.scripts.gather_baseline_results

    # Also write the full detail and a flat CSV
    python -m app.scripts.gather_baseline_results \\
        --output-json data/baseline_results.json \\
        --output-csv data/baseline_results.csv

    deactivate
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.schemas.baseline_result_row import BaselineResultRow


def _load_run(run_dir: Path) -> List[BaselineResultRow]:
    """Join one run's config and result into typed rows, one per split.

    Args:
        run_dir: A single ``data/training_runs/<run-name>/`` directory.

    Returns:
        One row per evaluated split. Empty if the run has no result yet
        (still in progress, or failed before writing one).
    """
    config_path = run_dir / "config.json"
    result_path = run_dir / "result.json"
    if not config_path.exists() or not result_path.exists():
        logger.warning(f"Skipping incomplete run: {run_dir.name}")
        return []

    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))

    rows: List[BaselineResultRow] = []
    for evaluation in result.get("evaluations", []):
        rows.append(
            BaselineResultRow(
                run_name=result["run_name"],
                detector_backend=config["detector_backend"],
                checkpoint=config.get("model_id"),
                eval_only=config.get("eval_only", False),
                split=evaluation["split"],
                clip_count=evaluation["clip_count"],
                eer=evaluation["eer"],
                strict_clip_count=evaluation.get("strict_clip_count", 0),
                strict_eer=evaluation.get("strict_eer", -1.0),
                per_attack_eer=evaluation.get("per_attack_eer", {}),
                per_attack_clips=evaluation.get("per_attack_clips", {}),
                low_confidence_attacks=evaluation.get("low_confidence_attacks", []),
            )
        )
    return rows


def _short_checkpoint(row: BaselineResultRow) -> str:
    """Shorten a checkpoint identifier for display in a narrow table column.

    Args:
        row: The row to summarise.

    Returns:
        The Hugging Face repo id unchanged, or the local .pt file's basename.
    """
    if not row.checkpoint:
        return "(published default)"
    if "/" in row.checkpoint and not row.checkpoint.endswith(".pt"):
        return row.checkpoint
    return Path(row.checkpoint).name


def _print_markdown_table(rows: List[BaselineResultRow]) -> None:
    """Print the compact, paste-ready summary table.

    Sorted by eval-split pooled EER ascending, so the best model leads.

    Args:
        rows: All gathered rows, every split.
    """
    by_run: dict = {}
    for row in rows:
        by_run.setdefault(row.run_name, {})[row.split] = row

    def sort_key(item):
        splits = item[1]
        eval_row = splits.get("eval")
        return eval_row.eer if eval_row else 999.0

    ordered = sorted(by_run.items(), key=sort_key)

    print()
    print("| Run | Backend | Checkpoint | Mode | Dev EER | Dev Strict | Eval EER | Eval Strict |")
    print("|---|---|---|---|---|---|---|---|")
    for run_name, splits in ordered:
        any_row = next(iter(splits.values()))
        backend = any_row.detector_backend
        checkpoint = _short_checkpoint(any_row)
        mode = "zero-shot" if any_row.eval_only else "fine-tuned"

        dev = splits.get("dev")
        eval_ = splits.get("eval")
        dev_eer = f"{dev.eer:.3f}%" if dev else "--"
        dev_strict = f"{dev.strict_eer:.3f}%" if dev and dev.strict_eer >= 0 else "--"
        eval_eer = f"{eval_.eer:.3f}%" if eval_ else "--"
        eval_strict = f"{eval_.strict_eer:.3f}%" if eval_ and eval_.strict_eer >= 0 else "--"

        print(
            f"| {run_name} | {backend} | {checkpoint} | {mode} | "
            f"{dev_eer} | {dev_strict} | {eval_eer} | {eval_strict} |"
        )
    print()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Gather every training-run result into a paper-ready summary."
    )
    parser.add_argument(
        "--runs-root", type=Path, default=Path("data/training_runs"),
        help="Parent directory of run subdirectories.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Optional path to write the full detail, including per-attack figures.",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Optional path to write a flat CSV, one row per model per split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.runs_root.is_dir():
        logger.error(f"No such directory: {args.runs_root}")
        raise SystemExit(1)

    all_rows: List[BaselineResultRow] = []
    for run_dir in sorted(args.runs_root.iterdir()):
        if run_dir.is_dir():
            all_rows.extend(_load_run(run_dir))

    if not all_rows:
        logger.error(f"No completed runs found under {args.runs_root}")
        raise SystemExit(1)

    logger.info(f"Gathered {len(all_rows)} split-results from {args.runs_root}")
    _print_markdown_table(all_rows)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([row.model_dump() for row in all_rows], indent=2),
            encoding="utf-8",
        )
        logger.info(f"Full detail written: {args.output_json}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "run_name", "detector_backend", "checkpoint", "eval_only",
                    "split", "clip_count", "eer", "strict_clip_count", "strict_eer",
                ]
            )
            for row in all_rows:
                writer.writerow(
                    [
                        row.run_name, row.detector_backend, row.checkpoint,
                        row.eval_only, row.split, row.clip_count, row.eer,
                        row.strict_clip_count, row.strict_eer,
                    ]
                )
        logger.info(f"Flat CSV written: {args.output_csv}")
