"""
Partial spoof corpus orchestrator.

Coordinates the 12-job HABLA-Spoof production run (6 attacks x 2
partitions) and aggregates the per-pipeline CSV exports into the master
corpus tables. Designed for the two-machine workflow: each job runs as
its own ml-server03 process inside the matching venv with the matching
GPU pin, while this orchestrator owns dispatch tracking, aggregation,
and reporting.

Modes:
    single   - Run one (attack, partition) job inside the current process.
    aggregate- Concatenate per-pipeline samples.csv / spoofed_words.csv
               into corpus_samples.csv / corpus_spoofed_words.csv.
    runbook  - Print the 12 ml-server03 shell commands needed to
               execute the full production sweep.
    status   - Summarise progress per (attack, partition) by reading
               checkpoints + samples.csv presence.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from app.pipeline.partial_spoof.pipeline_facade import PartialSpoofPipeline
from app.pipeline.partial_spoof.schemas.pipeline_config import (
    PartialSpoofPipelineConfig,
)
from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.utils.checkpoint_manager import CheckpointManager
from app.runner.parallel_launcher import ParallelLauncher


PARTITIONS: Tuple[str, str] = ("not_jittered", "jittered")
CORPUS_ROOT = Path("data/partial_spoof_output")
CORPUS_SAMPLES_CSV = CORPUS_ROOT / "corpus_samples.csv"
CORPUS_SPOOFED_WORDS_CSV = CORPUS_ROOT / "corpus_spoofed_words.csv"
CORPUS_SUMMARY_JSON = CORPUS_ROOT / "corpus_summary.json"
ATTACK_VENV_HINTS: Dict[str, str] = {
    "fishgram": "fishgram_env",
    "qwen": "qwen_env",
    "openvoice": "openvoice_env",
    "outetts": "outetts_env",
    "chatterbox": "chatterbox_env",
    "omnivoice": "omnivoice_env",
}


class PartialSpoofOrchestrator:
    """Top-level driver for the 12-job HABLA-Spoof production sweep.

    The orchestrator deliberately does NOT launch subprocesses across
    venvs from inside the active interpreter, because each attack has
    its own isolated Python environment on ml-server03. Instead it
    exposes:
      - A single-job runner that drives the facade in the current
        process (used inside each per-attack venv).
      - A pure aggregation pass that concatenates per-pipeline CSV
        outputs into corpus tables (no GPU, runs in any env).
      - A runbook printer that emits the 12 shell commands so Master
        Tomas can launch them across GPUs / venvs manually.
      - A status report that reads checkpoint state plus samples.csv
        presence to summarise progress without re-running anything.

    Attributes:
        attack_weights: Target weight distribution copied from settings.
        attacks: Ordered list of attack identifiers (defines runbook order).
        corpus_root: Root directory for per-pipeline outputs.
    """

    def __init__(self) -> None:
        """Initialise the orchestrator from the partial spoof settings."""
        self.attack_weights = dict(settings.ATTACK_WEIGHTS)
        self.attacks = list(self.attack_weights.keys())
        self.corpus_root = CORPUS_ROOT

    def run_single(
        self,
        attack: str,
        partition: str,
        device_override: Optional[str] = None,
    ) -> None:
        """Run one (attack, partition) job using the manifest-driven facade.

        Args:
            attack: Attack identifier (must appear in settings.ATTACK_WEIGHTS).
            partition: 'not_jittered' or 'jittered'.
            device_override: Optional device string (e.g. 'cuda:1').

        Raises:
            ValueError: If attack or partition is unknown.
        """
        if attack not in self.attacks:
            raise ValueError(
                f"Unknown attack '{attack}'; expected one of {self.attacks}."
            )
        if partition not in PARTITIONS:
            raise ValueError(
                f"Unknown partition '{partition}'; expected one of {PARTITIONS}."
            )

        logger.info("=" * 80)
        logger.info(f"ORCHESTRATOR JOB: {attack} / {partition}")
        logger.info("=" * 80)

        enable_jitter = partition == "jittered"
        config = PartialSpoofPipelineConfig(
            attack_system=attack,
            use_manifest=True,
            bonafide_file_partition_override=partition,
            manifest_slice_attack_override=attack,
            manifest_slice_partition_override=partition,
            enable_boundary_jitter_override=enable_jitter,
            device_override=device_override,
        )
        pipeline = PartialSpoofPipeline(config=config)
        pipeline.run()

    def aggregate(self) -> None:
        """Concatenate per-pipeline CSVs into the corpus master CSVs.

        Reads every (attack, partition) cell that has samples.csv and
        spoofed_words.csv, concatenates them in a deterministic order,
        and writes the corpus tables plus a summary JSON for the paper.

        Per-pipeline CSVs are NOT modified or deleted; the corpus tables
        are pure derivatives and can be regenerated at will.
        """
        logger.info("=" * 80)
        logger.info("ORCHESTRATOR AGGREGATE: building corpus CSVs")
        logger.info("=" * 80)

        samples_paths: List[Tuple[str, str, Path]] = []
        words_paths: List[Tuple[str, str, Path]] = []
        for attack in self.attacks:
            for partition in PARTITIONS:
                run_dir = self.corpus_root / attack / partition
                samples_csv = run_dir / "samples.csv"
                words_csv = run_dir / "spoofed_words.csv"
                if samples_csv.exists():
                    samples_paths.append((attack, partition, samples_csv))
                if words_csv.exists():
                    words_paths.append((attack, partition, words_csv))

        self.corpus_root.mkdir(parents=True, exist_ok=True)
        samples_count = self._concatenate_csv(
            sources=samples_paths,
            destination=CORPUS_SAMPLES_CSV,
        )
        words_count = self._concatenate_csv(
            sources=words_paths,
            destination=CORPUS_SPOOFED_WORDS_CSV,
        )
        self._emit_corpus_summary(
            samples_count=samples_count,
            words_count=words_count,
            samples_paths=samples_paths,
        )

        logger.info("-" * 60)
        logger.info(f"corpus_samples.csv        : {samples_count} rows")
        logger.info(f"corpus_spoofed_words.csv  : {words_count} rows")
        logger.info(f"corpus_summary.json       : {CORPUS_SUMMARY_JSON}")
        logger.info("=" * 80)

    def print_runbook(self, gpu_default: int = 1) -> None:
        """Emit the 12 ml-server03 shell commands for the full sweep.

        Args:
            gpu_default: Default GPU index to embed in the commands.
                Master Tomas can override per-attack as needed.
        """
        logger.info("=" * 80)
        logger.info("HABLA-SPOOF PRODUCTION SWEEP RUNBOOK")
        logger.info("=" * 80)
        logger.info(
            f"  Generate the manifest first (once, any venv with Parakeet):"
        )
        logger.info(
            f"    python -m app.scripts.generate_partial_spoof_manifest"
        )
        logger.info("")
        for attack in self.attacks:
            for partition in PARTITIONS:
                venv = ATTACK_VENV_HINTS.get(attack, "<env>")
                logger.info(f"# {attack} / {partition}")
                logger.info(
                    f"export CUDA_VISIBLE_DEVICES={gpu_default} && "
                    f"source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/{venv}/bin/activate && "
                    f"python -m app.runner.partial_spoof_orchestrator "
                    f"--mode single --attack {attack} --partition {partition} "
                    f"&& deactivate"
                )
                logger.info("")
        logger.info("# After all 12 jobs complete:")
        logger.info(
            "python -m app.runner.partial_spoof_orchestrator --mode aggregate"
        )
        logger.info("=" * 80)

    def run_parallel(
        self,
        gpu: int,
        max_concurrent: int,
        skip_complete: bool = True,
        order: str = "weighted",
    ) -> None:
        """Dispatch the full 12-job sweep with bounded concurrency on one GPU.

        Each (attack, partition) cell runs in its own subprocess under
        its own venv. The launcher caps concurrency to max_concurrent
        and reaps-and-relaunches as jobs complete. Already-complete
        cells (samples.csv present) are skipped when skip_complete is
        True so the launcher is safe to re-invoke after a partial run.

        Args:
            gpu: GPU index to share across all children.
            max_concurrent: Maximum number of concurrent children.
                Recommended 4 on a 46 GB A40 (each pipeline ~8 GB peak).
            skip_complete: When True, skip cells whose samples.csv
                already exists. False forces dispatch of everything.
            order: Job ordering policy. 'weighted' interleaves by
                attack weight (largest first), 'alphabetic' uses
                attack-then-partition lexical order, 'slow_first'
                front-loads Chatterbox / OuteTTS so the fast attacks
                can fill the remaining slots.
        """
        jobs = self._build_job_list(skip_complete=skip_complete, order=order)
        if not jobs:
            logger.info(
                "No pending jobs (all cells already have samples.csv). "
                "Use --no-skip-complete to force redispatch."
            )
            return

        launcher = ParallelLauncher(
            attack_venv_map=dict(ATTACK_VENV_HINTS),
        )
        launcher.launch(
            jobs=jobs,
            gpu=gpu,
            max_concurrent=max_concurrent,
        )

    def _build_job_list(
        self,
        skip_complete: bool,
        order: str,
    ) -> List[Tuple[str, str]]:
        """Build the (attack, partition) job list per order policy.

        Args:
            skip_complete: When True, drop cells whose samples.csv exists.
            order: 'weighted', 'alphabetic', or 'slow_first'.

        Returns:
            Ordered list of (attack, partition) tuples to dispatch.
        """
        candidate_jobs: List[Tuple[str, str]] = []
        if order == "weighted":
            attacks_ordered = sorted(
                self.attacks,
                key=lambda a: self.attack_weights[a],
                reverse=True,
            )
        elif order == "slow_first":
            slow_set = {"chatterbox", "outetts"}
            slow = [a for a in self.attacks if a in slow_set]
            fast = [a for a in self.attacks if a not in slow_set]
            attacks_ordered = slow + sorted(
                fast, key=lambda a: self.attack_weights[a], reverse=True,
            )
        else:
            attacks_ordered = sorted(self.attacks)

        for attack in attacks_ordered:
            for partition in PARTITIONS:
                candidate_jobs.append((attack, partition))

        if not skip_complete:
            return candidate_jobs

        pending: List[Tuple[str, str]] = []
        for attack, partition in candidate_jobs:
            samples_csv = self.corpus_root / attack / partition / "samples.csv"
            if samples_csv.exists():
                logger.info(
                    f"  [SKIP] {attack}/{partition}: samples.csv already present"
                )
                continue
            pending.append((attack, partition))
        return pending

    def print_status(self) -> None:
        """Report progress per (attack, partition) cell.

        Reads .checkpoint.json plus samples.csv presence in every cell.
        Does not load any models or touch GPU.
        """
        logger.info("=" * 80)
        logger.info("HABLA-SPOOF PRODUCTION STATUS")
        logger.info("=" * 80)
        for attack in self.attacks:
            for partition in PARTITIONS:
                run_dir = self.corpus_root / attack / partition
                checkpoint_path = run_dir / CheckpointManager.CHECKPOINT_FILENAME
                samples_csv = run_dir / "samples.csv"
                state = self._summarise_cell(run_dir, checkpoint_path, samples_csv)
                logger.info(
                    f"  {attack:<12} {partition:<14} "
                    f"cloned={state['cloned']:<6} "
                    f"spliced={state['spliced']:<6} "
                    f"samples_csv={state['samples_csv']:<6} "
                    f"failed={state['failed']:<5} "
                    f"{state['summary']}"
                )
        logger.info("=" * 80)

    def _concatenate_csv(
        self,
        sources: List[Tuple[str, str, Path]],
        destination: Path,
    ) -> int:
        """Concatenate per-pipeline CSVs in deterministic order.

        Reads the header from the first non-empty source as the canonical
        schema. Subsequent rows are appended; rows with missing schema
        columns get empty values. If a source has columns NOT in the
        canonical header those extras are dropped with a warning.

        Args:
            sources: List of (attack, partition, csv_path) tuples.
            destination: Output corpus CSV path.

        Returns:
            Number of data rows written to destination.
        """
        if not sources:
            logger.warning(
                f"No source CSVs found for {destination.name}; "
                "writing an empty file with no header."
            )
            destination.write_text("", encoding="utf-8")
            return 0

        canonical_header: List[str] = []
        for _, _, path in sources:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header:
                    canonical_header = header
                    break

        if not canonical_header:
            logger.warning(
                f"All source CSVs for {destination.name} are empty; "
                "writing an empty file with no header."
            )
            destination.write_text("", encoding="utf-8")
            return 0

        total_rows = 0
        with open(destination, "w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(
                out_handle,
                fieldnames=canonical_header,
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for attack, partition, path in sources:
                with open(path, "r", encoding="utf-8", newline="") as in_handle:
                    reader = csv.DictReader(in_handle)
                    for row in reader:
                        filtered = {
                            field: row.get(field, "") for field in canonical_header
                        }
                        writer.writerow(filtered)
                        total_rows += 1
                logger.debug(
                    f"  + {attack}/{partition}: rows so far={total_rows}"
                )
        return total_rows

    def _emit_corpus_summary(
        self,
        samples_count: int,
        words_count: int,
        samples_paths: List[Tuple[str, str, Path]],
    ) -> None:
        """Write a small summary JSON with marginals for the paper.

        Args:
            samples_count: Total rows in corpus_samples.csv.
            words_count: Total rows in corpus_spoofed_words.csv.
            samples_paths: Source paths that were aggregated.
        """
        per_cell: Dict[str, Dict[str, int]] = {a: {p: 0 for p in PARTITIONS} for a in self.attacks}
        per_attack_total = {a: 0 for a in self.attacks}
        per_partition_total = {p: 0 for p in PARTITIONS}
        per_quality = {"high": 0, "medium": 0, "low": 0}

        for attack, partition, path in samples_paths:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    per_cell[attack][partition] += 1
                    per_attack_total[attack] += 1
                    per_partition_total[partition] += 1
                    flag = row.get("quality_flag", "")
                    if flag in per_quality:
                        per_quality[flag] += 1

        actual_weights: Dict[str, float] = {}
        if samples_count > 0:
            for attack, count in per_attack_total.items():
                actual_weights[attack] = count / samples_count

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "samples_total": samples_count,
            "spoofed_words_total": words_count,
            "per_attack_total": per_attack_total,
            "per_partition_total": per_partition_total,
            "per_cell": per_cell,
            "per_quality_flag": per_quality,
            "attack_weights_target": self.attack_weights,
            "attack_weights_actual": actual_weights,
        }
        with open(CORPUS_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _summarise_cell(
        self,
        run_dir: Path,
        checkpoint_path: Path,
        samples_csv: Path,
    ) -> Dict:
        """Build a status row dict for one (attack, partition) cell.

        Args:
            run_dir: The per-pipeline output directory.
            checkpoint_path: Path to .checkpoint.json.
            samples_csv: Path to samples.csv.

        Returns:
            Dict with cloned, spliced, failed, samples_csv row count
            (or 0 if missing), and a short text summary.
        """
        cloned = spliced = failed = 0
        samples_csv_rows = 0
        summary = "missing"
        if not run_dir.exists():
            return {
                "cloned": cloned,
                "spliced": spliced,
                "failed": failed,
                "samples_csv": samples_csv_rows,
                "summary": summary,
            }

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                cloned = len(payload.get("cloned", []))
                spliced = len(payload.get("spliced", []))
                failed = len(payload.get("failed_generation", {}))
            except (OSError, json.JSONDecodeError):
                summary = "checkpoint_unreadable"

        if samples_csv.exists():
            with open(samples_csv, "r", encoding="utf-8", newline="") as f:
                samples_csv_rows = sum(1 for _ in f) - 1
                samples_csv_rows = max(samples_csv_rows, 0)
            summary = "complete" if samples_csv_rows > 0 else "samples_csv_empty"
        else:
            summary = "in_progress" if cloned > 0 or spliced > 0 else "not_started"

        return {
            "cloned": cloned,
            "spliced": spliced,
            "failed": failed,
            "samples_csv": samples_csv_rows,
            "summary": summary,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for the orchestrator entry point.

    Returns:
        argparse.ArgumentParser ready to parse sys.argv[1:].
    """
    parser = argparse.ArgumentParser(
        description="HABLA-Spoof production orchestrator",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "aggregate", "runbook", "status", "parallel"),
        default="status",
        help="Operating mode (default: status).",
    )
    parser.add_argument("--attack", help="Attack identifier (single mode).")
    parser.add_argument(
        "--partition",
        choices=PARTITIONS,
        help="Partition identifier (single mode).",
    )
    parser.add_argument(
        "--device",
        help="Optional device override, e.g. 'cuda:1' (single mode).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU index for runbook commands or parallel children (default: 1).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent pipelines for --mode parallel "
             "(default: 4; safe ceiling on a 46 GB A40 with ~8 GB per pipeline).",
    )
    parser.add_argument(
        "--order",
        choices=("weighted", "alphabetic", "slow_first"),
        default="weighted",
        help="Job ordering policy for --mode parallel (default: weighted).",
    )
    parser.add_argument(
        "--no-skip-complete",
        action="store_true",
        help="In --mode parallel, dispatch every (attack, partition) cell "
             "even if samples.csv already exists. Default skips completed cells.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    orchestrator = PartialSpoofOrchestrator()

    if args.mode == "single":
        if not args.attack or not args.partition:
            parser.error("--attack and --partition are required for --mode single")
        orchestrator.run_single(
            attack=args.attack,
            partition=args.partition,
            device_override=args.device,
        )
    elif args.mode == "aggregate":
        orchestrator.aggregate()
    elif args.mode == "runbook":
        orchestrator.print_runbook(gpu_default=args.gpu)
    elif args.mode == "parallel":
        orchestrator.run_parallel(
            gpu=args.gpu,
            max_concurrent=args.max_concurrent,
            skip_complete=not args.no_skip_complete,
            order=args.order,
        )
    else:
        orchestrator.print_status()
