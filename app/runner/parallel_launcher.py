"""
Parallel launcher for the HABLA-Spoof 12-job sweep on a single GPU.

Each attack pipeline lives in its own isolated venv and consumes ~6-8 GB
of VRAM. On a 46 GB A40 we can comfortably keep four pipelines resident
at once; five fits but leaves only ~6 GB headroom for runtime spikes.

The launcher spawns one bash subprocess per pending (attack, partition)
job. Each child sources its venv, cds into the repo, and runs
`python -m app.runner.partial_spoof_orchestrator --mode single ...`.
All children inherit `CUDA_VISIBLE_DEVICES=<gpu>` from the launcher,
which is set explicitly so a stale value in the parent shell does not
leak.

Reap-and-relaunch policy: whenever any child exits, the next pending
job is dispatched into the free slot. This keeps the slot pool
saturated throughout, which matters because Chatterbox and OuteTTS
take days while OmniVoice and Qwen finish in hours.

Children stream their output to `logs/parallel_<attack>_<partition>.log`
so the launcher's own stdout stays clean and tailable.
"""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class ParallelLauncher:
    """Run up to N per-attack pipeline jobs concurrently on one GPU.

    Designed for ml-server03's two-machine workflow: invoke once,
    monitor via the launcher's tail, and rely on the per-cell checkpoint
    files for crash safety. Killing the launcher with SIGINT propagates
    SIGTERM to all running children, which Python's loguru-based step
    code handles by flushing the in-flight checkpoint before exit.

    Attributes:
        repo_root: Absolute path to the project repository.
        envs_root: Absolute path to the parent of the per-attack venvs.
        log_dir: Directory for per-job stdout/stderr capture.
        attack_venv_map: Mapping from attack identifier to venv folder.
        orchestrator_module: Python -m target for each child.
        poll_interval_seconds: Sleep between reap-and-relaunch passes.
    """

    DEFAULT_REPO_ROOT = Path("/home") / os.environ.get("USER", "tomas") / "ANTI-SPOOFING-VOICE-LATIN-AMERICA"
    DEFAULT_ENVS_DIR_NAME = "envs"
    DEFAULT_LOG_DIR_NAME = "logs"
    DEFAULT_POLL_SECONDS = 5
    ORCHESTRATOR_MODULE = "app.runner.partial_spoof_orchestrator"

    def __init__(
        self,
        attack_venv_map: Dict[str, str],
        repo_root: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        poll_interval_seconds: int = DEFAULT_POLL_SECONDS,
    ) -> None:
        """Initialise the launcher.

        Args:
            attack_venv_map: Mapping from attack name (e.g. 'qwen') to
                venv directory name relative to envs_root
                (e.g. 'qwen_env').
            repo_root: Project root directory. Defaults to
                ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA on ml-server03.
            log_dir: Directory for per-job logs. Defaults to
                {repo_root}/logs.
            poll_interval_seconds: How often to reap completed children
                and check for new dispatch slots.
        """
        self.attack_venv_map = attack_venv_map
        self.repo_root = repo_root or self._resolve_repo_root()
        self.envs_root = self.repo_root / self.DEFAULT_ENVS_DIR_NAME
        self.log_dir = log_dir or (self.repo_root / self.DEFAULT_LOG_DIR_NAME)
        self.poll_interval_seconds = poll_interval_seconds
        self._running: Dict[subprocess.Popen, Tuple[str, str, Path]] = {}
        self._completed: List[Tuple[str, str, int, Path]] = []
        self._shutdown_requested = False

    def launch(
        self,
        jobs: List[Tuple[str, str]],
        gpu: int,
        max_concurrent: int,
    ) -> List[Tuple[str, str, int, Path]]:
        """Dispatch all jobs with at most max_concurrent running at once.

        Blocks until every job has terminated. Reports per-job exit code
        and log path. SIGINT in the launcher is caught and propagated as
        SIGTERM to all running children so their checkpoint writers can
        flush before exit.

        Args:
            jobs: Ordered list of (attack, partition) pairs. The order
                determines dispatch priority when multiple slots free
                simultaneously.
            gpu: GPU index passed to children via CUDA_VISIBLE_DEVICES.
            max_concurrent: Maximum number of children to keep running
                simultaneously. Tuned to the GPU's VRAM budget.

        Returns:
            List of (attack, partition, exit_code, log_path) tuples in
            completion order.

        Raises:
            ValueError: If jobs is empty or max_concurrent < 1.
            FileNotFoundError: If a venv directory does not exist.
        """
        if not jobs:
            raise ValueError("jobs list is empty")
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

        self._validate_venvs(jobs)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._install_signal_handlers()

        pending = list(jobs)
        total = len(pending)
        logger.info("=" * 80)
        logger.info(
            f"PARALLEL LAUNCHER: {total} jobs, max {max_concurrent} concurrent, GPU {gpu}"
        )
        logger.info(f"Repo root         : {self.repo_root}")
        logger.info(f"Log directory     : {self.log_dir}")
        logger.info("=" * 80)

        try:
            while pending or self._running:
                while (
                    pending
                    and len(self._running) < max_concurrent
                    and not self._shutdown_requested
                ):
                    attack, partition = pending.pop(0)
                    self._spawn(attack, partition, gpu)

                completed_this_pass = self._reap_completed()
                if completed_this_pass:
                    self._log_progress(total)

                if self._shutdown_requested and not self._running:
                    logger.warning(
                        "Shutdown complete after SIGINT; pending jobs not dispatched."
                    )
                    break

                if self._running:
                    time.sleep(self.poll_interval_seconds)
        except Exception:
            logger.exception("Parallel launcher aborted; sending SIGTERM to children.")
            self._terminate_all()
            raise

        logger.info("=" * 80)
        logger.info(
            f"PARALLEL LAUNCHER DONE: {len(self._completed)}/{total} jobs completed"
        )
        for attack, partition, exit_code, log_path in self._completed:
            status = "OK" if exit_code == 0 else f"FAIL({exit_code})"
            logger.info(
                f"  {attack:<12} {partition:<14} {status:<10} log={log_path}"
            )
        logger.info("=" * 80)
        return list(self._completed)

    def _spawn(self, attack: str, partition: str, gpu: int) -> None:
        """Start one subprocess for an (attack, partition) job.

        Args:
            attack: Attack identifier (e.g. 'qwen').
            partition: Partition identifier ('not_jittered' or 'jittered').
            gpu: GPU index for CUDA_VISIBLE_DEVICES.
        """
        venv_name = self.attack_venv_map[attack]
        venv_activate = self.envs_root / venv_name / "bin" / "activate"
        log_path = self.log_dir / f"parallel_{attack}_{partition}.log"
        timestamp = datetime.now(timezone.utc).isoformat()
        command = (
            f"source {venv_activate} && "
            f"cd {self.repo_root} && "
            f"python -m {self.ORCHESTRATOR_MODULE} "
            f"--mode single --attack {attack} --partition {partition}"
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_handle = open(log_path, "a", encoding="utf-8")
        log_handle.write(
            f"\n=== Job start {timestamp}: {attack}/{partition} on GPU {gpu} ===\n"
        )
        log_handle.flush()

        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(self.repo_root),
            preexec_fn=os.setsid,
        )
        self._running[proc] = (attack, partition, log_path)
        logger.info(
            f"  [DISPATCH] {attack:<12} {partition:<14} pid={proc.pid:<6} log={log_path}"
        )

    def _reap_completed(self) -> List[Tuple[str, str, int]]:
        """Move finished children from _running to _completed.

        Returns:
            List of (attack, partition, exit_code) tuples reaped this pass.
        """
        reaped: List[Tuple[str, str, int]] = []
        for proc in list(self._running):
            return_code = proc.poll()
            if return_code is None:
                continue
            attack, partition, log_path = self._running.pop(proc)
            self._completed.append((attack, partition, return_code, log_path))
            status = "OK" if return_code == 0 else f"FAIL({return_code})"
            logger.info(
                f"  [REAP] {attack:<12} {partition:<14} {status:<10} pid={proc.pid}"
            )
            reaped.append((attack, partition, return_code))
        return reaped

    def _log_progress(self, total: int) -> None:
        """Emit a one-line progress snapshot.

        Args:
            total: Total number of jobs that will eventually run.
        """
        running = ", ".join(
            f"{a}/{p}" for a, p, _ in self._running.values()
        )
        logger.info(
            f"  Progress: {len(self._completed)}/{total} complete | "
            f"{len(self._running)} running: [{running}]"
        )

    def _install_signal_handlers(self) -> None:
        """Wire SIGINT/SIGTERM to a graceful child-shutdown routine."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, _frame) -> None:
        """Mark shutdown and propagate signal to running children.

        Args:
            signum: The received signal number.
            _frame: Unused signal frame.
        """
        if self._shutdown_requested:
            logger.warning(
                f"Signal {signum} received again; force-killing children."
            )
            self._terminate_all(force=True)
            sys.exit(1)
        self._shutdown_requested = True
        logger.warning(
            f"Signal {signum} received; requesting child shutdown. "
            "Send the signal again to force-kill."
        )
        self._terminate_all(force=False)

    def _terminate_all(self, force: bool = False) -> None:
        """Send SIGTERM (or SIGKILL when force=True) to every running child.

        Args:
            force: If True, send SIGKILL instead of SIGTERM.
        """
        sig = signal.SIGKILL if force else signal.SIGTERM
        for proc in list(self._running):
            attack, partition, _ = self._running[proc]
            try:
                os.killpg(os.getpgid(proc.pid), sig)
                logger.warning(
                    f"  Sent {'SIGKILL' if force else 'SIGTERM'} to "
                    f"{attack}/{partition} (pgid={os.getpgid(proc.pid)})"
                )
            except ProcessLookupError:
                continue

    def _validate_venvs(self, jobs: List[Tuple[str, str]]) -> None:
        """Verify every required venv exists before dispatching anything.

        Args:
            jobs: The full job list to be launched.

        Raises:
            ValueError: If an attack has no venv mapping.
            FileNotFoundError: If the mapped venv directory is missing.
        """
        for attack, partition in jobs:
            if attack not in self.attack_venv_map:
                raise ValueError(
                    f"No venv mapping for attack '{attack}'. "
                    f"Known attacks: {sorted(self.attack_venv_map)}"
                )
            venv_dir = self.envs_root / self.attack_venv_map[attack]
            activate = venv_dir / "bin" / "activate"
            if not activate.exists():
                raise FileNotFoundError(
                    f"Venv activate script missing for {attack}/{partition}: "
                    f"{activate}. Check that the env is created at "
                    f"{venv_dir}."
                )

    def _resolve_repo_root(self) -> Path:
        """Resolve the repository root directory.

        Returns:
            Path to the repo root. Uses DEFAULT_REPO_ROOT if it exists,
            otherwise falls back to the current working directory.
        """
        if self.DEFAULT_REPO_ROOT.exists():
            return self.DEFAULT_REPO_ROOT
        return Path.cwd()
