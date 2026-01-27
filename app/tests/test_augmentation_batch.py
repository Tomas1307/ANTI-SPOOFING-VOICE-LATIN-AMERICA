"""
Tests for run_augmentation_batch.py

Validates:
1. Factor validation (valid and invalid formats)
2. Argument parsing defaults
3. Batch run_batch function with mocked pipeline
4. Log files are created per factor
5. Failure in one factor doesn't stop subsequent runs
"""

import os
import sys
import shutil
import tempfile
from unittest import mock
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Test factor validation
# ---------------------------------------------------------------------------

def test_factor_format_validation():
    """Valid and invalid factor strings are handled correctly."""
    valid = ["2x", "3x", "5x", "10x", "15x"]
    invalid = ["2", "x3", "abc", "3.5x", ""]

    for f in valid:
        assert f.endswith("x"), f"Expected valid: {f}"
        int(f[:-1])  # should not raise

    for f in invalid:
        ok = False
        try:
            assert f.endswith("x")
            int(f[:-1])
        except (AssertionError, ValueError, AssertionError):
            ok = True
        assert ok, f"Expected invalid: {f}"

    print("  [OK] Factor format validation")


# ---------------------------------------------------------------------------
# 2. Test argument defaults
# ---------------------------------------------------------------------------

def test_default_factors():
    """DEFAULT_FACTORS is exactly [2x, 3x, 5x, 10x]."""
    from app.scripts.run_augmentation_batch import DEFAULT_FACTORS
    assert DEFAULT_FACTORS == ["2x", "3x", "5x", "10x"]
    print("  [OK] Default factors are correct")


# ---------------------------------------------------------------------------
# 3. Test batch runner with mocked pipeline
# ---------------------------------------------------------------------------

def test_run_batch_mocked():
    """
    Mocks AugmentationPipeline so no real data is needed.
    Verifies that run_batch calls the pipeline for each factor
    and creates log files.
    """
    from app.scripts.run_augmentation_batch import run_batch

    tmp_dir = tempfile.mkdtemp(prefix="test_batch_")

    try:
        call_log = []

        class FakePipeline:
            def __init__(self, **kwargs):
                self.min_factor = kwargs["min_factor"]
                self.output_dir = Path(tmp_dir) / f"augmented_{self.min_factor}_balanced_5050"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.log_path = self.output_dir / f"{self.min_factor}.txt"

                import logging
                self.logger = logging.getLogger(f"test_{self.min_factor}")
                self.logger.setLevel(logging.INFO)
                self.logger.handlers.clear()
                fh = logging.FileHandler(str(self.log_path), encoding="utf-8")
                self.logger.addHandler(fh)

            def run(self):
                call_log.append(self.min_factor)
                self.logger.info(f"Fake run for {self.min_factor}")

        with mock.patch(
            "app.scripts.run_augmentation_batch.AugmentationPipeline", FakePipeline
        ):
            results = run_batch(
                factors=["2x", "3x"],
                target_ratio=0.50,
                output=tmp_dir,
                seed=42,
            )

        # All factors were called
        assert call_log == ["2x", "3x"], f"Expected ['2x', '3x'], got {call_log}"

        # All results are success
        assert results == {"2x": "success", "3x": "success"}, f"Unexpected results: {results}"

        # Log files exist
        for factor in ["2x", "3x"]:
            log_file = Path(tmp_dir) / f"augmented_{factor}_balanced_5050" / f"{factor}.txt"
            assert log_file.exists(), f"Log file missing: {log_file}"
            content = log_file.read_text()
            assert f"Fake run for {factor}" in content, f"Log content wrong for {factor}"

        print("  [OK] Batch runner calls pipeline for each factor and creates logs")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Test partial failure handling
# ---------------------------------------------------------------------------

def test_run_batch_partial_failure():
    """
    If one factor fails, subsequent factors still run.
    """
    from app.scripts.run_augmentation_batch import run_batch

    tmp_dir = tempfile.mkdtemp(prefix="test_batch_fail_")

    try:
        class FakePipeline:
            def __init__(self, **kwargs):
                self.min_factor = kwargs["min_factor"]
                self.output_dir = Path(tmp_dir) / f"augmented_{self.min_factor}_balanced_5050"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.log_path = self.output_dir / f"{self.min_factor}.txt"

                import logging
                self.logger = logging.getLogger(f"test_fail_{self.min_factor}")
                self.logger.setLevel(logging.INFO)
                self.logger.handlers.clear()
                fh = logging.FileHandler(str(self.log_path), encoding="utf-8")
                self.logger.addHandler(fh)

            def run(self):
                if self.min_factor == "3x":
                    raise RuntimeError("Simulated failure for 3x")
                self.logger.info(f"Fake run for {self.min_factor}")

        with mock.patch(
            "app.scripts.run_augmentation_batch.AugmentationPipeline", FakePipeline
        ):
            results = run_batch(
                factors=["2x", "3x", "5x"],
                target_ratio=0.50,
                output=tmp_dir,
                seed=42,
            )

        assert results["2x"] == "success"
        assert "Simulated failure" in results["3x"]
        assert results["5x"] == "success"

        print("  [OK] Partial failure: subsequent factors still run")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. Test log file naming matches min_factor
# ---------------------------------------------------------------------------

def test_log_file_naming():
    """Log file is named {min_factor}.txt inside the output dir."""
    from app.scripts.run_augmentation_batch import run_batch

    tmp_dir = tempfile.mkdtemp(prefix="test_batch_name_")

    try:
        class FakePipeline:
            def __init__(self, **kwargs):
                self.min_factor = kwargs["min_factor"]
                self.output_dir = Path(tmp_dir) / f"augmented_{self.min_factor}_balanced_5050"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.log_path = self.output_dir / f"{self.min_factor}.txt"

                import logging
                self.logger = logging.getLogger(f"test_name_{self.min_factor}")
                self.logger.setLevel(logging.INFO)
                self.logger.handlers.clear()
                fh = logging.FileHandler(str(self.log_path), encoding="utf-8")
                self.logger.addHandler(fh)

            def run(self):
                self.logger.info("ok")

        with mock.patch(
            "app.scripts.run_augmentation_batch.AugmentationPipeline", FakePipeline
        ):
            run_batch(
                factors=["10x"],
                target_ratio=0.50,
                output=tmp_dir,
                seed=42,
            )

        expected = Path(tmp_dir) / "augmented_10x_balanced_5050" / "10x.txt"
        assert expected.exists(), f"Expected log at {expected}"
        print("  [OK] Log file naming: {min_factor}.txt")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nRunning batch augmentation tests...\n")
    test_factor_format_validation()
    test_default_factors()
    test_run_batch_mocked()
    test_run_batch_partial_failure()
    test_log_file_naming()
    print("\nAll tests passed!\n")
