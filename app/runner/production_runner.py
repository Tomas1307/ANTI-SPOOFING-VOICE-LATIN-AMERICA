"""
Production Runner for Attack Pipelines.

Provides interactive console menu, state detection, checkpoint/resume,
and retry logic for running Qwen3-TTS and FishGram attack pipelines
at production scale (HABLA v2, 1567 speakers, ~35k samples).
"""
import json
import time
import requests
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


PIPELINES = {
    "1": {
        "name": "qwen",
        "display": "Qwen3-TTS (local model, 1.7B params)",
        "system_id": "QWEN3TTS",
        "output_dir_setting": "data/qwen_output",
    },
    "2": {
        "name": "fishgram",
        "display": "FishGram / Fish Speech (HTTP API, 4B params)",
        "system_id": "FISHGRAM",
        "output_dir_setting": "data/fishgram_output",
    },
    "3": {
        "name": "omnivoice",
        "display": "OmniVoice (k2-fsa, diffusion LM, 646 languages)",
        "system_id": "OMNIVOICE",
        "output_dir_setting": "data/omnivoice_output",
    },
}


class ProductionRunner:
    """Interactive production runner for attack pipelines.

    Provides a console-driven workflow for running Qwen3-TTS and FishGram
    attack pipelines at production scale. Handles:
    - Pipeline selection via interactive menu
    - State detection using existing JSON metadata files as checkpoints
    - Resume support by skipping completed steps and already-generated WAVs
    - Retry loop for regenerating samples rejected by Step 4 validation
    - FishGram server health pre-check before generation

    Attributes:
        pipeline_name: Selected pipeline identifier ('qwen' or 'fishgram').
        pipeline_info: Metadata dict for the selected pipeline.
    """

    def __init__(self):
        """Initialize production runner."""
        self.pipeline_name: Optional[str] = None
        self.pipeline_info: Optional[Dict] = None

    def run(self):
        """Execute the interactive production runner workflow.

        Presents the pipeline selection menu, detects existing state,
        offers resume or fresh start, then executes the pipeline with
        retry logic for rejected samples.
        """
        self._print_banner()

        # Select pipeline
        self.pipeline_info = self._select_pipeline()
        if self.pipeline_info is None:
            return
        self.pipeline_name = self.pipeline_info["name"]

        # Detect existing state
        output_dir = Path(self.pipeline_info["output_dir_setting"])
        system_id = self.pipeline_info["system_id"]
        state = self._detect_state(output_dir, system_id)

        # Determine run mode (fresh vs resume)
        run_mode = self._prompt_resume(state, output_dir)
        if run_mode is None:
            return

        # FishGram server health pre-check
        if self.pipeline_name == "fishgram" and run_mode["run_step_3"]:
            if not self._check_fishgram_server():
                return

        # Execute pipeline
        start_time = time.time()
        self._execute_pipeline(run_mode)
        elapsed = time.time() - start_time

        # Print summary
        self._print_summary(output_dir, elapsed)

    def _print_banner(self):
        """Print the runner banner."""
        print("=" * 70)
        print("HABLA ANTI-SPOOFING - PRODUCTION ATTACK RUNNER")
        print("=" * 70)
        print()
        print("Select pipeline:")
        for key, info in PIPELINES.items():
            print(f"  [{key}] {info['display']}")
        print("  [q] Quit")
        print()

    def _select_pipeline(self) -> Optional[Dict]:
        """Prompt user to select an attack pipeline.

        Returns:
            Pipeline info dict, or None if user quits.
        """
        choice = input("Pipeline choice: ").strip().lower()

        if choice == "q":
            print("Exiting.")
            return None

        if choice not in PIPELINES:
            print(f"Invalid choice: '{choice}'. Exiting.")
            return None

        info = PIPELINES[choice]
        print(f"\nSelected: {info['display']}")
        return info

    def _detect_state(self, output_dir: Path, system_id: str) -> Dict:
        """Detect existing pipeline output state.

        Checks which JSON metadata files exist and counts generated WAV files.
        The JSON files themselves serve as natural checkpoints — no separate
        checkpoint system needed.

        Args:
            output_dir: Pipeline output directory.
            system_id: System identifier prefix for WAV files (e.g. 'QWEN3TTS').

        Returns:
            Dict with state flags and counts.
        """
        gen_dir = output_dir / "generated"

        has_references = (output_dir / "reference_metadata.json").exists()
        has_prompts = (output_dir / "text_prompts.json").exists()
        has_generation = (output_dir / "generation_metadata.json").exists()
        has_validation = (output_dir / "validated_samples.json").exists()

        generated_count = 0
        if gen_dir.exists():
            generated_count = len(list(gen_dir.glob(f"{system_id}_*.wav")))

        expected_count = 0
        if has_prompts:
            prompts_path = output_dir / "text_prompts.json"
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            expected_count = sum(len(texts) for texts in prompts.values())

        return {
            "has_references": has_references,
            "has_prompts": has_prompts,
            "has_generation": has_generation,
            "has_validation": has_validation,
            "generated_count": generated_count,
            "expected_count": expected_count,
        }

    def _prompt_resume(self, state: Dict, output_dir: Path) -> Optional[Dict]:
        """Present resume options based on detected state.

        Args:
            state: State dict from _detect_state.
            output_dir: Pipeline output directory.

        Returns:
            Dict with run configuration (step toggles + skip_existing), or None to quit.
        """
        has_any = any([
            state["has_references"],
            state["has_prompts"],
            state["generated_count"] > 0,
        ])

        if not has_any:
            print("\nNo existing output detected. Starting fresh run.")
            return {
                "run_step_1": True,
                "run_step_2": True,
                "run_step_3": True,
                "run_step_4": True,
                "run_step_5": True,
                "skip_existing": False,
            }

        # Show existing state
        print("\n--- Existing output detected ---")
        print(f"  reference_metadata.json  : {'FOUND' if state['has_references'] else 'MISSING'}")
        print(f"  text_prompts.json        : {'FOUND' if state['has_prompts'] else 'MISSING'}")

        if state["expected_count"] > 0:
            print(
                f"  generated/ WAV files     : "
                f"{state['generated_count']}/{state['expected_count']}"
            )
        else:
            print(f"  generated/ WAV files     : {state['generated_count']}")

        print(f"  generation_metadata.json : {'FOUND' if state['has_generation'] else 'MISSING'}")
        print(f"  validated_samples.json   : {'FOUND' if state['has_validation'] else 'MISSING'}")
        print()

        # Determine resume point
        if state["has_references"] and state["has_prompts"]:
            gen_complete = (
                state["has_generation"]
                and state["generated_count"] >= state["expected_count"]
            )

            if gen_complete and state["has_validation"]:
                print("  All steps appear complete.")
                print("  [r] Re-run Steps 4-5 (re-validate and reformat)")
                print("  [f] Fresh start (re-run everything)")
                print("  [q] Quit")
            elif gen_complete:
                print("  Generation complete but not validated.")
                print("  [r] Resume from Step 4 (validate + format)")
                print("  [f] Fresh start (re-run everything)")
                print("  [q] Quit")
            else:
                remaining = state["expected_count"] - state["generated_count"]
                print(
                    f"  Generation incomplete: {remaining} samples remaining."
                )
                print(
                    f"  [r] Resume Step 3 from "
                    f"{state['generated_count']}/{state['expected_count']}, "
                    f"then Steps 4-5"
                )
                print("  [f] Fresh start (re-run everything)")
                print("  [q] Quit")
        else:
            print("  Partial metadata found. Recommend fresh start.")
            print("  [f] Fresh start (re-run everything)")
            print("  [q] Quit")

        print()
        choice = input("Choice: ").strip().lower()

        if choice == "q":
            print("Exiting.")
            return None

        if choice == "f":
            print("\nFresh start selected. Existing output will be overwritten.")
            return {
                "run_step_1": True,
                "run_step_2": True,
                "run_step_3": True,
                "run_step_4": True,
                "run_step_5": True,
                "skip_existing": False,
            }

        if choice == "r":
            gen_complete = (
                state["has_generation"]
                and state["generated_count"] >= state["expected_count"]
            )

            if gen_complete and state["has_validation"]:
                # All done, re-run validation + format
                return {
                    "run_step_1": False,
                    "run_step_2": False,
                    "run_step_3": False,
                    "run_step_4": True,
                    "run_step_5": True,
                    "skip_existing": False,
                }
            elif gen_complete:
                # Generation done, run validation + format
                return {
                    "run_step_1": False,
                    "run_step_2": False,
                    "run_step_3": False,
                    "run_step_4": True,
                    "run_step_5": True,
                    "skip_existing": False,
                }
            else:
                # Resume generation, then validate + format
                return {
                    "run_step_1": False,
                    "run_step_2": False,
                    "run_step_3": True,
                    "run_step_4": True,
                    "run_step_5": True,
                    "skip_existing": True,
                }

        print(f"Invalid choice: '{choice}'. Exiting.")
        return None

    def _check_fishgram_server(self) -> bool:
        """Verify the Fish Speech API server is reachable.

        Returns:
            True if server is reachable, False otherwise.
        """
        from app.pipeline.fishgram_attack.settings import settings as fg_settings

        api_url = fg_settings.FISH_SPEECH_API_URL
        print(f"\nChecking Fish Speech server at {api_url}...")

        try:
            response = requests.get(f"{api_url}/", timeout=5)
            if response.status_code == 200:
                print("  Server health check: OK")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        print(f"  ERROR: Fish Speech server is not reachable at {api_url}")
        print()
        print("  Start the server on ml-server03 with:")
        print("  cd ~/fish-speech")
        print(
            "  source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/"
            "envs/fishgram_env/bin/activate"
        )
        print(
            "  CUDA_VISIBLE_DEVICES=<GPU> python3 -m tools.api_server "
            "--llama-checkpoint-path checkpoints/s1-mini "
            "--decoder-checkpoint-path checkpoints/s1-mini/codec.pth "
            "--device cuda --half --listen 0.0.0.0:8080"
        )
        return False

    def _execute_pipeline(self, run_mode: Dict):
        """Execute the selected pipeline with retry logic.

        Runs the pipeline facade with the given step configuration.
        After Step 3+4 complete, checks for rejected samples and
        retries generation up to MAX_GENERATION_RETRIES times.

        Args:
            run_mode: Dict with step toggles and skip_existing flag.
        """
        if self.pipeline_name == "qwen":
            self._execute_qwen(run_mode)
        elif self.pipeline_name == "fishgram":
            self._execute_fishgram(run_mode)
        elif self.pipeline_name == "omnivoice":
            self._execute_omnivoice(run_mode)

    def _execute_qwen(self, run_mode: Dict):
        """Execute Qwen3-TTS pipeline with retry loop.

        Args:
            run_mode: Dict with step toggles and skip_existing flag.
        """
        from app.pipeline.qwen_attack import (
            QwenAttackPipeline,
            QwenPipelineConfig,
            settings,
        )

        # Configure production mode
        settings.VALIDATION_MODE = False

        config = QwenPipelineConfig(
            run_step_1=run_mode["run_step_1"],
            run_step_2=run_mode["run_step_2"],
            run_step_3=run_mode["run_step_3"],
            run_step_4=run_mode["run_step_4"],
            run_step_5=run_mode["run_step_5"],
            skip_existing_step_3=run_mode["skip_existing"],
        )

        pipeline = QwenAttackPipeline(config=config)
        pipeline.run()

        # Retry loop for rejected samples
        if run_mode["run_step_3"] and run_mode["run_step_4"]:
            self._retry_rejected(
                settings=settings,
                pipeline_cls=QwenAttackPipeline,
                config_cls=QwenPipelineConfig,
                system_id="QWEN3TTS",
            )

    def _execute_fishgram(self, run_mode: Dict):
        """Execute FishGram pipeline with retry loop.

        Args:
            run_mode: Dict with step toggles and skip_existing flag.
        """
        from app.pipeline.fishgram_attack import (
            FishGramAttackPipeline,
            FishGramPipelineConfig,
            settings,
        )

        # Configure production mode
        settings.VALIDATION_MODE = False

        config = FishGramPipelineConfig(
            run_step_1=run_mode["run_step_1"],
            run_step_2=run_mode["run_step_2"],
            run_step_3=run_mode["run_step_3"],
            run_step_4=run_mode["run_step_4"],
            run_step_5=run_mode["run_step_5"],
            skip_existing_step_3=run_mode["skip_existing"],
        )

        pipeline = FishGramAttackPipeline(config=config)
        pipeline.run()

        # Retry loop for rejected samples
        if run_mode["run_step_3"] and run_mode["run_step_4"]:
            self._retry_rejected(
                settings=settings,
                pipeline_cls=FishGramAttackPipeline,
                config_cls=FishGramPipelineConfig,
                system_id="FISHGRAM",
            )

    def _execute_omnivoice(self, run_mode: Dict):
        """Execute OmniVoice pipeline with retry loop.

        OmniVoice runs as a local diffusion language model loaded into the
        active GPU process; no HTTP server is required. Step 4 may reject
        samples for non-verbal prefix artifacts (reference voice bleed) in
        addition to the standard WER/CER and duration checks; the retry
        loop regenerates them until clean or MAX_GENERATION_RETRIES is hit.

        Args:
            run_mode: Dict with step toggles and skip_existing flag.
        """
        from app.pipeline.omnivoice_attack import (
            OmniVoiceAttackPipeline,
            settings,
        )
        from app.pipeline.omnivoice_attack.schemas.pipeline_config import (
            OmniVoicePipelineConfig,
        )

        settings.VALIDATION_MODE = False

        config = OmniVoicePipelineConfig(
            run_step_1=run_mode["run_step_1"],
            run_step_2=run_mode["run_step_2"],
            run_step_3=run_mode["run_step_3"],
            run_step_4=run_mode["run_step_4"],
            run_step_5=run_mode["run_step_5"],
            skip_existing_step_3=run_mode["skip_existing"],
        )

        pipeline = OmniVoiceAttackPipeline(config=config)
        pipeline.run()

        if run_mode["run_step_3"] and run_mode["run_step_4"]:
            self._retry_rejected(
                settings=settings,
                pipeline_cls=OmniVoiceAttackPipeline,
                config_cls=OmniVoicePipelineConfig,
                system_id="OMNIVOICE",
            )

    def _retry_rejected(self, settings, pipeline_cls, config_cls, system_id: str):
        """Retry loop for regenerating rejected samples.

        After initial generation + validation, checks for rejected samples.
        If any exist, deletes their WAV files and re-runs Step 3 (with skip_existing)
        and Step 4 until no rejections remain or MAX_GENERATION_RETRIES is reached.

        Args:
            settings: Pipeline settings module with MAX_GENERATION_RETRIES and OUTPUT_DIR.
            pipeline_cls: Pipeline facade class (QwenAttackPipeline or FishGramAttackPipeline).
            config_cls: Pipeline config class (QwenPipelineConfig or FishGramPipelineConfig).
            system_id: System identifier for WAV file naming.
        """
        max_retries = settings.MAX_GENERATION_RETRIES
        output_dir = settings.OUTPUT_DIR
        gen_dir = output_dir / "generated"

        for retry_round in range(1, max_retries + 1):
            # Load validation results to find rejected samples
            validated_path = output_dir / "validated_samples.json"
            if not validated_path.exists():
                logger.warning("No validated_samples.json found, skipping retry loop")
                break

            # Load the validation CSV to find rejected sample IDs
            rejected_ids = self._find_rejected_sample_ids(output_dir, system_id)

            if not rejected_ids:
                print(f"\n  All samples passed validation. No retries needed.")
                break

            print(f"\n--- Retry round {retry_round}/{max_retries} ---")
            print(f"  Rejected: {len(rejected_ids)} samples")

            # Delete rejected WAV files
            deleted_count = 0
            for sample_id in rejected_ids:
                parts = sample_id.split("_", 1)
                if len(parts) == 2:
                    wav_path = gen_dir / f"{system_id}_{sample_id}.wav"
                else:
                    wav_path = gen_dir / f"{system_id}_{sample_id}.wav"

                if wav_path.exists():
                    wav_path.unlink()
                    deleted_count += 1

            print(f"  Deleted {deleted_count} rejected WAV files")
            print(f"  Re-generating with skip_existing=True...")

            # Re-run Step 3 (skip_existing fills gaps) + Step 4 + Step 5
            retry_config = config_cls(
                run_step_1=False,
                run_step_2=False,
                run_step_3=True,
                run_step_4=True,
                run_step_5=True,
                skip_existing_step_3=True,
            )

            retry_pipeline = pipeline_cls(config=retry_config)
            retry_pipeline.run()

        # Final check
        final_rejected = self._find_rejected_sample_ids(output_dir, system_id)
        if final_rejected:
            print(
                f"\n  {len(final_rejected)} samples remain rejected "
                f"after {max_retries} retry rounds (kept in CSV, excluded from LA/)"
            )

    def _find_rejected_sample_ids(self, output_dir: Path, system_id: str) -> list:
        """Find sample IDs that were rejected by Step 4 validation.

        Compares generation_metadata.json (all generated) against
        validated_samples.json (those that passed) to find rejections.

        Args:
            output_dir: Pipeline output directory.
            system_id: System identifier prefix.

        Returns:
            List of rejected sample_id strings.
        """
        gen_meta_path = output_dir / "generation_metadata.json"
        val_meta_path = output_dir / "validated_samples.json"

        if not gen_meta_path.exists() or not val_meta_path.exists():
            return []

        with open(gen_meta_path, "r", encoding="utf-8") as f:
            generated = json.load(f)

        with open(val_meta_path, "r", encoding="utf-8") as f:
            validated = json.load(f)

        generated_ids = set(generated.keys())
        validated_ids = set(validated.keys())

        rejected_ids = list(generated_ids - validated_ids)
        return rejected_ids

    def _print_summary(self, output_dir: Path, elapsed: float):
        """Print production run summary.

        Args:
            output_dir: Pipeline output directory.
            elapsed: Total elapsed time in seconds.
        """
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        system_id = self.pipeline_info["system_id"]
        gen_dir = output_dir / "generated"

        wav_count = 0
        if gen_dir.exists():
            wav_count = len(list(gen_dir.glob(f"{system_id}_*.wav")))

        # Load counts from metadata if available
        speaker_count = 0
        ref_path = output_dir / "reference_metadata.json"
        if ref_path.exists():
            with open(ref_path, "r", encoding="utf-8") as f:
                refs = json.load(f)
            speaker_count = len(refs)

        validated_count = 0
        val_path = output_dir / "validated_samples.json"
        if val_path.exists():
            with open(val_path, "r", encoding="utf-8") as f:
                validated = json.load(f)
            validated_count = len(validated)

        print()
        print("=" * 70)
        print(f"PRODUCTION RUN COMPLETE - {self.pipeline_info['display']}")
        print("=" * 70)
        print(f"  Total time      : {hours}h {minutes}m {seconds}s")
        print(f"  Speakers        : {speaker_count}")
        print(f"  Generated WAVs  : {wav_count}")
        print(f"  Passed valid.   : {validated_count}")
        print(f"  Rejected        : {wav_count - validated_count}")
        print()
        print(f"  Output directory : {output_dir}")
        print("=" * 70)
