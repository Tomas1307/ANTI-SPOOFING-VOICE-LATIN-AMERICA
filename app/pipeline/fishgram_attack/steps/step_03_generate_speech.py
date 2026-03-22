"""
Step 3: Generate Synthetic Speech

Generates synthetic Spanish speech using Fish Speech HTTP API server.
The Fish Speech server must be running on ml-server03 before executing this step.
"""
import base64
import json
import time
import requests
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from app.pipeline.fishgram_attack.settings import settings
from app.pipeline.fishgram_attack.schemas.generation_result import GenerationResult


class SpeechGenerator:
    """Generates synthetic speech using Fish Speech HTTP API.

    Sends text and reference audio to the Fish Speech API server
    running on ml-server03. The server handles model inference
    and returns synthesized audio bytes.

    The Fish Speech server must be started separately before running
    this step. See guide/how_to_run_fishgram/README.md for server startup instructions.

    Attributes:
        output_dir: Directory where generated audio files are saved.
        api_url: URL of the Fish Speech HTTP API server.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        skip_existing: bool = False
    ):
        """Initialize speech generator.

        Args:
            output_dir: Output directory (default: from settings).
            skip_existing: When True, skip WAV files that already exist on disk (for resume).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.skip_existing = skip_existing
        self.api_url = settings.FISH_SPEECH_API_URL

    def _check_server_health(self) -> bool:
        """Verify the Fish Speech API server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
        except requests.Timeout:
            return False

    def _generate_single(
        self,
        text: str,
        reference_audio_path: Path,
        reference_text: str,
        output_path: Path
    ) -> float:
        """Generate a single synthetic audio sample via Fish Speech API.

        Sends the text and reference audio to the Fish Speech HTTP server,
        receives synthesized audio bytes, and saves them to disk.

        Args:
            text: The Spanish text to synthesize.
            reference_audio_path: Path to the speaker reference audio file.
            reference_text: Transcript of the reference audio (empty string if unknown).
            output_path: Path where the generated WAV file will be saved.

        Returns:
            The generation time in seconds.

        Raises:
            RuntimeError: If the API server returns an error response.
            requests.ConnectionError: If the server is unreachable.
        """
        start_time = time.time()

        # Read reference audio bytes
        with open(reference_audio_path, "rb") as f:
            ref_audio_bytes = f.read()

        payload = {
            "text": text,
            "references": [
                {
                    "audio": base64.b64encode(ref_audio_bytes).decode("utf-8"),
                    "text": reference_text
                }
            ],
            "format": settings.FISH_SPEECH_FORMAT,
            "top_p": settings.FISH_SPEECH_TOP_P,
            "temperature": settings.FISH_SPEECH_TEMPERATURE,
            "repetition_penalty": settings.FISH_SPEECH_REPETITION_PENALTY,
            "streaming": False,
            "normalize": True,
            "max_new_tokens": 1024
        }

        response = requests.post(
            f"{self.api_url}/v1/tts",
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Fish Speech API error {response.status_code}: {response.text}"
            )

        generation_time = time.time() - start_time

        # Save audio response to file
        audio_bytes = response.content
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        return generation_time

    def execute(self) -> GenerationResult:
        """Generate synthetic speech for all speaker-text pairs.

        Connects to the Fish Speech HTTP API server, sends each text prompt
        with its corresponding speaker reference audio, and saves the
        generated audio files.

        Returns:
            GenerationResult with metadata path, counts, and statistics.

        Raises:
            ConnectionError: If the Fish Speech API server is not reachable.
        """
        logger.info("Generating synthetic speech...")
        logger.info(f"  Fish Speech API: {self.api_url}")
        if self.skip_existing:
            logger.info("  Resume mode: skip_existing=True (will skip already-generated WAVs)")

        # Verify server is running
        if not self._check_server_health():
            raise ConnectionError(
                f"Fish Speech API server is not reachable at {self.api_url}. "
                f"Start the server with: cd ~/fish-speech && "
                f"CUDA_VISIBLE_DEVICES=1 python -m tools.api_server "
                f"--listen 0.0.0.0:8080 "
                f"--llama-checkpoint-path checkpoints/s1-mini "
                f"--decoder-checkpoint-path checkpoints/s1-mini/codec.pth "
                f"--decoder-config-name modded_dac_vq"
            )

        logger.info("  Server health check: OK")

        # Create output directory
        gen_dir = self.output_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        ref_metadata_path = self.output_dir / "reference_metadata.json"
        prompts_path = self.output_dir / "text_prompts.json"

        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        # Generate speech for each speaker-text pair
        generated = {}
        failed = []
        rtf_values = []

        total_pairs = sum(len(texts) for texts in prompts.values())
        logger.info(f"Generating {total_pairs} synthetic samples...")

        with tqdm(total=total_pairs, desc="Generating") as pbar:
            for speaker_id in sorted(references.keys()):
                ref_path = Path(references[speaker_id]["reference_path"])
                split = references[speaker_id]["split"]

                # Verify reference audio exists
                if not ref_path.exists():
                    logger.error(f"Reference audio not found for {speaker_id}: {ref_path}")
                    failed.extend([p["text_id"] for p in prompts[speaker_id]])
                    pbar.update(len(prompts[speaker_id]))
                    continue

                # Generate for each text prompt
                for prompt_data in prompts[speaker_id]:
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"
                    output_path = gen_dir / f"FISHGRAM_{speaker_id}_{text_id}.wav"

                    # Skip existing files when resuming
                    if self.skip_existing and output_path.exists():
                        try:
                            synthetic_audio, _ = librosa.load(
                                output_path, sr=settings.SAMPLE_RATE
                            )
                            audio_duration = len(synthetic_audio) / settings.SAMPLE_RATE
                            generated[sample_id] = {
                                "speaker_id": speaker_id,
                                "text_id": text_id,
                                "text": text,
                                "audio_path": str(output_path),
                                "duration_seconds": audio_duration,
                                "generation_time_seconds": 0.0,
                                "rtf": 0.0,
                                "split": split,
                                "skipped_existing": True
                            }
                            logger.debug(f"Skipping existing: {output_path.name}")
                        except Exception as e:
                            logger.warning(f"Existing file unreadable {output_path.name}: {e}")
                        pbar.update(1)
                        continue

                    try:
                        generation_time = self._generate_single(
                            text=text,
                            reference_audio_path=ref_path,
                            reference_text="",
                            output_path=output_path
                        )

                        # Load generated audio to get duration
                        synthetic_audio, _ = librosa.load(
                            output_path, sr=settings.SAMPLE_RATE
                        )
                        audio_duration = len(synthetic_audio) / settings.SAMPLE_RATE
                        rtf = generation_time / audio_duration if audio_duration > 0 else 0.0
                        rtf_values.append(rtf)

                        # Store metadata
                        generated[sample_id] = {
                            "speaker_id": speaker_id,
                            "text_id": text_id,
                            "text": text,
                            "audio_path": str(output_path),
                            "duration_seconds": audio_duration,
                            "generation_time_seconds": generation_time,
                            "rtf": rtf,
                            "split": split
                        }

                        logger.debug(
                            f"Generated {sample_id}: {audio_duration:.1f}s "
                            f"in {generation_time:.1f}s (RTF={rtf:.2f})"
                        )

                    except Exception as e:
                        logger.error(f"Generation failed for {sample_id}: {e}")
                        failed.append(sample_id)

                    pbar.update(1)

        # Save generation metadata
        gen_metadata_path = self.output_dir / "generation_metadata.json"
        with open(gen_metadata_path, "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2, ensure_ascii=False)

        avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0

        logger.info(f"Generated {len(generated)} samples")
        logger.info(f"  Failed: {len(failed)}")
        logger.info(f"  Average RTF: {avg_rtf:.2f}")
        logger.info(f"  Metadata saved to: {gen_metadata_path}")

        return GenerationResult(
            generated_samples_path=gen_metadata_path,
            total_generated=len(generated),
            failed_generations=failed,
            avg_rtf=avg_rtf
        )
