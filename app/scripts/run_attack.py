"""
Production Attack Runner - Entry Point.

Interactive console for running Qwen3-TTS and FishGram attack pipelines
at production scale on HABLA v2 dataset (1,567 speakers, ~35k samples).

Usage on ml-server03:
    export CUDA_VISIBLE_DEVICES=1
    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/<pipeline>_env/bin/activate
    python3 app/scripts/run_attack.py
"""
from app.runner import ProductionRunner


if __name__ == "__main__":
    runner = ProductionRunner()
    runner.run()
