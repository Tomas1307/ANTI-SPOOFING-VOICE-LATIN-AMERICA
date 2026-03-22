#!/usr/bin/env bash
# =============================================================================
# run_validations.sh - Run all 4 attack pipelines in validation mode
#
# Executes each pipeline sequentially, switching virtual environments between
# runs. Each pipeline generates 3 speakers x 2 samples = 6 attack samples,
# validates them (Parakeet STT + WER/CER + NISQA + ECAPA), and writes:
#   - validated_samples.json (passed samples)
#   - metrics/<SYSTEM_ID>_validation.csv (all samples, passed + rejected)
#   - LA/ directory (ASVspoof2019 format output)
#
# Usage (on ml-server03):
#   cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
#   bash app/scripts/run_validations.sh [--gpu N] [--skip-fishgram]
#
# Options:
#   --gpu N           GPU device index (default: 1). Check nvidia-smi first.
#   --skip-fishgram   Skip FishGram pipeline (requires Fish Speech server).
#
# FishGram requires the Fish Speech HTTP server running on a SEPARATE GPU.
# Start it in another terminal BEFORE running this script:
#   cd ~/fish-speech
#   source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
#   export CUDA_VISIBLE_DEVICES=3
#   python3 tools/api_server.py --listen 0.0.0.0:8080 \
#       --checkpoint-path checkpoints/s1-mini/
# =============================================================================

set -euo pipefail

REPO_DIR="$HOME/ANTI-SPOOFING-VOICE-LATIN-AMERICA"
ENVS_DIR="$REPO_DIR/envs"
GPU=1
SKIP_FISHGRAM=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --skip-fishgram)
            SKIP_FISHGRAM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

echo "================================================================"
echo "  ATTACK PIPELINE VALIDATION RUNNER"
echo "================================================================"
echo "  Repository : $REPO_DIR"
echo "  GPU        : $GPU"
echo "  Skip FishGram: $SKIP_FISHGRAM"
echo "  Timestamp  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

cd "$REPO_DIR"

PASSED=0
FAILED=0
SKIPPED=0

run_pipeline() {
    local name="$1"
    local env_name="$2"
    local test_script="$3"

    echo "----------------------------------------------------------------"
    echo "  PIPELINE: $name"
    echo "  Env: $env_name"
    echo "  Script: $test_script"
    echo "----------------------------------------------------------------"

    local env_path="$ENVS_DIR/$env_name/bin/activate"
    if [[ ! -f "$env_path" ]]; then
        echo "  ERROR: Virtual environment not found at $env_path"
        echo "  SKIPPING $name"
        echo ""
        FAILED=$((FAILED + 1))
        return 1
    fi

    # shellcheck disable=SC1090
    source "$env_path"

    if python3 "$test_script"; then
        echo ""
        echo "  $name: PASSED"
        PASSED=$((PASSED + 1))
    else
        echo ""
        echo "  $name: FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi

    deactivate
    echo ""
}

# --- Chatterbox ---
run_pipeline "CHATTERBOX" "chatterbox_env" "test_chatterbox_pipeline.py"

# --- OpenVoice ---
run_pipeline "OPENVOICE" "openvoice_env" "test_openvoice_pipeline.py"

# --- Qwen ---
run_pipeline "QWEN3TTS" "qwen_env" "test_qwen_pipeline.py"

# --- FishGram (requires Fish Speech server) ---
if [[ "$SKIP_FISHGRAM" == "true" ]]; then
    echo "----------------------------------------------------------------"
    echo "  PIPELINE: FISHGRAM (SKIPPED by --skip-fishgram)"
    echo "----------------------------------------------------------------"
    echo ""
    SKIPPED=$((SKIPPED + 1))
else
    run_pipeline "FISHGRAM" "fishgram_env" "test_fishgram_pipeline.py"
fi

echo "================================================================"
echo "  VALIDATION COMPLETE"
echo "================================================================"
echo "  Passed  : $PASSED"
echo "  Failed  : $FAILED"
echo "  Skipped : $SKIPPED"
echo "  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Metrics CSVs:"
echo "    data/chatterbox_output/metrics/CHATTERBOX_validation.csv"
echo "    data/openvoice_output/metrics/OPENVOICE_validation.csv"
echo "    data/qwen_output/metrics/QWEN3TTS_validation.csv"
echo "    data/fishgram_output/metrics/FISHGRAM_validation.csv"
echo "================================================================"
