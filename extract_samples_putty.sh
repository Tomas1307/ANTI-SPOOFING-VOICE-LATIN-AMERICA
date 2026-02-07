#!/bin/bash
#
# Extract samples for Martin's artifact analysis
# Run on PuTTY/Linux server where augmented data is stored
#
# Usage:
#   bash extract_samples_putty.sh
#   bash extract_samples_putty.sh 15  # Extract 15 speakers instead of 10
#

# Configuration
N_SPEAKERS=${1:-5}  # Default 10 speakers, or use first argument
FACTOR="3x"
SOURCE_DIR="data/augmented/augmented_${FACTOR}_balanced_5050/LA/ASVspoof2019_LA_train"
OUTPUT_DIR="datos_martin"
PROTOCOL_FILE="${SOURCE_DIR}/ASVspoof2019.LA.cm.train.trn.txt"
FLAC_DIR="${SOURCE_DIR}/flac"

echo "========================================================================"
echo "Extracting ${N_SPEAKERS} speakers from ${FACTOR} augmented dataset"
echo "========================================================================"
echo ""

# Check if source exists
if [ ! -f "${PROTOCOL_FILE}" ]; then
    echo "ERROR: Protocol file not found: ${PROTOCOL_FILE}"
    echo ""
    echo "Available augmented datasets:"
    ls -d data/augmented/augmented_* 2>/dev/null || echo "  None found"
    exit 1
fi

if [ ! -d "${FLAC_DIR}" ]; then
    echo "ERROR: FLAC directory not found: ${FLAC_DIR}"
    exit 1
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}/flac"

# Count total samples
TOTAL_SAMPLES=$(wc -l < "${PROTOCOL_FILE}")
echo "Total samples in protocol: ${TOTAL_SAMPLES}"
echo ""

# Extract unique speakers and count their samples
echo "Analyzing speakers..."
awk '{print $1}' "${PROTOCOL_FILE}" | sort | uniq -c | sort -rn > /tmp/speaker_counts.txt
TOTAL_SPEAKERS=$(wc -l < /tmp/speaker_counts.txt)
echo "Total speakers: ${TOTAL_SPEAKERS}"
echo ""

# Select top N speakers by sample count
echo "Selecting top ${N_SPEAKERS} speakers with most samples..."
head -n ${N_SPEAKERS} /tmp/speaker_counts.txt | awk '{print $2}' > /tmp/selected_speakers.txt

echo "Selected speakers:"
cat /tmp/selected_speakers.txt | while read speaker; do
    n_bonafide=$(grep "^${speaker} " "${PROTOCOL_FILE}" | grep " bonafide$" | wc -l)
    n_spoof=$(grep "^${speaker} " "${PROTOCOL_FILE}" | grep " spoof$" | wc -l)
    echo "  ${speaker}: ${n_bonafide} bonafide, ${n_spoof} spoof"
done
echo ""

# Extract protocol entries for selected speakers
echo "Extracting protocol entries..."
> "${OUTPUT_DIR}/protocol.txt"
while read speaker; do
    grep "^${speaker} " "${PROTOCOL_FILE}" >> "${OUTPUT_DIR}/protocol.txt"
done < /tmp/selected_speakers.txt

N_ENTRIES=$(wc -l < "${OUTPUT_DIR}/protocol.txt")
echo "Extracted ${N_ENTRIES} protocol entries"
echo ""

# Copy FLAC files
echo "Copying FLAC files (this may take a while)..."
COPIED=0
MISSING=0

awk '{print $2}' "${OUTPUT_DIR}/protocol.txt" | while read audio_id; do
    src="${FLAC_DIR}/${audio_id}.flac"
    dst="${OUTPUT_DIR}/flac/${audio_id}.flac"

    if [ -f "${src}" ]; then
        cp "${src}" "${dst}"
        COPIED=$((COPIED + 1))

        # Progress indicator every 100 files
        if [ $((COPIED % 100)) -eq 0 ]; then
            echo "  Copied ${COPIED} files..."
        fi
    else
        echo "  WARNING: Missing file: ${src}"
        MISSING=$((MISSING + 1))
    fi
done

echo "Done copying FLAC files"
echo ""

# Create README
echo "Creating README..."
cat > "${OUTPUT_DIR}/README.txt" << 'EOF'
======================================================================
AUGMENTED VOICE SAMPLES FOR ARTIFACT ANALYSIS
======================================================================

Dataset Information:
  Augmentation factor: 10x
  Target ratio: 50/50 (bonafide/spoof)
  Number of speakers: See below

File Structure:
  flac/           - Audio files in FLAC format
  protocol.txt    - Mapping of audio IDs to metadata
  README.txt      - This file

Protocol File Format:
  SPEAKER_ID AUDIO_ID SYSTEM_ID KEY

  - SPEAKER_ID: Speaker identifier
  - AUDIO_ID: Unique file identifier (e.g., LA_T_0000001)
  - SYSTEM_ID: Augmentation type ('-' = clean/original)
  - KEY: 'bonafide' or 'spoof'

======================================================================
AUGMENTATION TYPES
======================================================================

1. RIR + Noise (60%): Room acoustics + background noise
   Format: RIR_{ROOM}_{NOISE}_SNR{DB}

   ROOM: SMALL, MEDIUM, LARGE
   NOISE: NOI (noise), SPE (speech), MUS (music)
   SNR: Signal-to-noise ratio in dB

   Example: RIR_SMALL_NOI_SNR12
   - Small room
   - Environmental noise
   - 12 dB SNR

2. Codec Degradation (30%): Telephone/VoIP artifacts
   Format: CODEC_{SR}K_LOSS{PCT}PCT[_BP][_Q{BITS}]

   SR: Sample rate (8K or 16K Hz)
   LOSS: Packet loss percentage (0-5%)
   _BP: Bandpass filter applied (300-3400 Hz)
   _Q{BITS}: Quantization (8 or 12 bits)

   Example: CODEC_8K_LOSS2PCT_BP_Q8
   - 8 kHz codec simulation
   - 2% packet loss
   - Bandpass filter applied
   - 8-bit quantization

3. RawBoost (10%): Signal-dependent distortions
   Format: RAWBOOST_{OP1}[_{OP2}...]

   Operations:
   LF = Linear FIR filtering
   NL = Nonlinear tanh distortion
   AN = Additive noise (signal-dependent)
   GV = Gain variation (AGC effects)
   CL = Hard clipping

   Example: RAWBOOST_LF_AN_GV
   - Linear filter applied
   - Additive noise
   - Gain variation

======================================================================
USAGE FOR ARTIFACT ANALYSIS
======================================================================

1. Load protocol.txt to understand sample metadata
2. Load corresponding FLAC files from flac/ directory
3. Compare clean samples (SYSTEM_ID = '-') with augmented versions
4. Analyze artifacts by augmentation type:
   - RIR: Reverb characteristics, noise floor
   - CODEC: Bandwidth limitation, quantization noise, packet loss gaps
   - RAWBOOST: Nonlinear distortions, clipping effects
5. Group by SPEAKER_ID to study speaker-specific effects

Example analysis workflow:
  1. Find clean bonafide: grep "- bonafide" protocol.txt
  2. Find augmented versions: grep "SPEAKER_ID.*bonafide" protocol.txt
  3. Compare spectrograms between clean and augmented
  4. Measure SNR, spectral centroid, formant shifts, etc.

======================================================================
EOF

# Append speaker list to README
echo "" >> "${OUTPUT_DIR}/README.txt"
echo "Selected Speakers:" >> "${OUTPUT_DIR}/README.txt"
cat /tmp/selected_speakers.txt | while read speaker; do
    n_bonafide=$(grep "^${speaker} " "${OUTPUT_DIR}/protocol.txt" | grep " bonafide$" | wc -l)
    n_spoof=$(grep "^${speaker} " "${OUTPUT_DIR}/protocol.txt" | grep " spoof$" | wc -l)
    echo "  ${speaker}: ${n_bonafide} bonafide, ${n_spoof} spoof" >> "${OUTPUT_DIR}/README.txt"
done
echo "" >> "${OUTPUT_DIR}/README.txt"
echo "======================================================================" >> "${OUTPUT_DIR}/README.txt"

# Final statistics
FINAL_FILES=$(ls "${OUTPUT_DIR}/flac" | wc -l)
FINAL_SIZE=$(du -sh "${OUTPUT_DIR}" | awk '{print $1}')

echo "========================================================================"
echo "EXTRACTION COMPLETE"
echo "========================================================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}/"
echo "  FLAC files: ${FINAL_FILES}"
echo "  Protocol entries: ${N_ENTRIES}"
echo "  Total size: ${FINAL_SIZE}"
echo ""
echo "Files ready to share with Martin for artifact analysis!"
echo ""
echo "To compress for transfer:"
echo "  tar -czf datos_martin.tar.gz datos_martin/"
echo ""

# Cleanup temp files
rm -f /tmp/speaker_counts.txt /tmp/selected_speakers.txt
