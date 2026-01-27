#!/usr/bin/env python3
"""
Simple script to convert FLAC audio files to WAV format.
Usage: python convert_flac_to_wav.py <input_file.flac>
"""

import sys
import os
from pydub import AudioSegment


def convert_flac_to_wav(flac_path):
    """
    Convert a FLAC file to WAV format.
    
    Args:
        flac_path: Path to the input FLAC file
    
    Returns:
        Path to the output WAV file
    """
    # Check if file exists
    if not os.path.exists(flac_path):
        raise FileNotFoundError(f"File not found: {flac_path}")
    
    # Check if it's a FLAC file
    if not flac_path.lower().endswith('.flac'):
        raise ValueError("Input file must be a .flac file")
    
    # Load the FLAC file
    print(f"Loading {flac_path}...")
    audio = AudioSegment.from_file(flac_path, format="flac")
    
    # Create output path (same directory, same name, .wav extension)
    wav_path = os.path.splitext(flac_path)[0] + '.wav'
    
    # Export as WAV
    print(f"Converting to WAV...")
    audio.export(wav_path, format="wav")
    
    print(f"✓ Conversion complete: {wav_path}")
    return wav_path


if __name__ == "__main__":
    
    lista = ['LA_T_0000038.flac','LA_T_0000039.flac','LA_T_0000040.flac','LA_T_0000041.flac','LA_T_0000042.flac',]
    for i in lista:
        
        input_file = i 
        output_file = convert_flac_to_wav(input_file)
    
    sys.exit(1)