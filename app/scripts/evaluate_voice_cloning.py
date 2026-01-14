"""
Simple Voice Cloning Test - Debug Version

Minimal script to test voice cloning with audio normalization.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Input audio array.
        target_db: Target dB level (default -20dB is good for speech).
        
    Returns:
        Normalized audio.
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)
    
    # Calculate gain
    gain = target_rms / rms
    
    # Apply gain
    normalized = audio * gain
    
    # Clip to prevent clipping
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def simple_test(
    model_path: str = "speech_tts_finetuned",
    speaker_id: str = "arf_00295",
    text: str = "Hola, esta es una prueba simple"
):
    """
    Simple test without file saving to debug.
    
    Args:
        model_path: Path to model.
        speaker_id: Speaker ID.
        text: Text to synthesize.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print("SIMPLE VOICE CLONING TEST")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Speaker: {speaker_id}")
    print(f"Text: {text}\n")
    
    # Load model
    print("Loading model...")
    try:
        processor = SpeechT5Processor.from_pretrained(model_path)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_path).to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load embeddings
    print("Loading embeddings...")
    try:
        embeddings = torch.load("app/embedding/speaker_embeddings.pt")
        if speaker_id not in embeddings:
            print(f"✗ Speaker {speaker_id} not found!")
            print(f"Available: {list(embeddings.keys())[:5]}...")
            return
        
        speaker_emb = embeddings[speaker_id].to(device)
        print(f"✓ Speaker embedding loaded\n")
    except Exception as e:
        print(f"✗ Error loading embeddings: {e}")
        return
    
    # Synthesize
    print("Synthesizing...")
    try:
        inputs = processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        import time
        start = time.time()
        
        with torch.no_grad():
            spec = model.generate_speech(input_ids, speaker_emb.unsqueeze(0))
            wave = vocoder(spec)
        
        duration = time.time() - start
        
        audio = wave.cpu().squeeze().numpy()
        audio_len = len(audio) / 16000
        
        print(f"✓ Success!\n")
        print(f"Results:")
        print(f"  Generation time: {duration:.2f}s")
        print(f"  Audio length: {audio_len:.2f}s")
        print(f"  RTF: {duration/audio_len:.2f}x")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        # Try to save with soundfile
        print(f"\nTrying to save audio...")
        try:
            import soundfile as sf
            output = "test_simple.wav"
            
            # Normalize audio before saving
            print(f"Original audio RMS: {np.sqrt(np.mean(audio**2)):.6f}")
            normalized_audio = normalize_audio(audio, target_db=-20.0)
            print(f"Normalized audio RMS: {np.sqrt(np.mean(normalized_audio**2)):.6f}")
            print(f"Gain applied: {(normalized_audio.max() / audio.max() if audio.max() > 0 else 1):.1f}x")
            
            # Save normalized audio
            sf.write(output, normalized_audio, 16000)
            print(f"✓ Saved normalized audio to: {output}")
        except ImportError:
            print("⚠ soundfile not installed, skipping save")
            print("  Install with: pip install soundfile")
        except Exception as e:
            print(f"⚠ Could not save: {e}")
        
    except Exception as e:
        print(f"✗ Error during synthesis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="speech_tts_finetuned")
    parser.add_argument("--speaker", default="arf_00295")
    parser.add_argument("--text", default="Hola, esta es una prueba simple del modelo")
    
    args = parser.parse_args()
    
    simple_test(args.model, args.speaker, args.text)