"""
Quick Voice Cloning Test

Simple script to quickly test the fine-tuned model with a single text and speaker.
"""

import torch
import torchaudio
from pathlib import Path
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


def test_voice_cloning(
    text: str,
    speaker_id: str,
    model_path: str,
    embeddings_path: str = "app/embedding/speaker_embeddings.pt",
    output_path: str = "test_output.wav"
):
    """
    Quick test of voice cloning model.
    
    Args:
        text: Text to synthesize.
        speaker_id: Target speaker ID.
        model_path: Path to fine-tuned model.
        embeddings_path: Path to speaker embeddings.
        output_path: Where to save audio.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print("VOICE CLONING QUICK TEST")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Speaker: {speaker_id}")
    print(f"Text: {text}")
    print()
    
    # Load model
    print("Loading model...")
    processor = SpeechT5Processor.from_pretrained(model_path)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_path).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    print("✓ Model loaded")
    
    # Load speaker embeddings
    print("Loading speaker embeddings...")
    speaker_embeddings = torch.load(embeddings_path)
    
    if speaker_id not in speaker_embeddings:
        available = list(speaker_embeddings.keys())[:10]
        print(f"\n✗ Error: Speaker '{speaker_id}' not found!")
        print(f"Available speakers: {available}...")
        print(f"Total speakers: {len(speaker_embeddings)}")
        return
    
    speaker_embedding = speaker_embeddings[speaker_id].to(device)
    print(f"✓ Speaker embedding loaded ({len(speaker_embeddings)} total speakers)")
    
    # Synthesize
    print("\nSynthesizing speech...")
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    import time
    start = time.time()
    
    with torch.no_grad():
        spectrogram = model.generate_speech(
            input_ids,
            speaker_embedding.unsqueeze(0)
        )
        waveform = vocoder(spectrogram)
    
    duration = time.time() - start
    
    # Save audio
    audio = waveform.cpu().squeeze()
    torchaudio.save(output_path, audio.unsqueeze(0), 16000)
    
    audio_length = len(audio) / 16000
    rtf = duration / audio_length
    
    print(f"✓ Speech synthesized!")
    print(f"\nResults:")
    print(f"  Generation time: {duration:.2f}s")
    print(f"  Audio length: {audio_length:.2f}s")
    print(f"  Real-time factor: {rtf:.2f}x")
    print(f"  Output saved: {output_path}")
    
    if rtf < 1.0:
        print(f"  ✓ Faster than real-time!")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick voice cloning test")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., speech_tts_finetuned/checkpoint-5000)"
    )
    
    parser.add_argument(
        "--speaker",
        type=str,
        required=True,
        help="Speaker ID to use"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default="Hola, esta es una prueba del sistema de clonación de voz.",
        help="Text to synthesize"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="test_output.wav",
        help="Output audio file"
    )
    
    parser.add_argument(
        "--embeddings",
        type=str,
        default="app/embedding/speaker_embeddings.pt",
        help="Path to speaker embeddings"
    )
    
    args = parser.parse_args()
    
    test_voice_cloning(
        text=args.text,
        speaker_id=args.speaker,
        model_path=args.model,
        embeddings_path=args.embeddings,
        output_path=args.output
    )