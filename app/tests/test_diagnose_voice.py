"""
Voice Cloning Diagnostic Script

Performs detailed analysis to identify why audio is silent or empty.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


def diagnose_model(
    model_path: str = "speech_tts_finetuned",
    speaker_id: str = "arf_00295",
    text: str = "Hola mundo"
):
    """
    Diagnose voice cloning model issues.
    
    Args:
        model_path: Path to model.
        speaker_id: Speaker ID.
        text: Test text.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print("VOICE CLONING DIAGNOSTIC")
    print(f"{'='*70}\n")
    
    # 1. Load model
    print("1. Loading model...")
    try:
        processor = SpeechT5Processor.from_pretrained(model_path)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_path).to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        print("   ✓ Model loaded\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return
    
    # 2. Check tokenization
    print(f"2. Testing tokenization for: '{text}'")
    try:
        inputs = processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Input IDs: {input_ids[0].tolist()[:20]}...")
        print(f"   Num tokens: {input_ids.shape[1]}")
        
        # Decode back
        decoded = processor.tokenizer.decode(input_ids[0])
        print(f"   Decoded: '{decoded}'")
        
        if input_ids.shape[1] < 2:
            print("   ⚠️  WARNING: Very few tokens! Text might not be processed correctly.")
        else:
            print("   ✓ Tokenization looks good\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return
    
    # 3. Load embeddings
    print(f"3. Loading speaker embedding for '{speaker_id}'")
    try:
        embeddings = torch.load("app/embedding/speaker_embeddings.pt")
        
        if speaker_id not in embeddings:
            print(f"   ✗ Speaker not found!")
            print(f"   Available speakers: {list(embeddings.keys())[:10]}...")
            return
        
        speaker_emb = embeddings[speaker_id].to(device)
        print(f"   Embedding shape: {speaker_emb.shape}")
        print(f"   Embedding range: [{speaker_emb.min():.4f}, {speaker_emb.max():.4f}]")
        print(f"   Embedding mean: {speaker_emb.mean():.4f}")
        print(f"   Embedding std: {speaker_emb.std():.4f}")
        
        if speaker_emb.std() < 0.01:
            print("   ⚠️  WARNING: Embedding has very low variance!")
        else:
            print("   ✓ Embedding looks valid\n")
            
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return
    
    # 4. Generate spectrogram
    print("4. Generating spectrogram...")
    try:
        with torch.no_grad():
            input_ids_gpu = input_ids.to(device)
            spec = model.generate_speech(input_ids_gpu, speaker_emb.unsqueeze(0))
        
        print(f"   Spectrogram shape: {spec.shape}")
        print(f"   Spectrogram range: [{spec.min():.4f}, {spec.max():.4f}]")
        print(f"   Spectrogram mean: {spec.mean():.4f}")
        print(f"   Spectrogram std: {spec.std():.4f}")
        
        if spec.abs().max() < 0.01:
            print("   ⚠️  WARNING: Spectrogram has very low energy!")
            print("   This usually means the model didn't learn properly.")
        else:
            print("   ✓ Spectrogram generated\n")
            
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return
    
    # 5. Convert to waveform
    print("5. Converting to waveform...")
    try:
        with torch.no_grad():
            wave = vocoder(spec)
        
        audio = wave.cpu().squeeze().numpy()
        
        print(f"   Audio shape: {audio.shape}")
        print(f"   Audio length: {len(audio) / 16000:.2f}s")
        print(f"   Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
        print(f"   Audio RMS: {np.sqrt(np.mean(audio**2)):.6f}")
        print(f"   Non-zero samples: {np.count_nonzero(audio)}/{len(audio)} ({np.count_nonzero(audio)/len(audio)*100:.1f}%)")
        
        if np.count_nonzero(audio) < len(audio) * 0.1:
            print("   ⚠️  WARNING: Most samples are zero! Audio is likely silent.")
        elif audio.std() < 0.001:
            print("   ⚠️  WARNING: Very low variance! Audio might be silent or constant.")
        else:
            print("   ✓ Waveform generated\n")
            
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
        return
    
    # 6. Save and analyze
    print("6. Saving audio...")
    try:
        output = "diagnostic_output.wav"
        sf.write(output, audio, 16000)
        print(f"   ✓ Saved to: {output}\n")
        
        # Re-read to verify
        audio_read, sr = sf.read(output)
        print(f"   Verification:")
        print(f"     Read shape: {audio_read.shape}")
        print(f"     Read range: [{audio_read.min():.6f}, {audio_read.max():.6f}]")
        
    except Exception as e:
        print(f"   ✗ Error saving: {e}\n")
    
    # 7. Final diagnosis
    print(f"{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}\n")
    
    issues = []
    
    if input_ids.shape[1] < 2:
        issues.append("❌ Tokenization issue: Text not properly encoded")
    
    if speaker_emb.std() < 0.01:
        issues.append("❌ Speaker embedding issue: Low variance (possibly zeros)")
    
    if spec.abs().max() < 0.01:
        issues.append("❌ Model issue: Spectrogram has very low energy")
        issues.append("   → Model likely not trained properly")
    
    if audio.std() < 0.001:
        issues.append("❌ Vocoder issue: Output is nearly silent")
    
    if np.count_nonzero(audio) < len(audio) * 0.1:
        issues.append("❌ Audio issue: Most samples are zero")
    
    if not issues:
        print("✅ No obvious issues detected!")
        print("   If audio is still silent, check:")
        print("   - Volume settings")
        print("   - Audio player")
        print("   - File permissions")
    else:
        print("Issues detected:\n")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n💡 Recommendations:")
        
        if any("Model" in issue for issue in issues):
            print("   1. Check training logs - did loss decrease?")
            print("   2. Try a different checkpoint (e.g., checkpoint-5000)")
            print("   3. Verify training completed successfully")
        
        if any("embedding" in issue for issue in issues):
            print("   1. Regenerate speaker embeddings")
            print("   2. Check if embeddings file is corrupted")
            print("   3. Try a different speaker")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="speech_tts_finetuned")
    parser.add_argument("--speaker", default="arf_00295")
    parser.add_argument("--text", default="Hola mundo")
    
    args = parser.parse_args()
    
    diagnose_model(args.model, args.speaker, args.text)