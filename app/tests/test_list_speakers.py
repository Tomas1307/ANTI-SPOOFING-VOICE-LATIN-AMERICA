"""
List Available Speakers

Utility script to show all available speaker IDs and their metadata.
"""

import torch
import pandas as pd
from pathlib import Path


def list_speakers(
    embeddings_path: str = "app/embedding/speaker_embeddings.pt",
    metadata_path: str = "app/embedding/metadata.csv"
):
    """
    List all available speakers with metadata.
    
    Args:
        embeddings_path: Path to speaker embeddings file.
        metadata_path: Path to metadata CSV.
    """
    print(f"\n{'='*70}")
    print("AVAILABLE SPEAKERS FOR VOICE CLONING")
    print(f"{'='*70}\n")
    
    # Load embeddings
    print(f"Loading embeddings from: {embeddings_path}")
    speaker_embeddings = torch.load(embeddings_path)
    
    print(f"✓ Found {len(speaker_embeddings)} speakers\n")
    
    # Try to load metadata
    if Path(metadata_path).exists():
        print(f"Loading metadata from: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        
        print(f"\nSpeakers by country:")
        country_map = {
            "ar": "Argentina",
            "cl": "Chile",
            "co": "Colombia",
            "pe": "Peru",
            "ve": "Venezuela"
        }
        
        # Count by country
        country_counts = {}
        for speaker_id in speaker_embeddings.keys():
            country_code = speaker_id[:2]
            country = country_map.get(country_code, "Unknown")
            country_counts[country] = country_counts.get(country, 0) + 1
        
        for country, count in sorted(country_counts.items()):
            print(f"  {country}: {count} speakers")
        
        # Show sample speakers
        print(f"\n{'='*70}")
        print("SAMPLE SPEAKERS (first 20):")
        print(f"{'='*70}\n")
        
        speaker_list = list(speaker_embeddings.keys())[:20]
        
        for i, speaker_id in enumerate(speaker_list, 1):
            country_code = speaker_id[:2]
            country = country_map.get(country_code, "Unknown")
            embedding_shape = speaker_embeddings[speaker_id].shape
            
            print(f"{i:2d}. {speaker_id:20s} | {country:12s} | Embedding: {embedding_shape}")
        
        if len(speaker_embeddings) > 20:
            print(f"\n... and {len(speaker_embeddings) - 20} more speakers")
        
        # Show how to use
        print(f"\n{'='*70}")
        print("USAGE EXAMPLES:")
        print(f"{'='*70}\n")
        
        sample_speaker = speaker_list[0]
        print(f"Test with a specific speaker:")
        print(f"  python test_voice_cloning_quick.py \\")
        print(f"    --model speech_tts_finetuned/checkpoint-5000 \\")
        print(f"    --speaker {sample_speaker} \\")
        print(f'    --text "Hola, esta es una prueba"')
        
        print(f"\nFull evaluation:")
        print(f"  python evaluate_voice_cloning.py \\")
        print(f"    --model speech_tts_finetuned/checkpoint-5000 \\")
        print(f"    --speakers {sample_speaker} {speaker_list[1] if len(speaker_list) > 1 else ''}")
        
    else:
        print(f"\n⚠ Metadata file not found: {metadata_path}")
        print(f"\nAll speaker IDs:")
        
        for i, speaker_id in enumerate(list(speaker_embeddings.keys())[:50], 1):
            print(f"  {i:2d}. {speaker_id}")
        
        if len(speaker_embeddings) > 50:
            print(f"  ... and {len(speaker_embeddings) - 50} more")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="List available speakers")
    
    parser.add_argument(
        "--embeddings",
        type=str,
        default="app/embedding/speaker_embeddings.pt",
        help="Path to speaker embeddings"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        default="app/embedding/metadata.csv",
        help="Path to metadata CSV"
    )
    
    args = parser.parse_args()
    
    list_speakers(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata
    )