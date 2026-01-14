"""
List Available Speakers

Utility script to show all available speaker IDs.
Works without metadata.csv file.
"""

import torch
from pathlib import Path


def list_speakers(
    embeddings_path: str = "app/embedding/speaker_embeddings.pt"
):
    """
    List all available speakers.
    
    Args:
        embeddings_path: Path to speaker embeddings file.
    """
    print(f"\n{'='*70}")
    print("AVAILABLE SPEAKERS FOR VOICE CLONING")
    print(f"{'='*70}\n")
    
    # Load embeddings
    print(f"Loading embeddings from: {embeddings_path}")
    speaker_embeddings = torch.load(embeddings_path)
    
    print(f"✓ Found {len(speaker_embeddings)} speakers\n")
    
    # Count by country
    country_map = {
        "ar": "Argentina",
        "cl": "Chile",
        "co": "Colombia",
        "pe": "Peru",
        "ve": "Venezuela"
    }
    
    country_counts = {}
    speakers_by_country = {}
    
    for speaker_id in speaker_embeddings.keys():
        country_code = speaker_id[:2]
        country = country_map.get(country_code, "Unknown")
        
        country_counts[country] = country_counts.get(country, 0) + 1
        
        if country not in speakers_by_country:
            speakers_by_country[country] = []
        speakers_by_country[country].append(speaker_id)
    
    print(f"Speakers by country:")
    for country, count in sorted(country_counts.items()):
        print(f"  {country}: {count} speakers")
    
    # Show sample speakers
    print(f"\n{'='*70}")
    print("SAMPLE SPEAKERS (first 30):")
    print(f"{'='*70}\n")
    
    speaker_list = sorted(list(speaker_embeddings.keys()))[:30]
    
    for i, speaker_id in enumerate(speaker_list, 1):
        country_code = speaker_id[:2]
        country = country_map.get(country_code, "Unknown")
        embedding_shape = speaker_embeddings[speaker_id].shape
        
        print(f"{i:2d}. {speaker_id:20s} | {country:12s} | Embedding: {embedding_shape}")
    
    if len(speaker_embeddings) > 30:
        print(f"\n... and {len(speaker_embeddings) - 30} more speakers")
    
    # Show how to use
    print(f"\n{'='*70}")
    print("USAGE EXAMPLES:")
    print(f"{'='*70}\n")
    
    sample_speaker = speaker_list[0]

    
    print(f"\nFull evaluation:")
    print(f"  python evaluate_voice_cloning.py \\")
    print(f"    --model speech_tts_finetuned \\")
    print(f"    --speakers {sample_speaker}")
    
    if len(speaker_list) > 1:
        print(f" {speaker_list[1]}")
    if len(speaker_list) > 2:
        print(f" {speaker_list[2]}")
    
    print(f"\nCompare all checkpoints:")
    print(f"  python compare_checkpoints.py \\")
    print(f"    --speaker {sample_speaker}")
    
    print(f"\n{'='*70}")
    print(f"SPEAKERS BY COUNTRY (first 5 per country):")
    print(f"{'='*70}\n")
    
    for country in sorted(speakers_by_country.keys()):
        speakers = sorted(speakers_by_country[country])[:5]
        print(f"{country}:")
        for speaker in speakers:
            print(f"  - {speaker}")
        if len(speakers_by_country[country]) > 5:
            print(f"  ... and {len(speakers_by_country[country]) - 5} more")
        print()
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="List available speakers")
    
    parser.add_argument(
        "--embeddings",
        type=str,
        default="app/embedding/speaker_embeddings.pt",
        help="Path to speaker embeddings"
    )
    
    args = parser.parse_args()
    
    list_speakers(embeddings_path=args.embeddings)