import os
import glob
import torch
import whisper
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional


class DatasetTranscriber:
    """
    Automated audio transcription system for large-scale dataset processing.
    
    This class provides functionality to recursively scan audio directories,
    transcribe speech content using OpenAI's Whisper model, and generate
    structured metadata CSV files with comprehensive speaker information.
    
    Attributes:
        device: Computation device for model inference (cuda or cpu).
        model: Loaded Whisper model instance for transcription.
    """

    def __init__(self, model_size: str = "tiny", device: Optional[str] = None):
        """
        Initialize the transcription system with specified model configuration.

        Args:
            model_size: Whisper model variant to use. Options are 'tiny', 'base', 
                       'small', 'medium', 'large'. Larger models provide better 
                       accuracy at the cost of speed.
            device: Target computation device. If None, automatically selects 
                   CUDA if available, otherwise CPU.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Whisper ({model_size}) on device: {self.device}")
        
        self.model = whisper.load_model(model_size, device=self.device)


    def transcribe_dataset(
        self, 
        root_path: str, 
        output_csv: str = "metadata.csv", 
        save_interval: int = 500
    ) -> pd.DataFrame:
        """
        Process all audio files in dataset and generate transcription metadata.
        
        This method recursively scans the specified directory structure, transcribes
        each audio file using Whisper, extracts speaker and split information from
        the directory hierarchy, and saves results periodically to prevent data loss.
        
        The expected directory structure is:
        root_path/speaker_id/{train,val,test}/*.wav
        
        Speaker IDs should follow the format: {country_code}{gender_code}_id
        where country_code is 'ar', 'cl', 'co', 'pe', or 've' and gender_code
        is 'f' or 'm'.
        
        Args:
            root_path: Root directory containing partition_dataset_by_speaker folder.
            output_csv: Destination path for the output CSV file.
            save_interval: Number of files to process before saving a checkpoint.
            
        Returns:
            DataFrame containing all transcription results with columns:
            file_path, speaker_id, country, gender, split, text.
            
        Raises:
            FileNotFoundError: If no WAV files are found in the specified path.
        """
        wav_files = glob.glob(os.path.join(root_path, "**", "*.wav"), recursive=True)
        total_files = len(wav_files)
        
        if total_files == 0:
            raise FileNotFoundError(f"No .wav files found in {root_path}")

        print(f"Found {total_files} files. Starting transcription...")
        
        results = []
        
        for i, wav_path in enumerate(tqdm(wav_files, desc="Transcribing")):
            try:
                result = self.model.transcribe(wav_path, language="es")
                text = result["text"].strip()
                
                parts = wav_path.split(os.sep)
                
                speaker_id = None
                split_type = None
                
                for idx, part in enumerate(parts):
                    if part in ["train", "test", "val"]:
                        speaker_id = parts[idx - 1]
                        split_type = part
                        break
                
                if not speaker_id:
                    print(f"\nWarning: Could not extract speaker_id from {wav_path}")
                    continue
                
                country_code = speaker_id[:2]
                gender_code = speaker_id[2]
                
                country_map = {
                    "ar": "Argentina",
                    "cl": "Chile", 
                    "co": "Colombia",
                    "pe": "Peru",
                    "ve": "Venezuela"
                }
                
                gender_map = {
                    "f": "female",
                    "m": "male"
                }
                
                country = country_map.get(country_code, "unknown")
                gender = gender_map.get(gender_code, "unknown")
                
                rel_path = os.path.relpath(wav_path, start=os.getcwd())
                
                results.append({
                    "file_path": rel_path,
                    "speaker_id": speaker_id,
                    "country": country,
                    "gender": gender,
                    "split": split_type,
                    "text": text
                })
                
                if i > 0 and i % save_interval == 0:
                    self._save_to_csv(results, output_csv)
                    
            except Exception as e:
                print(f"\nError processing {os.path.basename(wav_path)}: {e}")
                continue

        self._save_to_csv(results, output_csv)
        print(f"\nTranscription complete. Metadata saved to {output_csv}")
        
        return pd.DataFrame(results)

    def _save_to_csv(self, data: List[Dict], filepath: str):
        """
        Persist transcription results to pipe-separated CSV file.
        
        Args:
            data: List of dictionaries containing transcription results.
            filepath: Destination path for CSV output.
        """
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, sep="|")



def main():
    """
    Main entry point for dataset transcription.
    
    Configures transcription parameters and executes the full transcription
    pipeline with automatic checkpoint saving for interrupt recovery.
    """
    dataset_root = os.path.join("data", "partition_dataset_by_speaker")
    output_file = "metadata.csv"
    
    transcriber = DatasetTranscriber(model_size="tiny")
    
    try:
        transcriber.transcribe_dataset(dataset_root, output_file)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Partial results may have been saved.")
    except Exception as e:
        print(f"\nError during transcription: {e}")
        raise


if __name__ == "__main__":
    main()