import os
import glob
import torch
import whisper
import pandas as pd
from tqdm import tqdm

class DatasetTranscriber:
    """
    Handles the automated transcription of a large audio dataset using OpenAI's Whisper model.
    It scans a directory structure, transcribes wav files, and saves metadata to a CSV.
    """

    def __init__(self, model_size="tiny", device=None):
        """
        Initializes the Whisper model.

        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Whisper ({model_size}) on device: {self.device}")
        
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe_dataset(self, root_path, output_csv="metadata.csv", save_interval=500):
        """
        Recursively scans the dataset, transcribes audio files, and saves progress periodically.

        Args:
            root_path (str): Root directory containing the audio files.
            output_csv (str): Path to save the resulting CSV file.
            save_interval (int): Number of files to process before saving a checkpoint CSV.

        Returns:
            pd.DataFrame: The final dataframe containing file paths, speaker IDs, and transcriptions.
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
                
                # Extract metadata based on folder structure: Real/Country/SpeakerID/audio.wav
                parts = wav_path.split(os.sep)
                speaker_id = parts[-2]
                
                rel_path = os.path.relpath(wav_path, start=os.getcwd())
                
                results.append({
                    "file_path": rel_path,
                    "speaker_id": speaker_id,
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

    def _save_to_csv(self, data, filepath):
        """Helper method to save the list of dictionaries to a pipe-separated CSV."""
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, sep="|")

def main():
    # Configuration
    dataset_root = os.path.join("Latin_America_Spanish_anti_spoofing_dataset", "FinalDataset_16khz", "Real")
    output_file = "metadata.csv"
    
    # Execution
    transcriber = DatasetTranscriber(model_size="tiny")
    
    try:
        transcriber.transcribe_dataset(dataset_root, output_file)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Partial results may have been saved.")

if __name__ == "__main__":
    main()