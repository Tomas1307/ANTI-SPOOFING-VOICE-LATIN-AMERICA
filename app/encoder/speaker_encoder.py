import os
import glob
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

class SpeakerEncoder:
    """
    Handles the extraction of speaker embeddings using a pre-trained WavLM model.
    It computes the centroid (mean embedding) for each speaker to create a robust profile.
    """

    def __init__(self, model_name="microsoft/wavlm-base-plus-sv", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing SpeakerEncoder on device: {self.device}")
        
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.sample_rate = 16000

    def _process_audio(self, audio_path):
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            inputs = self.processor(
                audio, 
                return_tensors="pt", 
                sampling_rate=self.sample_rate, 
                padding=True
            )
            return inputs.to(self.device)
        except Exception:
            return None

    def compute_centroid(self, speaker_dir, max_samples=50):
        files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        if not files: return None

        selected_files = np.random.choice(files, max_samples, replace=False) if len(files) > max_samples else files
        embeddings = []

        with torch.no_grad():
            for filepath in selected_files:
                inputs = self._process_audio(filepath)
                if inputs is None: continue
                
                outputs = self.model(**inputs)
                embeddings.append(outputs.embeddings.squeeze().cpu())

        if not embeddings: return None

        centroid = torch.stack(embeddings).mean(dim=0)
        return torch.nn.functional.normalize(centroid, dim=0)

    def process_dataset(self, root_path, checkpoint_file="speaker_embeddings.pt"):
        """
        Walks through the dataset structure and generates embeddings.
        SAVES PROGRESS every 10 speakers.
        """
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Path not found: {root_path}")

        # --- 1. LOGIC TO RESUME ---
        if os.path.exists(checkpoint_file):
            print(f"Found checkpoint: {checkpoint_file}")
            print("Loading previous progress...")
            speaker_bank = torch.load(checkpoint_file)
            print(f"Resumed with {len(speaker_bank)} speakers already done.")
        else:
            speaker_bank = {}

        countries = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        
        # Counter for saving periodically
        processed_count = 0 

        for country in countries:
            country_path = os.path.join(root_path, country)
            speakers = [s for s in os.listdir(country_path) if os.path.isdir(os.path.join(country_path, s))]

            # Filter speakers that are already done
            pending_speakers = [s for s in speakers if s not in speaker_bank]
            
            if not pending_speakers:
                continue

            print(f"\nProcessing {country} ({len(pending_speakers)} new speakers)...")

            for speaker_id in tqdm(pending_speakers):
                speaker_path = os.path.join(country_path, speaker_id)
                embedding = self.compute_centroid(speaker_path)
                
                if embedding is not None:
                    speaker_bank[speaker_id] = embedding
                    processed_count += 1
                
                # --- 2. SAVE EVERY 10 SPEAKERS ---
                if processed_count % 10 == 0:
                    torch.save(speaker_bank, checkpoint_file)
            
            # Save at the end of each country just in case
            torch.save(speaker_bank, checkpoint_file)

        return speaker_bank

def main():
    # Configuration
    dataset_root = os.path.join("Latin_America_Spanish_anti_spoofing_dataset", "FinalDataset_16khz", "Real")
    output_path = "speaker_embeddings.pt"

    encoder = SpeakerEncoder()
    
    try:
        # Pass the output path as the checkpoint file
        embeddings = encoder.process_dataset(dataset_root, checkpoint_file=output_path)
        print(f"\nDONE! Final embeddings saved to {output_path}")
        print(f"Total speakers processed: {len(embeddings)}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress saved in checkpoint file.")

if __name__ == "__main__":
    main()