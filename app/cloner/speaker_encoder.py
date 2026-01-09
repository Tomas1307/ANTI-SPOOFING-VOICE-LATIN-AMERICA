import os
import glob
import torch
import librosa
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


class SpeakerEncoder:
    """
    Speaker embedding extraction system using pre-trained WavLM models.
    
    This class provides functionality to generate robust speaker embeddings
    by computing centroid representations from multiple audio samples per speaker.
    Embeddings are normalized and suitable for speaker verification and 
    multi-speaker TTS applications.
    
    Attributes:
        device: Computation device for model inference (cuda or cpu).
        processor: Feature extractor for audio preprocessing.
        model: Pre-trained WavLM model for x-vector extraction.
        sample_rate: Target sampling rate for audio processing (16kHz).
    """

    def __init__(
        self, 
        model_name: str = "microsoft/wavlm-base-plus-sv", 
        device: Optional[str] = None
    ):
        """
        Initialize the speaker encoding system.

        Args:
            model_name: HuggingFace model identifier for the WavLM variant to use.
            device: Target computation device. If None, automatically selects 
                   CUDA if available, otherwise CPU.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing SpeakerEncoder on device: {self.device}")
        
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.sample_rate = 16000

    def _process_audio(self, audio_path: str) -> Optional[Dict]:
        """
        Load and preprocess a single audio file for model inference.
        
        Args:
            audio_path: Path to the audio file to process.
            
        Returns:
            Preprocessed audio tensor ready for model input, or None if processing fails.
        """
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            inputs = self.processor(
                audio, 
                return_tensors="pt", 
                sampling_rate=self.sample_rate, 
                padding=True
            )
            return inputs.to(self.device)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def compute_centroid(
        self, 
        speaker_dir: str, 
        max_samples: int = 50, 
        use_train_only: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Generate a centroid embedding for a speaker from multiple audio samples.
        
        This method computes individual embeddings for multiple audio files belonging
        to a speaker, then averages them to create a robust centroid representation.
        The centroid is normalized to unit length for consistent similarity computation.
        
        Args:
            speaker_dir: Path to speaker directory containing train/val/test subdirectories.
            max_samples: Maximum number of audio files to use for centroid computation.
            use_train_only: If True, only uses files from the train split to prevent
                          information leakage into validation/test sets.
        
        Returns:
            Normalized 512-dimensional embedding tensor, or None if no valid 
            audio files were found or processed successfully.
        """
        if use_train_only:
            train_dir = os.path.join(speaker_dir, "train")
            if os.path.exists(train_dir):
                files = glob.glob(os.path.join(train_dir, "*.wav"))
            else:
                print(f"Warning: No train directory found in {speaker_dir}")
                return None
        else:
            files = glob.glob(os.path.join(speaker_dir, "**", "*.wav"), recursive=True)
        
        if not files:
            print(f"Warning: No audio files found for speaker {os.path.basename(speaker_dir)}")
            return None

        if len(files) > max_samples:
            selected_files = np.random.choice(files, max_samples, replace=False)
        else:
            selected_files = files

        embeddings = []

        with torch.no_grad():
            for filepath in selected_files:
                inputs = self._process_audio(filepath)
                if inputs is None:
                    continue
                
                outputs = self.model(**inputs)
                embeddings.append(outputs.embeddings.squeeze().cpu())

        if not embeddings:
            print(f"Warning: No valid embeddings extracted for {os.path.basename(speaker_dir)}")
            return None

        centroid = torch.stack(embeddings).mean(dim=0)
        
        return torch.nn.functional.normalize(centroid, dim=0)

    def process_dataset(
        self, 
        root_path: str, 
        checkpoint_file: str = "speaker_embeddings.pt", 
        save_interval: int = 10, 
        use_train_only: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process entire dataset and generate embeddings for all speakers.
        
        This method walks through the partition_dataset_by_speaker directory structure,
        generates embeddings for each speaker, and periodically saves progress to enable
        resumption after interruption.
        
        The expected directory structure is:
        root_path/speaker_id/{train,val,test}/*.wav
        
        Args:
            root_path: Path to partition_dataset_by_speaker directory.
            checkpoint_file: Path where embeddings are saved. If this file exists,
                           previously processed speakers will be skipped.
            save_interval: Save progress after processing this many speakers.
            use_train_only: If True, only uses train split for embeddings to prevent
                          data leakage.
        
        Returns:
            Dictionary mapping speaker IDs to their embedding tensors.
            
        Raises:
            FileNotFoundError: If the specified root_path does not exist.
        """
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Path not found: {root_path}")

        if os.path.exists(checkpoint_file):
            print(f"Found checkpoint: {checkpoint_file}")
            print("Loading previous progress...")
            speaker_bank = torch.load(checkpoint_file, weights_only=False)
            print(f"Resumed with {len(speaker_bank)} speakers already done.")
        else:
            speaker_bank = {}

        all_speakers = [
            d for d in os.listdir(root_path) 
            if os.path.isdir(os.path.join(root_path, d))
        ]
        
        pending_speakers = [s for s in all_speakers if s not in speaker_bank]
        
        if not pending_speakers:
            print("All speakers already processed!")
            return speaker_bank
        
        print(f"\nFound {len(all_speakers)} total speakers")
        print(f"Already processed: {len(speaker_bank)}")
        print(f"Pending: {len(pending_speakers)}")
        print(f"Using {'TRAIN ONLY' if use_train_only else 'ALL SPLITS'} for embeddings\n")
        
        processed_count = 0

        speaker_groups = self._group_speakers_by_metadata(pending_speakers)
        
        for group_name, speakers in speaker_groups.items():
            if not speakers:
                continue
                
            print(f"\nProcessing {group_name} ({len(speakers)} speakers)...")
            
            for speaker_id in tqdm(speakers, desc=group_name):
                speaker_path = os.path.join(root_path, speaker_id)
                embedding = self.compute_centroid(
                    speaker_path, 
                    use_train_only=use_train_only
                )
                
                if embedding is not None:
                    speaker_bank[speaker_id] = embedding
                    processed_count += 1
                else:
                    print(f"Skipping {speaker_id} (no valid embedding)")
                
                if processed_count % save_interval == 0:
                    torch.save(speaker_bank, checkpoint_file)
                    print(f"  Checkpoint saved ({len(speaker_bank)} speakers total)")

        torch.save(speaker_bank, checkpoint_file)
        print(f"\nFinal save complete: {len(speaker_bank)} speakers")
        
        return speaker_bank

    def _group_speakers_by_metadata(self, speaker_ids: List[str]) -> Dict[str, List[str]]:
        """
        Organize speakers into groups based on country and gender metadata.
        
        This method parses speaker IDs to extract country and gender information,
        grouping speakers for organized processing and progress tracking.
        
        Args:
            speaker_ids: List of speaker IDs in format {country}{gender}_id.
        
        Returns:
            Dictionary mapping group names (e.g., "Argentina Female") to lists
            of speaker IDs belonging to that group.
        """
        country_map = {
            "ar": "Argentina",
            "cl": "Chile",
            "co": "Colombia",
            "pe": "Peru",
            "ve": "Venezuela"
        }
        
        gender_map = {
            "f": "Female",
            "m": "Male"
        }
        
        groups = {}
        
        for speaker_id in speaker_ids:
            if len(speaker_id) >= 3:
                country_code = speaker_id[:2]
                gender_code = speaker_id[2]
                
                country = country_map.get(country_code, "Unknown")
                gender = gender_map.get(gender_code, "Unknown")
                
                group_name = f"{country} {gender}"
            else:
                group_name = "Unknown"
            
            if group_name not in groups:
                groups[group_name] = []
            
            groups[group_name].append(speaker_id)
        
        return dict(sorted(groups.items()))

    def verify_embeddings(self, embeddings_path: str):
        """
        Validate and display statistics for a saved embeddings file.
        
        Args:
            embeddings_path: Path to the speaker_embeddings.pt file to verify.
        """
        if not os.path.exists(embeddings_path):
            print(f"Error: File not found: {embeddings_path}")
            return
        
        print(f"\n{'='*70}")
        print(f"VERIFYING EMBEDDINGS: {embeddings_path}")
        print(f"{'='*70}")
        
        embeddings = torch.load(embeddings_path, weights_only=False)
        
        print(f"\nTotal speakers: {len(embeddings)}")
        
        if embeddings:
            first_speaker = list(embeddings.keys())[0]
            first_embedding = embeddings[first_speaker]
            print(f"Embedding dimension: {first_embedding.shape}")
            print(f"Embedding dtype: {first_embedding.dtype}")
        
        groups = self._group_speakers_by_metadata(list(embeddings.keys()))
        
        print(f"\n{'Group':<25} {'Count':>10}")
        print("-"*70)
        for group_name, speakers in sorted(groups.items()):
            print(f"{group_name:<25} {len(speakers):>10}")
        
        print(f"{'='*70}\n")


def main():
    """
    Main entry point for speaker embedding generation.
    
    Configures embedding extraction parameters and executes the full pipeline
    with automatic checkpoint management for interrupt recovery.
    """
    dataset_root = os.path.join("data", "partition_dataset_by_speaker")
    output_path = "speaker_embeddings.pt"
    
    encoder = SpeakerEncoder()
    
    try:
        embeddings = encoder.process_dataset(
            dataset_root, 
            checkpoint_file=output_path,
            save_interval=5,
            use_train_only=True
        )
        
        print(f"\n{'='*70}")
        print("EMBEDDING GENERATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Output file: {output_path}")
        print(f"Total speakers: {len(embeddings)}")
        
        encoder.verify_embeddings(output_path)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        print(f"Progress saved in: {output_path}")
        print("You can resume by running the script again.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print(f"Partial progress may be saved in: {output_path}")
        raise


if __name__ == "__main__":
    main()