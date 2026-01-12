from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import torch

class PipelineValidator:
    """
    Validation system for pipeline prerequisites and data integrity.
    
    Provides methods to verify directory structure, file existence, and
    data quality before executing pipeline stages.
    """
    
    def __init__(self):
        self.dataset_root = Path("data/partition_dataset_by_speaker")
        self.embeddings_dir = Path("app/embedding")
        self.metadata_file = self.embeddings_dir / "metadata.csv"
        self.embeddings_file = self.embeddings_dir / "speaker_embeddings.pt"
        self.output_dir = Path("speech_tts_finetuned")
    
    def validate_dataset_structure(self) -> Tuple[bool, Optional[str]]:
        """
        Verify dataset directory structure and speaker organization.
        
        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if not self.dataset_root.exists():
            return False, f"Dataset not found: {self.dataset_root}"
        
        speakers = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        
        if not speakers:
            return False, f"No speakers found in {self.dataset_root}"
        
        sample_speaker = speakers[0]
        required_splits = ["train", "val", "test"]
        missing_splits = []
        
        for split in required_splits:
            if not (sample_speaker / split).exists():
                missing_splits.append(split)
        
        if missing_splits:
            return False, f"Speaker {sample_speaker.name} missing splits: {missing_splits}"
        
        return True, None
    
    def validate_metadata(self) -> Tuple[bool, Optional[str]]:
        """
        Verify metadata.csv exists and has required columns.
        
        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if not self.metadata_file.exists():
            return False, f"Metadata not found: {self.metadata_file}"
        
        try:
            
            df = pd.read_csv(self.metadata_file, sep="|")
            
            required_columns = ["file_path", "speaker_id", "split", "text"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return False, f"Metadata missing columns: {missing_columns}"
            
            if df.empty:
                return False, "Metadata file is empty"
            
            return True, None
            
        except Exception as e:
            return False, f"Error reading metadata: {e}"
    
    def validate_embeddings(self) -> Tuple[bool, Optional[str]]:
        """
        Verify speaker_embeddings.pt exists and has valid structure.
        
        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if not self.embeddings_file.exists():
            return False, f"Embeddings not found: {self.embeddings_file}"
        
        try:
            
            embeddings = torch.load(self.embeddings_file, weights_only=False)
            
            if not embeddings:
                return False, "Embeddings file is empty"
            
            first_speaker = list(embeddings.keys())[0]
            first_embedding = embeddings[first_speaker]
            
            if first_embedding.shape[0] != 512:
                return False, f"Invalid embedding dimension: {first_embedding.shape}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error loading embeddings: {e}"
