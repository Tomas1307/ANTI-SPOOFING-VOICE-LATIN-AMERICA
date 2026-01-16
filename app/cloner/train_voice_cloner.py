import os
import glob
import torch
import pandas as pd
import soundfile as sf
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from datasets import Dataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from app.schema import TTSDataCollatorWithPadding

# SOTA Evaluation imports
from transformers import SpeechT5HifiGan

from app.cloner.sota_evaluation_callback import create_sota_callback
SOTA_CALLBACK_AVAILABLE = True



class CheckpointResumptionCallback(TrainerCallback):
    """
    Callback to handle proper checkpoint resumption and state management.
    
    This callback ensures that training can be safely resumed from the last
    checkpoint by managing state persistence and recovery.
    """
    
    def on_save(self, args, state, control, **kwargs):
        """
        Called when a checkpoint is saved.
        
        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow object.
        """
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        
        if os.path.exists(state_path):
            print(f"Checkpoint saved at step {state.global_step}")


class VoiceClonerTrainer:
    """
    Manages the complete training pipeline for multi-speaker text-to-speech models.
    
    This class handles data loading, preprocessing, model initialization, and training
    with automatic checkpoint resumption capabilities. It supports both random train-test
    splitting and pre-existing data partitions.
    
    Attributes:
        metadata_path: Path to CSV file containing transcriptions and metadata.
        output_dir: Directory where model checkpoints and final model will be saved.
        use_presplit: Whether to use pre-existing train/val/test splits from metadata.
        device: Computation device (cuda or cpu).
        speaker_embeddings: Dictionary mapping speaker IDs to embedding tensors.
        processor: SpeechT5 processor for text and audio processing.
        model: SpeechT5 model instance for training.
    """

    def __init__(
        self, 
        metadata_path: str, 
        embeddings_path: str, 
        output_dir: str, 
        model_checkpoint: str = "microsoft/speecht5_tts",
        use_presplit: bool = True
    ):
        """
        Initialize the voice cloning trainer.
        
        Args:
            metadata_path: Path to metadata CSV with transcriptions.
            embeddings_path: Path to speaker embeddings .pt file.
            output_dir: Directory to save the fine-tuned model.
            model_checkpoint: Pretrained model to start from.
            use_presplit: If True, uses the 'split' column from metadata instead of random splitting.
        """
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.use_presplit = use_presplit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading resources on {self.device}...")
        self.speaker_embeddings = torch.load(
            embeddings_path, 
            map_location=self.device, 
            weights_only=False
        )
        self.processor = SpeechT5Processor.from_pretrained(model_checkpoint)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_checkpoint)
        
        # Load HiFi-GAN vocoder for evaluation callback
        print("Loading HiFi-GAN vocoder for evaluation...")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        if torch.cuda.is_available():
            self.vocoder = self.vocoder.to(self.device)
        
        self._print_dataset_stats()
        
        self.checkpoint_path = self._find_latest_checkpoint()
        if self.checkpoint_path:
            print(f"Found existing checkpoint: {self.checkpoint_path}")
            print("Training will resume from this checkpoint.")

    def _find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the most recent checkpoint in the output directory.
        
        Returns:
            Path to the latest checkpoint directory, or None if no checkpoints exist.
        """
        if not os.path.exists(self.output_dir):
            return None
        
        checkpoints = glob.glob(os.path.join(self.output_dir, "checkpoint-*"))
        if not checkpoints:
            return None
        
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda x: int(x.split("-")[-1])
        )
        
        latest = checkpoints_sorted[-1]
        return latest

    def _print_dataset_stats(self):
        """
        Print comprehensive statistics about the dataset for verification.
        
        Displays information about total samples, split distribution, country
        distribution, gender distribution, and unique speakers.
        """
        if not os.path.exists(self.metadata_path):
            print(f"Warning: Metadata file not found at {self.metadata_path}")
            return
            
        df = pd.read_csv(self.metadata_path, sep="|")
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total samples: {len(df)}")
        
        if 'split' in df.columns:
            print("\nSamples per split:")
            print(df['split'].value_counts().to_string())
        
        if 'country' in df.columns:
            print("\nSamples per country:")
            print(df['country'].value_counts().to_string())
        
        if 'gender' in df.columns:
            print("\nSamples per gender:")
            print(df['gender'].value_counts().to_string())
        
        print(f"\nUnique speakers: {df['speaker_id'].nunique()}")
        print("="*60 + "\n")

    def _prepare_example(self, example: Dict) -> Optional[Dict]:
        """
        Transform a single dataset example into model-compatible format.
        
        This method loads audio files, resamples if necessary, and processes both
        text and audio through the SpeechT5 processor. Returns Python lists instead
        of tensors for compatibility with HuggingFace Datasets multiprocessing.
        
        Args:
            example: Dictionary containing audio_path, text, and speaker_id.
            
        Returns:
            Dictionary with input_ids, labels, and speaker_embeddings as lists,
            or None if processing fails.
        """
        audio_path = example["audio_path"]
        
        try:
            speech_array, sampling_rate = sf.read(audio_path)
        except Exception:
            try:
                speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                return None

        if sampling_rate != 16000:
            speech_array = librosa.resample(
                speech_array, 
                orig_sr=sampling_rate, 
                target_sr=16000
            )
        
        inputs = self.processor(
            text=example["text"],
            audio_target=speech_array,
            sampling_rate=16000,
            return_tensors="pt"
        )

        speaker_id = example["speaker_id"]
        if speaker_id in self.speaker_embeddings:
            speaker_emb = self.speaker_embeddings[speaker_id]
        else:
            print(f"Warning: No embedding found for speaker {speaker_id}, using zeros")
            speaker_emb = torch.zeros(512)

        return {
            "input_ids": inputs["input_ids"][0].tolist(),
            "labels": inputs["labels"][0].tolist(),
            "speaker_embeddings": speaker_emb.tolist()
        }

    def load_and_process_data(
        self, 
        test_size: float = 0.1, 
        val_size: float = 0.1
    ) -> Dict[str, Dataset]:
        """
        Load metadata and process audio files into model-ready format.
        
        This method handles both pre-split datasets and random splitting. It loads
        the metadata CSV, processes all audio files through the SpeechT5 processor,
        and returns train/validation/test datasets.
        
        Args:
            test_size: Proportion for test set (only used if use_presplit=False).
            val_size: Proportion for validation set (only used if use_presplit=False).
            
        Returns:
            Dictionary containing 'train', 'validation', and 'test' Dataset objects.
        """
        print(f"Reading metadata from {self.metadata_path}...")
        df = pd.read_csv(self.metadata_path, sep="|")
        df = df.dropna(subset=["text"])
        
        if "file_path" in df.columns:
            df = df.rename(columns={"file_path": "audio_path"})
        
        if self.use_presplit and "split" in df.columns:
            print("Using pre-existing train/test/val splits from metadata...")
            
            train_df = df[df["split"] == "train"].reset_index(drop=True)
            test_df = df[df["split"] == "test"].reset_index(drop=True)
            val_df = df[df["split"] == "val"].reset_index(drop=True)
            
            print(f"Train samples: {len(train_df)}")
            print(f"Test samples: {len(test_df)}")
            print(f"Validation samples: {len(val_df)}")
            
            train_dataset = Dataset.from_pandas(train_df)
            test_dataset = Dataset.from_pandas(test_df)
            val_dataset = Dataset.from_pandas(val_df)
            
            print("Tokenizing and processing audio for train split...")
            train_dataset = train_dataset.map(
                self._prepare_example,
                remove_columns=train_dataset.column_names,
                num_proc=None
            )
            
            print("Tokenizing and processing audio for test split...")
            test_dataset = test_dataset.map(
                self._prepare_example,
                remove_columns=test_dataset.column_names,
                num_proc=None
            )
            
            print("Tokenizing and processing audio for validation split...")
            val_dataset = val_dataset.map(
                self._prepare_example,
                remove_columns=val_dataset.column_names,
                num_proc=None
            )
            
            return {
                "train": train_dataset,
                "test": test_dataset,
                "validation": val_dataset
            }
            
        else:
            print("Creating random train/test split...")
            dataset = Dataset.from_pandas(df)
            
            dataset_split = dataset.train_test_split(test_size=test_size, seed=42)
            
            train_val_split = dataset_split["train"].train_test_split(
                test_size=val_size / (1 - test_size),
                seed=42
            )
            
            print("Tokenizing and processing audio...")
            train_dataset = train_val_split["train"].map(
                self._prepare_example,
                remove_columns=train_val_split["train"].column_names,
                num_proc=None
            )
            
            val_dataset = train_val_split["test"].map(
                self._prepare_example,
                remove_columns=train_val_split["test"].column_names,
                num_proc=None
            )
            
            test_dataset = dataset_split["test"].map(
                self._prepare_example,
                remove_columns=dataset_split["test"].column_names,
                num_proc=None
            )
            
            return {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset
            }

    def train(
        self, 
        dataset: Dict[str, Dataset], 
        batch_size: int = 8, 
        max_steps: int = 4000,
        learning_rate: float = 1e-5,
        warmup_steps: int = 500,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 50
    ):
        """
        Execute the training loop with automatic checkpoint management.
        
        This method configures training arguments, initializes the trainer with
        checkpoint resumption support, and executes the training loop. If a checkpoint
        exists in the output directory, training will automatically resume from that point.
        
        Args:
            dataset: Dictionary with 'train', 'validation', and optionally 'test' splits.
            batch_size: Batch size per device.
            max_steps: Maximum number of training steps.
            learning_rate: Learning rate for optimization.
            warmup_steps: Number of warmup steps for learning rate scheduler.
            save_steps: Save checkpoint every N steps.
            eval_steps: Run evaluation every N steps.
            logging_steps: Log metrics every N steps.
        """
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            gradient_checkpointing=False,
            fp16=torch.cuda.is_available(),
            eval_strategy="steps",
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            load_best_model_at_end=False, greater_is_better=True,
            metric_for_best_model="eval_samples_per_second",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            predict_with_generate=True,
            report_to=["tensorboard"],
            resume_from_checkpoint=self.checkpoint_path,
        )

        eval_dataset = dataset.get("validation", dataset.get("test"))

        # Create callbacks list
        callbacks = [CheckpointResumptionCallback()]
        
        # Add SOTA evaluation callback if available
        if SOTA_CALLBACK_AVAILABLE:
            print("\n" + "="*70)
            print("INITIALIZING SOTA EVALUATION METRICS")
            print("="*70)
            print("Metrics: DNSMOS P.835, Speaker Similarity (ECAPA-TDNN)")
            print(f"Evaluation frequency: Every 1000 steps")
            print(f"Output: training_evaluation_sota/")
            print("="*70 + "\n")
            
            try:
                sota_callback = create_sota_callback(
                    processor=self.processor,
                    vocoder=self.vocoder,
                    speaker_embeddings_path=self.metadata_path.replace("metadata.csv", "speaker_embeddings.pt"),
                    eval_every_n_steps=1000
                )
                callbacks.append(sota_callback)
                print("✓ SOTA evaluation callback integrated successfully\n")
            except Exception as e:
                print(f"⚠️  Could not initialize SOTA callback: {e}")
                print("   Continuing with basic training...\n")

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            tokenizer=self.processor,
            data_collator=TTSDataCollatorWithPadding(processor=self.processor),
            callbacks=callbacks
        )

        if self.checkpoint_path:
            print(f"Resuming training from checkpoint: {self.checkpoint_path}")
        else:
            print("Starting training from scratch...")
            
        trainer.train(resume_from_checkpoint=self.checkpoint_path)
        
        print(f"Saving final model to {self.output_dir}...")
        trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)



def main():
    """
    Main entry point for voice cloning model training.
    
    Initializes the trainer, loads and processes the dataset, and executes
    the training loop with automatic checkpoint resumption if available.
    """
    metadata_file = "./metadata.csv"
    embeddings_file = "./speaker_embeddings.pt"
    output_model_dir = "speech_tts_finetuned"

    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found at {metadata_file}")
        print("Please run transcription script first.")
        return
    
    if not os.path.exists(embeddings_file):
        print(f"Error: Speaker embeddings file not found at {embeddings_file}")
        print("Please run speaker encoder script first.")
        return

    try:
        trainer = VoiceClonerTrainer(
            metadata_file, 
            embeddings_file, 
            output_model_dir,
            use_presplit=True
        )
        
        processed_data = trainer.load_and_process_data()
        
        trainer.train(
            processed_data, 
            batch_size=8, 
            max_steps=30000,  
            learning_rate=1e-5,
            warmup_steps=500,
            save_steps=1000,  
            eval_steps=1000,  
            logging_steps=50
        )
        
        print("\nTraining completed successfully.")
        print(f"Model saved to: {output_model_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Progress has been saved. Resume by running this script again.")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Check logs for details.")
        raise


if __name__ == "__main__":
    main()