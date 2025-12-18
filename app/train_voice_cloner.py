import os
import torch
import pandas as pd
import soundfile as sf
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

@dataclass
class TTSDataCollatorWithPadding:
    """
    Data collator that handles dynamic padding for both text (input_ids) and audio (labels),
    and stacks speaker embeddings for batch processing.
    
    Critically handles the reduction_factor from SpeechT5 config by rounding down
    label lengths to multiples of the reduction factor.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input features
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        speaker_features = [torch.tensor(feature["speaker_embeddings"]) for feature in features]
        label_features = [torch.tensor(feature["labels"]) for feature in features]

        # Pad text inputs
        batch = self.processor.pad(input_ids=input_ids, return_tensors="pt")

        # CRITICAL: Round down target lengths to multiple of reduction factor
        # This prevents dimension mismatch errors during loss computation
        reduction_factor = 2  # SpeechT5 default reduction_factor
        
        # Calculate target lengths and round down to multiples of reduction_factor
        target_lengths = torch.tensor([label.shape[0] for label in label_features])
        if reduction_factor > 1:
            target_lengths = target_lengths.new(
                [length - length % reduction_factor for length in target_lengths]
            )
        
        # Truncate labels to their target lengths (before padding)
        label_features = [label[:target_lengths[i]] for i, label in enumerate(label_features)]
        
        # Now pad the truncated labels
        max_label_length = max(label.shape[0] for label in label_features)
        feature_dim = label_features[0].shape[1]  # Should be 80 for mel spectrograms
        
        batch_size = len(label_features)
        padded_labels = torch.zeros(batch_size, max_label_length, feature_dim)
        
        # Create attention mask for labels
        labels_attention_mask = torch.zeros(batch_size, max_label_length, dtype=torch.long)
        
        # Fill in the actual values
        for i, label in enumerate(label_features):
            length = label.shape[0]
            padded_labels[i, :length, :] = label
            labels_attention_mask[i, :length] = 1
        
        # Replace padding with -100 to ignore loss correctly
        labels = padded_labels.masked_fill(
            labels_attention_mask.unsqueeze(-1).ne(1), -100
        )

        batch["labels"] = labels
        batch["speaker_embeddings"] = torch.stack(speaker_features)

        return batch

class VoiceClonerTrainer:
    """
    Manages the fine-tuning lifecycle of a Multi-Speaker SpeechT5 model.
    """

    def __init__(self, metadata_path: str, embeddings_path: str, output_dir: str, model_checkpoint="microsoft/speecht5_tts"):
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading resources on {self.device}...")
        self.speaker_embeddings = torch.load(embeddings_path, map_location=self.device, weights_only=False)
        self.processor = SpeechT5Processor.from_pretrained(model_checkpoint)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_checkpoint)

    def _prepare_example(self, example: Dict) -> Dict:
        """
        Maps a single dataset example to model inputs.
        
        IMPORTANT: Returns Python lists (not tensors) because HuggingFace Datasets
        with num_proc > 1 serializes data using Apache Arrow, which doesn't support
        PyTorch tensors. Tensors are created later in the data collator.
        """
        audio_path = example["audio_path"]
        
        try:
            speech_array, sampling_rate = sf.read(audio_path)
        except Exception:
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

        if sampling_rate != 16000:
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        
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
            speaker_emb = torch.zeros(512)

        # Convert tensors to lists for serialization compatibility
        # The data collator will convert them back to tensors
        return {
            "input_ids": inputs["input_ids"][0].tolist(),
            "labels": inputs["labels"][0].tolist(),
            "speaker_embeddings": speaker_emb.tolist()
        }

    def load_and_process_data(self, test_size=0.1):
        print(f"Reading metadata from {self.metadata_path}...")
        df = pd.read_csv(self.metadata_path, sep="|")
        df = df.dropna(subset=["text"])
        
        dataset = Dataset.from_pandas(df)
        dataset = dataset.rename_column("file_path", "audio_path")
        
        dataset = dataset.train_test_split(test_size=test_size)
        
        print("Tokenizing and processing audio...")
        tokenized_datasets = dataset.map(
            self._prepare_example,
            remove_columns=dataset["train"].column_names,
            num_proc=4 
        )
        
        return tokenized_datasets

    def train(self, dataset, batch_size=8, max_steps=4000):
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=max_steps,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            eval_strategy="steps",
            save_steps=500,
            eval_steps=500,
            logging_steps=50,
            load_best_model_at_end=True,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            predict_with_generate=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.processor,
            data_collator=TTSDataCollatorWithPadding(processor=self.processor),
        )

        print("Starting training...")
        trainer.train()
        
        print(f"Saving model to {self.output_dir}...")
        trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)

def main():
    metadata_file = "./app/embedding/metadata.csv"
    embeddings_file = "./app/embedding/speaker_embeddings.pt"
    output_model_dir = "speech_tts_finetuned"

    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found at {metadata_file}")
        return

    trainer = VoiceClonerTrainer(metadata_file, embeddings_file, output_model_dir)
    processed_data = trainer.load_and_process_data()
    trainer.train(processed_data, batch_size=8, max_steps=5000)

if __name__ == "__main__":
    main()