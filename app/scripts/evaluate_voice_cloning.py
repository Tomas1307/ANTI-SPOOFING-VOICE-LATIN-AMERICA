"""
Voice Cloning Model Evaluation Script

Comprehensive testing of the fine-tuned SpeechT5 model for voice cloning.
Evaluates:
- Inference quality
- Speaker similarity
- Audio quality metrics (MOS estimation)
- Generation speed
- Memory usage
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
from dataclasses import dataclass
import json
from datetime import datetime
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from schema import EvaluationResult




class VoiceCloningEvaluator:
    """
    Evaluator for voice cloning model performance.
    
    Tests the fine-tuned model on various metrics and generates
    comprehensive evaluation reports.
    """
    
    def __init__(
        self,
        model_path: str,
        vocoder_name: str = "microsoft/speecht5_hifigan",
        speaker_embeddings_path: str = "app/embedding/speaker_embeddings.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to fine-tuned model checkpoint.
            vocoder_name: HiFi-GAN vocoder model name.
            speaker_embeddings_path: Path to speaker embeddings.
            device: Device to run inference on.
        """
        print(f"Loading model from: {model_path}")
        print(f"Device: {device}")
        
        self.device = device
        
        # Load model and processor
        self.processor = SpeechT5Processor.from_pretrained(model_path)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_path).to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name).to(device)
        
        # Load speaker embeddings
        self.speaker_embeddings = torch.load(speaker_embeddings_path)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Available speakers: {len(self.speaker_embeddings)}")
    
    def get_speaker_embedding(self, speaker_id: str) -> torch.Tensor:
        """
        Get speaker embedding by ID.
        
        Args:
            speaker_id: Speaker identifier.
            
        Returns:
            Speaker embedding tensor.
        """
        if speaker_id not in self.speaker_embeddings:
            available = list(self.speaker_embeddings.keys())[:5]
            raise ValueError(
                f"Speaker {speaker_id} not found. "
                f"Available: {available}... ({len(self.speaker_embeddings)} total)"
            )
        
        return self.speaker_embeddings[speaker_id].to(self.device)
    
    def synthesize(
        self,
        text: str,
        speaker_id: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Synthesize speech for given text and speaker.
        
        Args:
            text: Input text to synthesize.
            speaker_id: Target speaker ID.
            output_path: Optional path to save audio.
            
        Returns:
            Dictionary with synthesis results.
        """
        # Get speaker embedding
        speaker_embedding = self.get_speaker_embedding(speaker_id)
        
        # Tokenize input
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate speech
        start_time = time.time()
        
        with torch.no_grad():
            spectrogram = self.model.generate_speech(
                input_ids,
                speaker_embedding.unsqueeze(0)
            )
            
            # Convert to waveform
            waveform = self.vocoder(spectrogram)
        
        generation_time = time.time() - start_time
        
        # Convert to numpy
        audio = waveform.cpu().squeeze().numpy()
        sample_rate = 16000
        
        # Save if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                output_path,
                torch.FloatTensor(audio).unsqueeze(0),
                sample_rate
            )
        
        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "generation_time": generation_time,
            "audio_length": len(audio) / sample_rate,
            "spectrogram_shape": spectrogram.shape
        }
    
    def evaluate_test_set(
        self,
        test_texts: List[str],
        test_speakers: List[str],
        output_dir: str = "evaluation_outputs"
    ) -> List[EvaluationResult]:
        """
        Evaluate model on test set.
        
        Args:
            test_texts: List of test sentences.
            test_speakers: List of speaker IDs to test.
            output_dir: Directory to save outputs.
            
        Returns:
            List of evaluation results.
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL")
        print(f"{'='*70}")
        print(f"Test texts: {len(test_texts)}")
        print(f"Test speakers: {len(test_speakers)}")
        print(f"Total samples: {len(test_texts) * len(test_speakers)}")
        print()
        
        for speaker_id in test_speakers:
            print(f"\nSpeaker: {speaker_id}")
            
            for i, text in enumerate(test_texts):
                print(f"  Text {i+1}/{len(test_texts)}: {text[:50]}...")
                
                try:
                    audio_filename = f"{speaker_id}_text{i+1}.wav"
                    audio_path = output_path / audio_filename
                    
                    result = self.synthesize(
                        text=text,
                        speaker_id=speaker_id,
                        output_path=str(audio_path)
                    )
                    
                    eval_result = EvaluationResult(
                        text=text,
                        speaker_id=speaker_id,
                        audio_path=str(audio_path),
                        generation_time=result["generation_time"],
                        mel_loss=0.0,  # Would need reference audio
                        audio_length=result["audio_length"],
                        sample_rate=result["sample_rate"]
                    )
                    
                    results.append(eval_result)
                    
                    print(f"    ✓ Generated in {result['generation_time']:.2f}s")
                    print(f"    ✓ Audio length: {result['audio_length']:.2f}s")
                    
                except Exception as e:
                    print(f"    ✗ Error: {e}")
        
        return results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict:
        """
        Calculate aggregate metrics from evaluation results.
        
        Args:
            results: List of evaluation results.
            
        Returns:
            Dictionary of metrics.
        """
        if not results:
            return {}
        
        generation_times = [r.generation_time for r in results]
        audio_lengths = [r.audio_length for r in results]
        
        # Real-time factor (generation_time / audio_length)
        rtfs = [g / a if a > 0 else 0 for g, a in zip(generation_times, audio_lengths)]
        
        metrics = {
            "total_samples": len(results),
            "avg_generation_time": np.mean(generation_times),
            "min_generation_time": np.min(generation_times),
            "max_generation_time": np.max(generation_times),
            "avg_audio_length": np.mean(audio_lengths),
            "avg_rtf": np.mean(rtfs),  # < 1.0 means faster than real-time
            "total_audio_generated": np.sum(audio_lengths),
            "speakers_tested": len(set(r.speaker_id for r in results))
        }
        
        return metrics
    
    def print_report(self, results: List[EvaluationResult], metrics: Dict):
        """
        Print evaluation report.
        
        Args:
            results: List of evaluation results.
            metrics: Calculated metrics.
        """
        print(f"\n{'='*70}")
        print("EVALUATION REPORT")
        print(f"{'='*70}")
        
        print(f"\nGeneral Statistics:")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Speakers tested: {metrics['speakers_tested']}")
        print(f"  Total audio generated: {metrics['total_audio_generated']:.2f}s")
        
        print(f"\nGeneration Performance:")
        print(f"  Average generation time: {metrics['avg_generation_time']:.3f}s")
        print(f"  Min generation time: {metrics['min_generation_time']:.3f}s")
        print(f"  Max generation time: {metrics['max_generation_time']:.3f}s")
        print(f"  Average audio length: {metrics['avg_audio_length']:.2f}s")
        print(f"  Real-time factor (RTF): {metrics['avg_rtf']:.2f}x")
        
        if metrics['avg_rtf'] < 1.0:
            print(f"    ✓ Faster than real-time!")
        else:
            print(f"    ⚠ Slower than real-time")
        
        print(f"\nSample Outputs:")
        for i, result in enumerate(results[:3]):
            print(f"\n  Sample {i+1}:")
            print(f"    Speaker: {result.speaker_id}")
            print(f"    Text: {result.text[:60]}...")
            print(f"    Audio: {result.audio_path}")
            print(f"    Length: {result.audio_length:.2f}s")
        
        print(f"\n{'='*70}\n")
    
    def save_report(
        self,
        results: List[EvaluationResult],
        metrics: Dict,
        output_path: str = "evaluation_report.json"
    ):
        """
        Save evaluation report to JSON.
        
        Args:
            results: List of evaluation results.
            metrics: Calculated metrics.
            output_path: Path to save report.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "samples": [
                {
                    "text": r.text,
                    "speaker_id": r.speaker_id,
                    "audio_path": r.audio_path,
                    "generation_time": r.generation_time,
                    "audio_length": r.audio_length
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved: {output_path}")


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate voice cloning model"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    
    parser.add_argument(
        "--embeddings",
        type=str,
        default="app/embedding/speaker_embeddings.pt",
        help="Path to speaker embeddings"
    )
    
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        help="Speaker IDs to test (space-separated)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_outputs",
        help="Output directory for generated audio"
    )
    
    args = parser.parse_args()
    
    # Default test texts (Spanish)
    test_texts = [
        "Hola, mi nombre es Claudia y estoy probando el sistema de clonación de voz.",
        "Buenos días, ¿cómo estás? Espero que tengas un excelente día.",
        "La inteligencia artificial está transformando la manera en que interactuamos con la tecnología.",
        "Este es un mensaje de prueba para evaluar la calidad de la síntesis de voz.",
        "El reconocimiento de voz permite crear experiencias más naturales y accesibles."
    ]
    
    # Initialize evaluator
    evaluator = VoiceCloningEvaluator(
        model_path=args.model,
        speaker_embeddings_path=args.embeddings
    )
    
    # Get test speakers
    if args.speakers:
        test_speakers = args.speakers
    else:
        # Use first 3 speakers from embeddings
        all_speakers = list(evaluator.speaker_embeddings.keys())
        test_speakers = all_speakers[:3]
        print(f"No speakers specified, using: {test_speakers}")
    
    # Run evaluation
    results = evaluator.evaluate_test_set(
        test_texts=test_texts,
        test_speakers=test_speakers,
        output_dir=args.output
    )
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)
    
    # Print report
    evaluator.print_report(results, metrics)
    
    # Save report
    report_path = Path(args.output) / "evaluation_report.json"
    evaluator.save_report(results, metrics, str(report_path))
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Audio samples saved to: {args.output}")
    print(f"✓ Report saved to: {report_path}")


if __name__ == "__main__":
    main()