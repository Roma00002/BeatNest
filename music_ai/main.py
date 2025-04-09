#!/usr/bin/env python3
import argparse
import os
from music_ai.training.trainer import MusicTrainer
from music_ai.data.preprocessor import MusicPreprocessor
from music_ai.generation.generator import MusicGenerator

def train(args):
    """Train the model on audio files."""
    # Initialize preprocessor and trainer
    preprocessor = MusicPreprocessor(n_mels=args.n_mels)
    trainer = MusicTrainer(
        input_shape=(args.n_mels, args.sequence_length),
        units=args.units,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    )
    
    print("Loading and preprocessing audio files...")
    X, y = preprocessor.load_dataset(args.data_dir, sequence_length=args.sequence_length)
    
    print(f"Dataset shapes - X: {X.shape}, y: {y.shape}")
    
    # Train model
    history = trainer.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2
    )
    
    print(f"Training completed. Final loss: {history['train_loss'][-1]:.4f}")

def generate(args):
    """Generate new audio using the trained model."""
    generator = MusicGenerator(model_path=args.model_path)
    
    # Generate new audio
    generator.generate_beat(
        output_path=args.output_path,
        seed_path=args.seed_path,
        length=args.length
    )
    
    print(f"Generated audio saved to: {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Music AI Training and Generation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-dir", required=True, help="Directory containing audio files")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--sequence-length", type=int, default=64, help="Length of input sequences")
    train_parser.add_argument("--n-mels", type=int, default=128, help="Number of mel bands")
    train_parser.add_argument("--units", type=int, default=256, help="Number of units in LSTM layers")
    train_parser.add_argument("--num-layers", type=int, default=3, help="Number of LSTM layers")
    train_parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout rate")
    train_parser.set_defaults(func=train)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new audio")
    gen_parser.add_argument("--model-path", required=True, help="Path to trained model")
    gen_parser.add_argument("--output-path", default="generated/output.wav", help="Output audio file path")
    gen_parser.add_argument("--seed-path", help="Path to seed audio file (optional)")
    gen_parser.add_argument("--length", type=int, default=100, help="Length of sequence to generate")
    gen_parser.set_defaults(func=generate)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)

if __name__ == "__main__":
    main() 