import argparse
import os
from data.preprocessor import MusicPreprocessor
from models.music_generator import MusicGenerator as Model
from training.trainer import MusicTrainer
from generation.generator import MusicGenerator
import tensorflow as tf
import numpy as np

def train(args):
    """Train the model on MIDI files."""
    # Initialize preprocessor
    preprocessor = MusicPreprocessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # Load and preprocess dataset
    print("Loading and preprocessing MIDI files...")
    X, y = preprocessor.load_dataset(
        args.data_dir,
        sequence_length=args.sequence_length
    )
    
    # Split into train and validation sets
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Initialize model
    model = Model(
        input_shape=(X.shape[1], X.shape[2]),
        units=args.units,
        num_layers=args.num_layers
    )
    
    # Initialize trainer
    trainer = MusicTrainer(
        model,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    trainer.train(
        train_data=(X_train, y_train),
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs
    )

def generate(args):
    """Generate new beats using the trained model."""
    # Initialize preprocessor
    preprocessor = MusicPreprocessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # Initialize model
    model = Model(
        input_shape=(args.n_mels, args.sequence_length),
        units=args.units,
        num_layers=args.num_layers
    )
    
    # Load checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_dir))
    
    if not checkpoint:
        raise ValueError("No checkpoint found. Train the model first.")
    
    # Initialize generator
    generator = MusicGenerator(
        model,
        preprocessor,
        temperature=args.temperature
    )
    
    # Generate beats
    generator.generate_beat(
        seed_path=args.seed_path,
        sequence_length=args.sequence_length,
        output_path=args.output_path
    )

def main():
    parser = argparse.ArgumentParser(description="AI Music Beat Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-dir", required=True, help="Directory containing MIDI files")
    train_parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save checkpoints")
    train_parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    train_parser.add_argument("--hop-length", type=int, default=512, help="Hop length for spectrogram")
    train_parser.add_argument("--n-mels", type=int, default=128, help="Number of mel bands")
    train_parser.add_argument("--sequence-length", type=int, default=64, help="Sequence length")
    train_parser.add_argument("--units", type=int, default=256, help="Number of LSTM units")
    train_parser.add_argument("--num-layers", type=int, default=3, help="Number of LSTM layers")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    
    # Generation arguments
    gen_parser = subparsers.add_parser("generate", help="Generate new beats")
    gen_parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory containing checkpoints")
    gen_parser.add_argument("--output-path", default="generated/output.mid", help="Output MIDI file path")
    gen_parser.add_argument("--seed-path", help="Path to seed MIDI file (optional)")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    gen_parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    gen_parser.add_argument("--hop-length", type=int, default=512, help="Hop length for spectrogram")
    gen_parser.add_argument("--n-mels", type=int, default=128, help="Number of mel bands")
    gen_parser.add_argument("--sequence-length", type=int, default=256, help="Length of generated sequence")
    gen_parser.add_argument("--units", type=int, default=256, help="Number of LSTM units")
    gen_parser.add_argument("--num-layers", type=int, default=3, help="Number of LSTM layers")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 