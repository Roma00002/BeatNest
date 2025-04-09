import os
import numpy as np
import librosa
import tensorflow as tf
from typing import Tuple, List
from pathlib import Path

class MusicPreprocessor:
    def __init__(self, n_mels: int = 128, sr: int = 22050):
        """
        Initialize the MusicPreprocessor.
        
        Args:
            n_mels (int): Number of mel bands for spectrogram
            sr (int): Sample rate for audio loading
        """
        self.n_mels = n_mels
        self.sr = sr

    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load an audio file and convert it to a mel spectrogram.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Mel spectrogram of the audio
        """
        # Load audio file
        y, sr = librosa.load(file_path, sr=self.sr)
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=512,
            n_fft=2048
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        
        return mel_spec

    def create_sequences(self, spectrogram: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input and target sequences from a spectrogram.
        
        Args:
            spectrogram (np.ndarray): Mel spectrogram of the audio
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input and target sequences
        """
        sequences = []
        targets = []
        
        # Create overlapping sequences
        for i in range(len(spectrogram[0]) - sequence_length):
            sequence = spectrogram[:, i:i + sequence_length]
            target = spectrogram[:, i + sequence_length]
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

    def load_dataset(self, data_dir: str, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess all audio files in a directory.
        
        Args:
            data_dir (str): Directory containing audio files
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input and target sequences for training
        """
        all_sequences = []
        all_targets = []
        
        # Get all audio files in the directory
        audio_files = [f for f in os.listdir(data_dir) if f.endswith(('.mp3', '.wav', '.ogg'))]
        
        if not audio_files:
            raise ValueError(f"No audio files found in {data_dir}")
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process each audio file
        for i, audio_file in enumerate(audio_files):
            print(f"Processing file {i+1}/{len(audio_files)}: {audio_file}")
            file_path = os.path.join(data_dir, audio_file)
            
            try:
                # Load and convert to spectrogram
                spectrogram = self.load_audio(file_path)
                
                # Create sequences
                sequences, targets = self.create_sequences(spectrogram, sequence_length)
                
                all_sequences.append(sequences)
                all_targets.append(targets)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue
        
        if not all_sequences:
            raise ValueError("No valid sequences were created from the audio files")
        
        # Concatenate all sequences
        X = np.concatenate(all_sequences, axis=0)
        y = np.concatenate(all_targets, axis=0)
        
        print(f"Created {len(X)} sequences from {len(audio_files)} audio files")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y

    def pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or truncate sequence to target length."""
        current_length = sequence.shape[1]
        
        if current_length > target_length:
            # Truncate
            return sequence[:, :target_length]
        elif current_length < target_length:
            # Pad with zeros
            padding = np.zeros((sequence.shape[0], target_length - current_length))
            return np.concatenate([sequence, padding], axis=1)
        else:
            return sequence

    def spectrogram_to_audio(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert spectrogram back to audio signal."""
        # Denormalize
        spec_db = spectrogram * 80 - 80  # Approximate inverse of normalization
        
        # Convert back to power spectrum
        spec = librosa.db_to_power(spec_db)
        
        # Perform Griffin-Lim algorithm to recover phase
        audio = librosa.feature.inverse.mel_to_audio(
            spec,
            sr=self.sr,
            hop_length=512
        )
        
        return audio 