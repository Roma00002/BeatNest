import os
import numpy as np
import librosa
import tensorflow as tf
from typing import Tuple, List
from pathlib import Path
import gc
import psutil
import h5py
import tempfile

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

    def load_dataset(self, audio_dir: str, sequence_length: int = 100, 
                    batch_size: int = 3, specific_files: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess audio files in batches to manage memory usage.
        
        Args:
            audio_dir (str): Directory containing audio files
            sequence_length (int): Length of sequences to generate
            batch_size (int): Number of files to process at once
            specific_files (List[str]): Specific files to process, if None process all files
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed dataset (X, y)
        """
        # Get list of audio files
        if specific_files is None:
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav', '.ogg'))]
        else:
            audio_files = [f for f in specific_files if f.endswith(('.mp3', '.wav', '.ogg'))]
            
        if not audio_files:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        print(f"\nProcesando {len(audio_files)} archivos de audio")
        print("Archivos a procesar:")
        for file in audio_files:
            print(f"- {file}")
        
        # Initialize lists to store processed data
        all_spectrograms = []
        all_sequences = []
        
        # Process files in batches
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}")
            
            # Clear memory before processing batch
            gc.collect()
            
            # Process each file in the batch
            for audio_file in batch_files:
                try:
                    # Load and preprocess audio
                    audio_path = os.path.join(audio_dir, audio_file)
                    print(f"Processing: {audio_file}")
                    
                    # Load audio file
                    y, sr = librosa.load(audio_path, sr=self.sr)
                    
                    # Extract spectrogram
                    spectrogram = self.load_audio(audio_path)
                    
                    # Generate sequences
                    sequences, targets = self.create_sequences(spectrogram, sequence_length)
                    
                    # Append to lists
                    all_spectrograms.append(spectrogram)
                    all_sequences.extend(sequences)
                    
                    # Clear memory after processing each file
                    del y, sr, spectrogram, sequences, targets
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {str(e)}")
                    continue
            
            # Clear memory after batch
            gc.collect()
        
        # Convert lists to numpy arrays
        X = np.array(all_sequences)
        y = np.array(all_sequences)  # Autoencoder target is same as input
        
        # Clear memory
        del all_spectrograms, all_sequences
        gc.collect()
        
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