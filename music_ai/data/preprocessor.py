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
        """Create input and target sequences from a spectrogram.
        
        Args:
            spectrogram: Mel spectrogram of shape (n_mels, time_steps)
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) where:
                X: Input sequences of shape (n_sequences, sequence_length, n_mels)
                y: Target sequences of shape (n_sequences, sequence_length, n_mels)
        """
        sequences = []
        targets = []
        
        # Calculate total possible sequences
        total_steps = len(spectrogram[0]) - sequence_length
        
        # If there are too many steps, use a stride to reduce the number of sequences
        stride = max(1, total_steps // 500)  # Aim for about 500 sequences per song
        
        # Create sequences with stride to reduce the total amount
        for i in range(0, total_steps, stride):
            # Input sequence (shape: sequence_length x n_mels)
            sequence = spectrogram[:, i:i + sequence_length].T
            sequences.append(sequence)
            
            # Target sequence (shape: sequence_length x n_mels)
            target = spectrogram[:, i + 1:i + sequence_length + 1].T
            targets.append(target)
        
        # Convert to numpy arrays and reshape to match model input
        X = np.array(sequences)  # Shape: (n_sequences, sequence_length, n_mels)
        y = np.array(targets)   # Shape: (n_sequences, sequence_length, n_mels)
        
        return X, y

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

    def process_batch(self, audio_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch of audio files and return the sequences and targets.
        
        Args:
            audio_files: List of paths to audio files
            
        Returns:
            Tuple of (X, y) where:
                X: Input sequences of shape (n_sequences, sequence_length, n_mels)
                y: Target sequences of shape (n_sequences, sequence_length, n_mels)
        """
        all_sequences = []
        all_targets = []
        
        for audio_file in audio_files:
            try:
                # Load and preprocess the audio file
                mel_spec = self.load_audio(audio_file)
                if mel_spec is None:
                    continue
                
                # Apply data augmentation with more variations
                # 1. Random pitch shift (70% chance)
                if np.random.random() > 0.3:
                    # Pitch shift by randomly shifting rows up/down
                    shift = np.random.randint(-3, 4)  # Increased range
                    if shift != 0:
                        if shift > 0:
                            mel_spec = np.vstack([mel_spec[shift:], np.zeros((shift, mel_spec.shape[1]))])
                        else:
                            mel_spec = np.vstack([np.zeros((-shift, mel_spec.shape[1])), mel_spec[:shift]])
                
                # 2. Time stretching (50% chance)
                if np.random.random() > 0.5:
                    stretch_factor = np.random.uniform(0.9, 1.1)  # 10% stretch/shrink
                    orig_len = mel_spec.shape[1]
                    new_len = int(orig_len * stretch_factor)
                    if new_len > orig_len:
                        # Stretch (interpolate)
                        indices = np.linspace(0, orig_len-1, new_len)
                        stretched = np.zeros((mel_spec.shape[0], new_len))
                        for i in range(mel_spec.shape[0]):
                            stretched[i] = np.interp(indices, np.arange(orig_len), mel_spec[i])
                        mel_spec = stretched[:, :orig_len]  # Keep original length
                    elif new_len < orig_len:
                        # Shrink (downsample)
                        indices = np.linspace(0, new_len-1, orig_len)
                        shrunk = np.zeros((mel_spec.shape[0], orig_len))
                        for i in range(mel_spec.shape[0]):
                            shrunk[i] = np.interp(np.arange(orig_len), indices, mel_spec[i, :new_len])
                        mel_spec = shrunk
                
                # 3. Add small random noise (30% chance)
                if np.random.random() > 0.7:
                    noise_level = np.random.uniform(0.01, 0.05)  # 1-5% noise
                    noise = np.random.normal(0, noise_level, mel_spec.shape)
                    mel_spec = mel_spec + noise
                
                # Normalize the mel spectrogram with improved normalization
                # Z-score normalization per frequency band
                mel_mean = np.mean(mel_spec, axis=1, keepdims=True)
                mel_std = np.std(mel_spec, axis=1, keepdims=True) + 1e-8
                mel_spec = (mel_spec - mel_mean) / mel_std
                
                # Create sequences with length 50 to match model input shape
                sequences, targets = self.create_sequences(mel_spec, sequence_length=50)
                
                # Ensure correct shape (n_sequences, sequence_length, n_mels)
                sequences = sequences.reshape(-1, 50, self.n_mels)
                targets = targets.reshape(-1, 50, self.n_mels)
                
                # Filter out zero-padding sequences with minimal information
                seq_entropy = np.sum(np.abs(sequences), axis=(1, 2))
                valid_indices = seq_entropy > np.percentile(seq_entropy, 5)  # Remove bottom 5% low-information sequences
                
                if np.sum(valid_indices) > 0:
                    sequences = sequences[valid_indices]
                    targets = targets[valid_indices]
                    
                    all_sequences.extend(sequences)
                    all_targets.extend(targets)
                
                # Clear memory
                del mel_spec, sequences, targets
                gc.collect()
                
            except Exception as e:
                print(f"Error procesando {audio_file}: {str(e)}")
                continue
        
        if not all_sequences:
            return None, None
            
        # Convert lists to numpy arrays
        X = np.array(all_sequences)
        y = np.array(all_targets)
        
        # Ensure final shape is correct
        X = X.reshape(-1, 50, self.n_mels)
        y = y.reshape(-1, 50, self.n_mels)
        
        # Shuffle the data for better training
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        print(f"✓ Generados {X.shape[0]} ejemplos de entrenamiento de {len(audio_files)} archivos de audio")
        
        return X, y 