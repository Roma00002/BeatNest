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

    def load_dataset(self, data_dir: str, sequence_length: int, batch_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess all audio files in a directory, processing them in batches.
        Saves processed data to disk to avoid memory issues.
        
        Args:
            data_dir (str): Directory containing audio files
            sequence_length (int): Length of input sequences
            batch_size (int): Number of files to process at once
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input and target sequences for training
        """
        def get_memory_usage():
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        
        def clear_memory():
            gc.collect()
            tf.keras.backend.clear_session()
            if tf.config.list_physical_devices('GPU'):
                tf.config.experimental.reset_memory_stats('GPU:0')
        
        # Get all audio files in the directory
        audio_files = [f for f in os.listdir(data_dir) if f.endswith(('.mp3', '.wav', '.ogg'))]
        
        if not audio_files:
            raise ValueError(f"No audio files found in {data_dir}")
        
        print(f"Found {len(audio_files)} audio files")
        print(f"Processing files in batches of {batch_size}")
        print(f"Initial memory usage: {get_memory_usage():.2f} MB")
        
        # Create temporary directory for processed data
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, 'processed_data.h5')
        
        # Initialize HDF5 file
        with h5py.File(temp_file, 'w') as h5f:
            # Create datasets with compression
            h5f.create_dataset('X', shape=(0, self.n_mels, sequence_length), 
                              maxshape=(None, self.n_mels, sequence_length),
                              dtype='float32', compression='gzip')
            h5f.create_dataset('y', shape=(0, self.n_mels), 
                              maxshape=(None, self.n_mels),
                              dtype='float32', compression='gzip')
            
            total_sequences = 0
            
            # Process files in batches
            for i in range(0, len(audio_files), batch_size):
                batch_files = audio_files[i:i + batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}")
                print(f"Memory before batch: {get_memory_usage():.2f} MB")
                
                batch_sequences = []
                batch_targets = []
                
                for audio_file in batch_files:
                    print(f"Processing: {audio_file}")
                    file_path = os.path.join(data_dir, audio_file)
                    
                    try:
                        # Clear memory before processing new file
                        clear_memory()
                        
                        # Load and convert to spectrogram
                        spectrogram = self.load_audio(file_path)
                        
                        # Create sequences
                        sequences, targets = self.create_sequences(spectrogram, sequence_length)
                        
                        # Convert to float32 to reduce memory usage
                        sequences = sequences.astype(np.float32)
                        targets = targets.astype(np.float32)
                        
                        batch_sequences.append(sequences)
                        batch_targets.append(targets)
                        
                        # Clear memory after processing file
                        del spectrogram, sequences, targets
                        clear_memory()
                        
                        print(f"Memory after processing {audio_file}: {get_memory_usage():.2f} MB")
                        
                    except Exception as e:
                        print(f"Error processing {audio_file}: {str(e)}")
                        continue
                
                if batch_sequences:
                    # Concatenate sequences in this batch
                    X_batch = np.concatenate(batch_sequences, axis=0)
                    y_batch = np.concatenate(batch_targets, axis=0)
                    
                    # Clear memory before saving to disk
                    del batch_sequences, batch_targets
                    clear_memory()
                    
                    # Save batch to HDF5 file
                    with h5py.File(temp_file, 'a') as h5f:
                        # Resize datasets
                        h5f['X'].resize(total_sequences + len(X_batch), axis=0)
                        h5f['y'].resize(total_sequences + len(y_batch), axis=0)
                        
                        # Append data
                        h5f['X'][total_sequences:] = X_batch
                        h5f['y'][total_sequences:] = y_batch
                        
                        total_sequences += len(X_batch)
                    
                    # Clear memory after saving
                    del X_batch, y_batch
                    clear_memory()
                    
                    print(f"Memory after batch: {get_memory_usage():.2f} MB")
                
                # Force garbage collection between batches
                gc.collect()
        
        # Load final data from HDF5 file
        with h5py.File(temp_file, 'r') as h5f:
            X = h5f['X'][:]
            y = h5f['y'][:]
        
        # Clean up temporary files
        os.remove(temp_file)
        os.rmdir(temp_dir)
        
        print(f"\nCreated {len(X)} sequences from {len(audio_files)} audio files")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Final memory usage: {get_memory_usage():.2f} MB")
        
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