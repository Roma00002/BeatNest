import os
import numpy as np
import pretty_midi
import librosa
from typing import List, Tuple
import tensorflow as tf

class MusicPreprocessor:
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512, n_mels: int = 128):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels

    def midi_to_spectrogram(self, midi_path: str) -> np.ndarray:
        """Convert MIDI file to mel spectrogram."""
        try:
            # Load MIDI file
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # Synthesize audio from MIDI
            audio = midi_data.synthesize(fs=self.sample_rate)
            
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            return mel_spec_norm
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            return None

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

    def load_dataset(self, data_dir: str, sequence_length: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Load all MIDI files from directory and convert to spectrograms."""
        spectrograms = []
        
        for file in os.listdir(data_dir):
            if file.endswith('.mid') or file.endswith('.midi'):
                midi_path = os.path.join(data_dir, file)
                spec = self.midi_to_spectrogram(midi_path)
                
                if spec is not None:
                    # Ensure minimum length
                    if spec.shape[1] >= sequence_length + 1:  # +1 for input/target shift
                        # Create sequences with overlap
                        for i in range(0, spec.shape[1] - sequence_length - 1, sequence_length // 2):
                            # Input sequence
                            input_seq = spec[:, i:i + sequence_length]
                            # Target sequence (siguiente valor para cada banda mel)
                            target_seq = spec[:, i + sequence_length]
                            
                            spectrograms.append((input_seq, target_seq))

        if not spectrograms:
            raise ValueError("No valid spectrograms were generated from the MIDI files")

        # Separar las secuencias de entrada y objetivo
        X, y = zip(*spectrograms)
        
        # Convertir a arrays de numpy
        X = np.array(X)  # shape: [batch, mel_bands, sequence_length]
        y = np.array(y)  # shape: [batch, mel_bands]
        
        # Expandir la dimensión del objetivo para que coincida con el formato esperado
        y = np.expand_dims(y, axis=2)  # shape: [batch, mel_bands, 1]
        
        print(f"Dataset shapes - X: {X.shape}, y: {y.shape}")  # Debug info
        
        return X, y

    def spectrogram_to_audio(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert spectrogram back to audio signal."""
        # Denormalize
        spec_db = spectrogram * 80 - 80  # Approximate inverse of normalization
        
        # Convert back to power spectrum
        spec = librosa.db_to_power(spec_db)
        
        # Perform Griffin-Lim algorithm to recover phase
        audio = librosa.feature.inverse.mel_to_audio(
            spec,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return audio 