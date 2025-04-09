import tensorflow as tf
import numpy as np
from typing import Optional
import os
import soundfile as sf
from music_ai.data.preprocessor import MusicPreprocessor

class MusicGenerator:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the MusicGenerator.
        
        Args:
            model_path (str, optional): Path to the trained model
        """
        self.preprocessor = MusicPreprocessor()
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)

    def generate_sequence(self, seed: np.ndarray, length: int = 100) -> np.ndarray:
        """
        Generate a new sequence from a seed spectrogram.
        
        Args:
            seed (np.ndarray): Seed spectrogram
            length (int): Length of sequence to generate
            
        Returns:
            np.ndarray: Generated spectrogram
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please provide a valid model path.")
        
        current_sequence = seed.copy()
        generated = []
        
        for _ in range(length):
            # Get the last sequence_length frames
            input_seq = current_sequence[:, -self.preprocessor.sequence_length:]
            input_seq = np.expand_dims(input_seq, axis=0)
            
            # Generate next frame
            next_frame = self.model.predict(input_seq, verbose=0)
            next_frame = np.squeeze(next_frame)
            
            # Add to sequence
            current_sequence = np.concatenate([current_sequence, next_frame.reshape(-1, 1)], axis=1)
            generated.append(next_frame)
        
        return np.array(generated).T

    def save_as_audio(self, spectrogram: np.ndarray, output_path: str):
        """
        Convert generated spectrogram to audio and save it.
        
        Args:
            spectrogram (np.ndarray): Generated spectrogram
            output_path (str): Path to save the audio file
        """
        # Convert spectrogram back to audio
        audio = self.preprocessor.spectrogram_to_audio(spectrogram)
        
        # Save audio file
        sf.write(output_path, audio, self.preprocessor.sr)

    def generate_beat(self, output_path: str, seed_path: Optional[str] = None, length: int = 100):
        """
        Generate a complete beat and save it as an audio file.
        
        Args:
            output_path (str): Path to save the generated audio
            seed_path (str, optional): Path to seed audio file
            length (int): Length of sequence to generate
        """
        if seed_path:
            # Load and convert seed audio to spectrogram
            seed = self.preprocessor.load_audio(seed_path)
        else:
            # Generate random seed
            seed = np.random.rand(self.preprocessor.n_mels, self.preprocessor.sequence_length)
        
        # Generate sequence
        generated_sequence = self.generate_sequence(seed, length)
        
        # Save as audio
        self.save_as_audio(generated_sequence, output_path) 