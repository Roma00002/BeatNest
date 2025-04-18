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
        # Define default sequence length
        self.sequence_length = 50
        
        if model_path and os.path.exists(model_path):
            try:
                # Try loading the complete model
                self.model = tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"Error loading complete model: {str(e)}")
                
                # If it's a weights file, we need to create a model first and then load the weights
                if model_path.endswith('.weights.h5'):
                    try:
                        from music_ai.training.trainer import MusicTrainer
                        # Create a new model with the same architecture
                        trainer = MusicTrainer(input_shape=(50, 128))
                        self.model = trainer.model
                        # Load weights
                        self.model.load_weights(model_path)
                        print("Successfully loaded model weights")
                    except Exception as e2:
                        print(f"Error loading model weights: {str(e2)}")
                        
                # Also try alternative model paths
                else:
                    # Try .h5 extension if not already tried
                    alt_path = model_path.replace('.weights.h5', '.h5')
                    if os.path.exists(alt_path):
                        try:
                            self.model = tf.keras.models.load_model(alt_path)
                            print(f"Loaded model from alternative path: {alt_path}")
                        except Exception:
                            pass
                    
                    # Try .weights.h5 extension if not already tried
                    alt_path = model_path.replace('.h5', '.weights.h5')
                    if os.path.exists(alt_path) and not model_path.endswith('.weights.h5'):
                        try:
                            from music_ai.training.trainer import MusicTrainer
                            trainer = MusicTrainer(input_shape=(50, 128))
                            self.model = trainer.model
                            self.model.load_weights(alt_path)
                            print(f"Loaded weights from alternative path: {alt_path}")
                        except Exception:
                            pass

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
            input_seq = current_sequence[:, -self.sequence_length:]
            
            # Reshape to match model's expected input shape (batch_size, sequence_length, features)
            input_seq = np.expand_dims(input_seq.T, axis=0)  # Shape: (1, sequence_length, n_mels)
            
            # Generate next frame
            next_frame = self.model.predict(input_seq, verbose=0)
            next_frame = np.squeeze(next_frame)
            
            # If next_frame is 2D (sequence_length, n_mels), take the last timestep
            if len(next_frame.shape) > 1:
                next_frame = next_frame[-1]
                
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
            
            # Ensure seed has correct dimensions
            if seed.shape[1] < self.sequence_length:
                # Pad if too short
                padding = np.zeros((seed.shape[0], self.sequence_length - seed.shape[1]))
                seed = np.concatenate([seed, padding], axis=1)
            elif seed.shape[1] > self.sequence_length:
                # Trim if too long
                seed = seed[:, :self.sequence_length]
        else:
            # Generate random seed (n_mels x sequence_length)
            seed = np.random.rand(self.preprocessor.n_mels, self.sequence_length)
        
        # Generate sequence
        generated_sequence = self.generate_sequence(seed, length)
        
        # Save as audio
        self.save_as_audio(generated_sequence, output_path) 