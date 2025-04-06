import tensorflow as tf
import numpy as np
from typing import Optional
import os
from midiutil import MIDIFile
from ..data.preprocessor import MusicPreprocessor

class MusicGenerator:
    def __init__(
        self,
        model: tf.keras.Model,
        preprocessor: MusicPreprocessor,
        temperature: float = 1.0
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.temperature = temperature

    def generate_sequence(
        self,
        initial_sequence: np.ndarray,
        sequence_length: int,
        temperature: Optional[float] = None
    ) -> np.ndarray:
        """Generate a new sequence of specified length."""
        if temperature is None:
            temperature = self.temperature

        # Ensure initial sequence has correct shape
        if len(initial_sequence.shape) == 2:
            initial_sequence = np.expand_dims(initial_sequence, 0)

        # Generate sequence
        generated = self.model.generate(
            tf.convert_to_tensor(initial_sequence),
            sequence_length
        )

        # Apply temperature
        if temperature != 1.0:
            generated = tf.exp(tf.math.log(generated) / temperature)
            generated = generated / tf.reduce_sum(generated, axis=-1, keepdims=True)

        return generated.numpy()

    def save_as_midi(
        self,
        sequence: np.ndarray,
        output_path: str,
        tempo: int = 120,
        base_note: int = 60,
        velocity: int = 100
    ):
        """Convert generated sequence to MIDI file."""
        # Create MIDI file
        midi = MIDIFile(1)  # One track
        midi.addTempo(0, 0, tempo)

        # Convert spectrogram to note events
        # This is a simple conversion - you might want to make this more sophisticated
        time_step = 0
        threshold = 0.5

        for time_slice in sequence[0].T:  # Remove batch dimension
            for note_idx, magnitude in enumerate(time_slice):
                if magnitude > threshold:
                    # Add note
                    note = base_note + note_idx
                    midi.addNote(
                        track=0,
                        channel=0,
                        pitch=note,
                        time=time_step,
                        duration=1,  # Quarter note
                        volume=int(magnitude * velocity)
                    )
            time_step += 0.25  # Sixteenth note steps

        # Save MIDI file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            midi.writeFile(f)

    def generate_beat(
        self,
        seed_path: Optional[str] = None,
        sequence_length: int = 256,
        output_path: str = "generated/output.mid"
    ):
        """Generate a complete beat and save it as a MIDI file."""
        if seed_path is not None:
            # Load and process seed file
            seed = self.preprocessor.midi_to_spectrogram(seed_path)
            seed = np.expand_dims(seed, 0)  # Add batch dimension
        else:
            # Create random seed
            seed = np.random.uniform(0, 1, (1, self.model.input_shape[0], 64))

        # Generate sequence
        generated_sequence = self.generate_sequence(
            seed,
            sequence_length
        )

        # Save as MIDI
        self.save_as_midi(generated_sequence, output_path)
        print(f"Generated beat saved to: {output_path}") 