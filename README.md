# AI Music Beat Generator

This project uses deep learning to generate musical beats using TensorFlow. It employs LSTM (Long Short-Term Memory) neural networks to learn patterns from MIDI files and generate new, original beats.

## Features

- Train on MIDI files to learn musical patterns
- Convert MIDI files to mel spectrograms for processing
- Generate new beats using LSTM neural networks
- Save generated beats as MIDI files
- Configurable model architecture and training parameters
- Temperature-based generation for controlling creativity

## Requirements

- Python 3.8 or higher
- TensorFlow 2.12 or higher
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd music-ai-generator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model on your MIDI files:

```bash
python -m music_ai.main train --data-dir path/to/midi/files
```

Optional training arguments:
- `--checkpoint-dir`: Directory to save model checkpoints (default: "checkpoints")
- `--sample-rate`: Audio sample rate (default: 44100)
- `--hop-length`: Hop length for spectrogram (default: 512)
- `--n-mels`: Number of mel bands (default: 128)
- `--sequence-length`: Sequence length (default: 64)
- `--units`: Number of LSTM units (default: 256)
- `--num-layers`: Number of LSTM layers (default: 3)
- `--learning-rate`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 100)

### Generating Beats

To generate new beats using a trained model:

```bash
python -m music_ai.main generate
```

Optional generation arguments:
- `--checkpoint-dir`: Directory containing model checkpoints (default: "checkpoints")
- `--output-path`: Output MIDI file path (default: "generated/output.mid")
- `--seed-path`: Path to seed MIDI file (optional)
- `--temperature`: Generation temperature (default: 1.0)
- `--sequence-length`: Length of generated sequence (default: 256)

## Model Architecture

The model uses a stack of LSTM layers followed by batch normalization layers to learn temporal patterns in music. The architecture includes:

- Multiple LSTM layers with configurable units
- Batch normalization for stable training
- Dropout for regularization
- Dense output layer with sigmoid activation

## Data Processing

The system processes MIDI files by:
1. Converting MIDI to audio using `pretty_midi`
2. Converting audio to mel spectrograms using `librosa`
3. Normalizing and preparing sequences for training
4. Converting generated spectrograms back to MIDI

## Tips for Best Results

1. Use a consistent set of MIDI files for training (similar style/genre)
2. Experiment with different temperature values during generation
3. Try different sequence lengths for various musical patterns
4. Use a seed file for more controlled generation
5. Adjust the number of LSTM units and layers based on your dataset size

## License

This project is licensed under the MIT License - see the LICENSE file for details. 