#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import argparse
from music_ai.training.trainer import MusicTrainer
from music_ai.data.preprocessor import MusicPreprocessor
import tensorflow as tf

# Añadir el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from music_ai.main import train

def find_and_move_midi_files():
    """Find MIDI files in common locations and move them to midi_files directory."""
    # Create midi_files directory if it doesn't exist
    os.makedirs('midi_files', exist_ok=True)
    
    # Common locations to look for MIDI files
    search_paths = [
        'midi_files',           # Current midi_files directory
        '.',                    # Current directory
        '..',                   # Parent directory
        '/content',             # Colab root directory
        '/content/BeatNest/midi_files',  # Project midi_files directory
        '/content/sample_data'  # Colab sample data directory
    ]
    
    midi_files_found = []
    
    # Search for MIDI files in all locations
    for path in search_paths:
        if os.path.exists(path):
            # Search for both .mid and .midi files
            for pattern in ['*.mid', '*.midi']:
                midi_pattern = os.path.join(path, pattern)
                midi_files_found.extend(glob.glob(midi_pattern))
    
    if not midi_files_found:
        print("\n❌ No se encontraron archivos MIDI en ninguna ubicación común.")
        print("\nPor favor, verifica:")
        print(f"1. Que los archivos estén en: {os.path.abspath('midi_files')}")
        print("2. Que los archivos terminen en .mid o .midi")
        print("3. Que tengas permisos de lectura en los archivos")
        print("\nUbicaciones buscadas:")
        for path in search_paths:
            if os.path.exists(path):
                print(f"  ✓ {os.path.abspath(path)}")
            else:
                print(f"  ✗ {path} (no existe)")
        return False
    
    # Move all found MIDI files to midi_files directory
    files_moved = 0
    for file_path in midi_files_found:
        file_name = os.path.basename(file_path)
        destination = os.path.join('midi_files', file_name)
        if os.path.abspath(file_path) != os.path.abspath(destination):
            try:
                shutil.copy2(file_path, destination)
                print(f"✓ Copiado: {file_name}")
                files_moved += 1
            except Exception as e:
                print(f"✗ Error al copiar {file_name}: {str(e)}")
    
    if files_moved == 0 and len(midi_files_found) > 0:
        print("\n✓ Los archivos MIDI ya están en la ubicación correcta")
    
    return True

def setup_environment():
    """Setup the environment for training."""
    # Create necessary directories
    os.makedirs('mp3_files', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    print("✓ Directorios creados:")
    print(f"  - mp3_files: {os.path.abspath('mp3_files')}")
    print(f"  - checkpoints: {os.path.abspath('checkpoints')}")

def find_audio_files():
    """Find audio files in the project directory."""
    search_paths = [
        'mp3_files',  # Local directory
        '/content/BeatNest/mp3_files',  # Google Colab path
        '/content/mp3_files'  # Alternative Colab path
    ]
    
    audio_files = []
    for path in search_paths:
        if os.path.exists(path):
            print(f"\nBuscando archivos MP3 en: {path}")
            for file in glob.glob(os.path.join(path, '*.mp3')):
                audio_files.append(file)
                print(f"✓ Encontrado: {file}")
    
    if not audio_files:
        print("\n❌ No se encontraron archivos MP3.")
        print("\nPor favor, sube tus archivos MP3 a una de estas ubicaciones:")
        for path in search_paths:
            print(f"- {path}")
        print("\nSi estás en Google Colab, puedes subir los archivos usando:")
        print("1. El botón de 'Upload' en el panel izquierdo")
        print("2. O usando el comando: !cp /content/drive/MyDrive/tus_archivos/*.mp3 mp3_files/")
        return None
    
    return audio_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the music generation model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=64, help='Length of input sequences')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel bands for spectrogram')
    args = parser.parse_args()

    # Setup environment
    setup_environment()
    
    # Find audio files
    audio_files = find_audio_files()
    if not audio_files:
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = MusicPreprocessor(n_mels=args.n_mels)
    
    try:
        # Load and preprocess dataset
        print("\nCargando y preprocesando archivos de audio...")
        X, y = preprocessor.load_dataset('mp3_files', sequence_length=args.sequence_length)
        
        # Print dataset information
        print(f"\nDataset cargado exitosamente:")
        print(f"- Forma de entrada: {X.shape}")
        print(f"- Forma de objetivo: {y.shape}")
        print(f"- Número de muestras: {X.shape[0]}")
        
        # Initialize trainer
        trainer = MusicTrainer(
            input_shape=(args.n_mels, args.sequence_length),
            units=256,
            num_layers=3,
            dropout_rate=0.3
        )
        
        # Train model
        print("\nIniciando entrenamiento...")
        history = trainer.train(
            X, y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.2
        )
        
        print("\nEntrenamiento completado exitosamente!")
        print(f"Pérdida final de entrenamiento: {history['train_loss'][-1]:.4f}")
        print(f"Pérdida final de validación: {history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 