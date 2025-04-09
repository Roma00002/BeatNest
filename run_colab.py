#!/usr/bin/env python3
import os
import sys
from music_ai.training.trainer import MusicTrainer
from music_ai.data.preprocessor import MusicPreprocessor

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
            print(f"\nBuscando archivos de audio en: {path}")
            for file in os.listdir(path):
                if file.endswith(('.mp3', '.wav', '.ogg')):
                    audio_files.append(os.path.join(path, file))
                    print(f"✓ Encontrado: {file}")
    
    if not audio_files:
        print("\n❌ No se encontraron archivos de audio.")
        print("\nPor favor, sube tus archivos de audio a una de estas ubicaciones:")
        for path in search_paths:
            print(f"- {path}")
        print("\nSi estás en Google Colab, puedes subir los archivos usando:")
        print("1. El botón de 'Upload' en el panel izquierdo")
        print("2. O usando el comando: !cp /content/drive/MyDrive/tus_archivos/*.mp3 mp3_files/")
        return None
    
    return audio_files

def main():
    # Setup environment
    setup_environment()
    
    # Find audio files
    audio_files = find_audio_files()
    if not audio_files:
        sys.exit(1)
    
    # Initialize preprocessor with default parameters
    preprocessor = MusicPreprocessor(n_mels=128)
    
    try:
        # Load and preprocess dataset
        print("\nCargando y preprocesando archivos de audio...")
        X, y = preprocessor.load_dataset('mp3_files', sequence_length=64)
        
        # Print dataset information
        print(f"\nDataset cargado exitosamente:")
        print(f"- Forma de entrada: {X.shape}")
        print(f"- Forma de objetivo: {y.shape}")
        print(f"- Número de muestras: {X.shape[0]}")
        
        # Initialize trainer with default parameters
        trainer = MusicTrainer(
            input_shape=(128, 64),  # n_mels, sequence_length
            units=256,
            num_layers=3,
            dropout_rate=0.3
        )
        
        # Train model
        print("\nIniciando entrenamiento...")
        history = trainer.train(
            X, y,
            epochs=100,
            batch_size=32,
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