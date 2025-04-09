#!/usr/bin/env python3
import os
import sys
import argparse
from music_ai.training.trainer import MusicTrainer
from music_ai.data.preprocessor import MusicPreprocessor
from music_ai.data.genres import get_genre_path, create_genre_directories, get_genre_name, GENRE_STRUCTURE

def print_genre_structure(structure: dict, level: int = 0):
    """Print the genre structure in a tree-like format."""
    for key, value in structure.items():
        if isinstance(value, dict):
            if 'name' in value:
                print("  " * level + f"├─ {value['name']}")
                if 'subgenres' in value:
                    print_genre_structure(value['subgenres'], level + 1)
            else:
                print("  " * level + f"├─ {key}")
                print_genre_structure(value, level + 1)
        else:
            print("  " * level + f"├─ {value}")

def select_genre() -> str:
    """Show available genres and let user select one."""
    print("\nGéneros disponibles:")
    print("=" * 50)
    print_genre_structure(GENRE_STRUCTURE)
    print("=" * 50)
    
    while True:
        genre_path = input("\nIngresa la ruta del género (ej: hiphop_rap/trap/hood_trap): ").strip()
        if get_genre_path(genre_path):
            return genre_path
        print("❌ Género no válido. Por favor, usa el formato correcto.")

def setup_environment(genre_path: str):
    """Setup the environment for training a specific genre."""
    # Create genre directories
    create_genre_directories()
    
    # Get full genre path
    genre_dir = get_genre_path(genre_path)
    if not genre_dir:
        print(f"❌ Género no válido: {genre_path}")
        print("Usa el formato: genero/subgenero o genero/subgenero/subsubgenero")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs(os.path.join(genre_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(genre_dir, 'models'), exist_ok=True)
    
    print(f"\n✓ Entrenando para el género: {get_genre_name(genre_path)}")
    print(f"✓ Directorios creados:")
    print(f"  - Audio: {os.path.abspath(os.path.join(genre_dir, 'audio'))}")
    print(f"  - Modelos: {os.path.abspath(os.path.join(genre_dir, 'models'))}")
    
    return genre_dir

def find_audio_files(genre_dir: str):
    """Find audio files in the genre directory."""
    audio_dir = os.path.join(genre_dir, 'audio')
    search_paths = [
        audio_dir,  # Genre-specific directory
        os.path.join('/content/BeatNest', audio_dir),  # Google Colab path
        os.path.join('/content', audio_dir)  # Alternative Colab path
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
        print("2. O usando el comando: !cp /content/drive/MyDrive/tus_archivos/*.mp3 " + audio_dir)
        return None
    
    return audio_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Entrenar modelo para un género específico')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--sequence-length', type=int, default=64, help='Longitud de la secuencia')
    parser.add_argument('--n-mels', type=int, default=128, help='Número de bandas mel')
    args = parser.parse_args()
    
    # Select genre
    genre_path = select_genre()
    
    # Setup environment
    genre_dir = setup_environment(genre_path)
    
    # Find audio files
    audio_files = find_audio_files(genre_dir)
    if not audio_files:
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = MusicPreprocessor(n_mels=args.n_mels)
    
    try:
        # Load and preprocess dataset
        print("\nCargando y preprocesando archivos de audio...")
        X, y = preprocessor.load_dataset(os.path.join(genre_dir, 'audio'), sequence_length=args.sequence_length)
        
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
        
        # Save model to genre-specific directory
        model_path = os.path.join(genre_dir, 'models', 'model.h5')
        trainer.model.save(model_path)
        
        print("\nEntrenamiento completado exitosamente!")
        print(f"Pérdida final de entrenamiento: {history['train_loss'][-1]:.4f}")
        print(f"Pérdida final de validación: {history['val_loss'][-1]:.4f}")
        print(f"Modelo guardado en: {model_path}")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 