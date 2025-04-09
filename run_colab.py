#!/usr/bin/env python3
import os
import sys
import argparse
from music_ai.training.trainer import MusicTrainer
from music_ai.data.preprocessor import MusicPreprocessor
from music_ai.data.genres import get_genre_path, create_genre_directories, get_genre_name, GENRE_STRUCTURE
import gradio as gr

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

def get_available_genres():
    """Get a list of available genres and their models."""
    genres = []
    for genre_key, genre_data in GENRE_STRUCTURE.items():
        if 'subgenres' in genre_data:
            for subgenre_key, subgenre_data in genre_data['subgenres'].items():
                if 'subgenres' in subgenre_data:
                    for subsubgenre_key, subsubgenre_data in subgenre_data['subgenres'].items():
                        path = f"{genre_key}/{subgenre_key}/{subsubgenre_key}"
                        model_path = os.path.join('generos', path, 'models', 'model.h5')
                        if os.path.exists(model_path):
                            genres.append((path, get_genre_name(path)))
                else:
                    path = f"{genre_key}/{subgenre_key}"
                    model_path = os.path.join('generos', path, 'models', 'model.h5')
                    if os.path.exists(model_path):
                        genres.append((path, get_genre_name(path)))
        else:
            path = genre_key
            model_path = os.path.join('generos', path, 'models', 'model.h5')
            if os.path.exists(model_path):
                genres.append((path, get_genre_name(path)))
    return genres

def generate_beat(genre_path: str, length: int = 100, temperature: float = 1.0):
    """Generate a beat using the selected genre's model."""
    from music_ai.generation.generator import MusicGenerator
    
    # Get model path for the selected genre
    model_path = os.path.join('generos', genre_path, 'models', 'model.h5')
    if not os.path.exists(model_path):
        return None, f"❌ No se encontró el modelo para el género {get_genre_name(genre_path)}"
    
    try:
        # Initialize generator with the genre-specific model
        generator = MusicGenerator(model_path=model_path)
        
        # Generate beat
        output_path = f"generated/{genre_path.replace('/', '_')}_beat.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        generator.generate_beat(
            output_path=output_path,
            length=length
        )
        
        return output_path, f"✓ Beat generado exitosamente para {get_genre_name(genre_path)}"
    
    except Exception as e:
        return None, f"❌ Error al generar el beat: {str(e)}"

def create_gradio_interface():
    """Create the Gradio interface."""
    # Get available genres
    available_genres = get_available_genres()
    if not available_genres:
        raise ValueError("No se encontraron modelos entrenados. Por favor, entrena al menos un modelo antes de generar beats.")
    
    # Create genre dropdown
    genre_choices = [(name, path) for path, name in available_genres]
    
    with gr.Blocks(title="BeatNest - Generador de Beats") as interface:
        gr.Markdown("# 🎵 BeatNest - Generador de Beats")
        gr.Markdown("Selecciona un género y genera un beat único.")
        
        with gr.Row():
            with gr.Column():
                genre_dropdown = gr.Dropdown(
                    choices=genre_choices,
                    label="Selecciona un género",
                    value=genre_choices[0][1] if genre_choices else None
                )
                
                length_slider = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=100,
                    step=50,
                    label="Longitud del beat (pasos)"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperatura (creatividad)"
                )
                
                generate_btn = gr.Button("🎵 Generar Beat")
            
            with gr.Column():
                output_audio = gr.Audio(label="Beat Generado")
                status_text = gr.Textbox(label="Estado")
        
        generate_btn.click(
            fn=generate_beat,
            inputs=[genre_dropdown, length_slider, temperature_slider],
            outputs=[output_audio, status_text]
        )
    
    return interface

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Entrenar modelo o generar beats')
    parser.add_argument('--mode', choices=['train', 'generate'], default='generate',
                      help='Modo de operación: entrenar o generar beats')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--sequence-length', type=int, default=64, help='Longitud de la secuencia')
    parser.add_argument('--n-mels', type=int, default=128, help='Número de bandas mel')
    args = parser.parse_args()
    
    if args.mode == 'train':
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
    
    else:  # generate mode
        # Launch Gradio interface
        interface = create_gradio_interface()
        interface.launch(share=True)

if __name__ == '__main__':
    main() 