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

def setup_environment(genre_path: str) -> str:
    """
    Setup the environment for training a specific genre.
    
    Args:
        genre_path (str): Path to the genre (e.g., 'hiphop_rap/trap')
        
    Returns:
        str: Path to the genre directory
    """
    # Create base directories
    base_dir = 'generos'
    genre_dir = os.path.join(base_dir, genre_path)
    
    # Create directory structure
    os.makedirs(os.path.join(genre_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(genre_dir, 'audio'), exist_ok=True)
    
    print(f"\nEstructura de directorios creada en: {genre_dir}")
    print(f"- Modelos: {os.path.join(genre_dir, 'models')}")
    print(f"- Audio: {os.path.join(genre_dir, 'audio')}")
    
    return genre_dir

def find_audio_files(audio_path: str):
    """Find audio files in the specified directory."""
    if not os.path.exists(audio_path):
        print(f"❌ La ruta no existe: {audio_path}")
        return None
    
    audio_files = []
    for file in os.listdir(audio_path):
        if file.endswith(('.mp3', '.wav', '.ogg')):
            audio_files.append(os.path.join(audio_path, file))
    
    if not audio_files:
        print(f"❌ No se encontraron archivos de audio en: {audio_path}")
        print("Asegúrate de que los archivos tengan extensión .mp3, .wav o .ogg")
        return None
    
    print(f"\n✓ Encontrados {len(audio_files)} archivos de audio:")
    for file in audio_files:
        print(f"- {os.path.basename(file)}")
    
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
    
    # Create genre dropdown
    genre_choices = [(name, path) for path, name in available_genres] if available_genres else []
    
    with gr.Blocks(title="BeatNest - Generador de Beats") as interface:
        gr.Markdown("# 🎵 BeatNest - Generador de Beats")
        gr.Markdown("Selecciona un género y genera un beat único.")
        
        with gr.Row():
            with gr.Column():
                genre_dropdown = gr.Dropdown(
                    choices=genre_choices,
                    label="Selecciona un género",
                    value=genre_choices[0][1] if genre_choices else None,
                    interactive=bool(genre_choices)  # Deshabilitar si no hay géneros
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
                
                generate_btn = gr.Button("🎵 Generar Beat", interactive=bool(genre_choices))  # Deshabilitar si no hay géneros
            
            with gr.Column():
                output_audio = gr.Audio(label="Beat Generado")
                status_text = gr.Textbox(label="Estado")
        
        if not genre_choices:
            gr.Markdown("""
            ## ⚠️ No hay modelos entrenados disponibles
            
            Para generar beats, primero necesitas entrenar un modelo:
            1. Ejecuta `!python run_colab.py --mode train`
            2. Selecciona un género
            3. Sube archivos de audio para entrenar
            4. Espera a que se complete el entrenamiento
            
            Una vez que tengas modelos entrenados, podrás generar beats.
            """)
        
        generate_btn.click(
            fn=generate_beat,
            inputs=[genre_dropdown, length_slider, temperature_slider],
            outputs=[output_audio, status_text]
        )
    
    return interface

def main():
    """Main function to run the training process."""
    try:
        # Select genre
        print("\n=== Selección de género ===")
        print("Géneros disponibles:")
        for genre, subgenres in GENRE_STRUCTURE.items():
            print(f"\n{genre}:")
            for subgenre in subgenres:
                print(f"  - {subgenre}")
        
        genre_path = input("\nIngresa el género y subgénero (ejemplo: hiphop_rap/trap): ").strip()
        if not genre_path:
            print("Debes especificar un género válido")
            return
        
        # Setup environment
        project_path = setup_environment(genre_path)
        
        # Ask for audio files path
        print("\n=== Buscando archivos de audio ===")
        audio_path = input("\nIngresa la ruta donde están tus archivos MP3: ").strip()
        audio_files = find_audio_files(audio_path)
        if not audio_files:
            print(f"No se encontraron archivos de audio en {audio_path}")
            print("Por favor, sube tus archivos MP3 a la carpeta correspondiente")
            return
        
        print(f"\nEncontrados {len(audio_files)} archivos de audio")
        print("Archivos encontrados:")
        for file in audio_files:
            print(f"- {file}")
        
        # Initialize preprocessor with memory-efficient settings
        preprocessor = MusicPreprocessor(
            sample_rate=22050,  # Reduced from 44100
            n_mels=128,         # Reduced from 256
            hop_length=512,     # Increased from 256
            n_fft=2048,         # Increased from 1024
            sequence_length=50  # Reduced from 100
        )
        
        # Load and preprocess dataset with smaller batch size
        X, y = preprocessor.load_and_preprocess_dataset(
            audio_files,
            batch_size=16,      # Reduced from 32
            songs_per_batch=5   # Reduced from 10
        )
        
        print(f"\nDataset shapes:")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Initialize trainer with memory-efficient settings
        trainer = MusicTrainer(
            input_shape=(X.shape[1], X.shape[2]),
            units=128,          # Reduced from 256
            num_layers=2,
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Train model with reduced epochs
        history = trainer.train(
            X, y,
            epochs=50,          # Reduced from 100
            batch_size=16,      # Reduced from 32
            validation_split=0.2,
            checkpoint_dir=os.path.join(project_path, 'models')
        )
        
        print("\nTraining completed!")
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 