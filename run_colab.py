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
    # Get full genre path
    genre_dir = get_genre_path(genre_path)
    if not genre_dir:
        print(f"❌ Género no válido: {genre_path}")
        print("Usa el formato: genero/subgenero o genero/subgenero/subsubgenero")
        sys.exit(1)
    
    # Create the genre directory structure if it doesn't exist
    base_dir = 'generos'
    genre_full_path = os.path.join(base_dir, genre_dir)
    os.makedirs(os.path.join(genre_full_path, 'models'), exist_ok=True)
    
    print("\n" + "="*50)
    print(f"✓ Entrenando para el género: {get_genre_name(genre_path)}")
    print(f"✓ Los modelos se guardarán en: {os.path.abspath(os.path.join(genre_full_path, 'models'))}")
    print("\n📁 Por favor, ingresa la ruta donde están tus archivos MP3 para entrenar")
    print("Ejemplo: /content/drive/MyDrive/mis_archivos_mp3")
    print("="*50 + "\n")
    
    return genre_full_path

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Entrenar modelo o generar beats')
    parser.add_argument('--mode', choices=['train', 'generate'], default='generate',
                      help='Modo de operación: entrenar o generar beats')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--sequence-length', type=int, default=64, help='Longitud de la secuencia')
    parser.add_argument('--n-mels', type=int, default=128, help='Número de bandas mel')
    parser.add_argument('--process-batch-size', type=int, default=3, help='Número de archivos a procesar a la vez')
    args = parser.parse_args()
    
    if args.mode == 'train':
        try:
            # Select genre
            genre_path = select_genre()
            
            # Setup environment
            genre_dir = setup_environment(genre_path)
            
            # Ask for audio files path
            audio_path = input("\nIngresa la ruta donde están tus archivos MP3: ").strip()
            audio_files = find_audio_files(audio_path)
            if not audio_files:
                sys.exit(1)
            
            # Initialize preprocessor
            preprocessor = MusicPreprocessor(n_mels=args.n_mels)
            
            # Load and preprocess dataset in batches
            print("\nCargando y preprocesando archivos de audio...")
            print("Este proceso puede tardar varios minutos dependiendo del número de archivos.")
            print("Se procesarán los archivos en lotes para evitar problemas de memoria.")
            
            X, y = preprocessor.load_dataset(
                audio_path,
                sequence_length=args.sequence_length,
                batch_size=args.process_batch_size
            )
            
            # Print dataset information
            print(f"\nDataset cargado exitosamente:")
            print(f"- Forma de entrada: {X.shape}")
            print(f"- Forma de objetivo: {y.shape}")
            print(f"- Número de muestras: {X.shape[0]}")
            
            # Initialize trainer with memory-efficient settings
            print("\nInicializando el modelo...")
            trainer = MusicTrainer(
                input_shape=(args.n_mels, args.sequence_length),
                units=256,
                num_layers=3,
                dropout_rate=0.3,
                batch_size=args.batch_size
            )
            
            # Train model with memory monitoring
            print("\nIniciando entrenamiento...")
            print("El entrenamiento se realizará con las siguientes optimizaciones:")
            print("- Procesamiento por lotes para reducir el uso de memoria")
            print("- Limpieza de memoria después de cada lote")
            print("- Guardado automático de checkpoints")
            
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
            
            # Clear memory after training
            del X, y, trainer, history
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"\nError durante el entrenamiento: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Always launch Gradio interface
    print("\nIniciando interfaz de generación de beats...")
    interface = create_gradio_interface()
    interface.launch(share=True)

if __name__ == '__main__':
    main() 