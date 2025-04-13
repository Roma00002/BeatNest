#!/usr/bin/env python3
import os
import sys
import argparse
from music_ai.training.trainer import MusicTrainer
from music_ai.data.preprocessor import MusicPreprocessor
from music_ai.data.genres import get_genre_path, create_genre_directories, get_genre_name, GENRE_STRUCTURE
import gradio as gr
import numpy as np

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

def setup_environment(genre_path: str = None) -> str:
    """Setup the environment for training or generation."""
    # Create necessary directories
    os.makedirs('mp3_files', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create genre directories if specified
    if genre_path:
        project_path = get_genre_path(genre_path)
        create_genre_directories(project_path)
        print(f"\n=== Directorios creados ===")
        print(f"Ruta del proyecto: {project_path}")
        print(f"Ruta de modelos: {os.path.join(project_path, 'models')}")
        print(f"Ruta de audio: {os.path.join(project_path, 'audio')}")
        return project_path
    
    return None

def find_audio_files(audio_path: str = None) -> list:
    """Find audio files in the specified directory."""
    if not audio_path:
        audio_path = 'mp3_files'
    
    # Check if directory exists
    if not os.path.exists(audio_path):
        print(f"El directorio {audio_path} no existe")
        print("Por favor, crea el directorio y sube tus archivos MP3")
        return []
    
    # Find audio files
    audio_files = []
    for file in os.listdir(audio_path):
        if file.endswith(('.mp3', '.wav', '.ogg')):
            audio_files.append(os.path.join(audio_path, file))
    
    if not audio_files:
        print(f"No se encontraron archivos de audio en {audio_path}")
        print("Por favor, sube tus archivos MP3 a la carpeta")
        print("En Google Colab, puedes usar el siguiente comando:")
        print("!cp /content/drive/MyDrive/tus_archivos/*.mp3 mp3_files/")
    
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
                        model_path = os.path.join('generos', path, 'models', 'model.weights.h5')
                        if os.path.exists(model_path):
                            genres.append((path, get_genre_name(path)))
                else:
                    path = f"{genre_key}/{subgenre_key}"
                    model_path = os.path.join('generos', path, 'models', 'model.weights.h5')
                    if os.path.exists(model_path):
                        genres.append((path, get_genre_name(path)))
        else:
            path = genre_key
            model_path = os.path.join('generos', path, 'models', 'model.weights.h5')
            if os.path.exists(model_path):
                genres.append((path, get_genre_name(path)))
    return genres

def generate_beat(genre_path: str, length: int = 100, temperature: float = 1.0):
    """Generate a beat using the selected genre's model."""
    from music_ai.generation.generator import MusicGenerator
    
    # Get model path for the selected genre
    model_path = os.path.join('generos', genre_path, 'models', 'model.weights.h5')
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
        # Force TensorFlow to use CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow as tf
        
        # Configure TensorFlow to use CPU
        tf.config.set_soft_device_placement(True)
        physical_devices = tf.config.list_physical_devices('CPU')
        if physical_devices:
            try:
                # Set memory growth for CPU devices
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except ValueError:
                # If memory growth is not supported, just continue
                pass
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='BeatNest - Entrenamiento y Generación de Beats')
        parser.add_argument('--mode', choices=['train', 'generate'], default='generate',
                          help='Modo de operación: train (entrenar) o generate (generar)')
        parser.add_argument('--genre', type=str, help='Género a entrenar (ejemplo: hiphop_rap/trap)')
        parser.add_argument('--songs_per_batch', type=int, default=10,
                          help='Número de canciones a procesar por lote')
        args = parser.parse_args()
        
        if args.mode == 'train':
            # Use provided genre or ask for it
            if args.genre:
                genre_path = args.genre
            else:
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
            models_dir = os.path.join(project_path, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
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
            
            # Initialize preprocessor and trainer
            preprocessor = MusicPreprocessor(n_mels=128, sr=22050)
            trainer = None
            model_weights_path = os.path.join(models_dir, 'model.weights.h5')
            
            # Process and train in smaller batches
            batch_size = args.songs_per_batch  # Use provided batch size or default
            for i in range(0, len(audio_files), batch_size):
                batch_files = audio_files[i:i + batch_size]
                print(f"\n=== Procesando lote {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size} ===")
                print(f"Procesando {len(batch_files)} canciones...")
                
                try:
                    # Process batch of songs (1 at a time to manage memory)
                    all_sequences = []
                    all_targets = []
                    
                    for audio_file in batch_files:
                        print(f"\nProcesando: {audio_file}")
                        
                        # Clear memory before processing
                        import gc
                        gc.collect()
                        
                        try:
                            X, y = preprocessor.load_dataset(
                                audio_path,
                                sequence_length=50,
                                batch_size=4,
                                specific_files=[audio_file]
                            )
                            all_sequences.extend(X)
                            all_targets.extend(y)
                            
                            # Clear memory after processing
                            del X, y
                            gc.collect()
                            
                        except Exception as e:
                            print(f"Error procesando {audio_file}: {str(e)}")
                            continue
                    
                    # Convert lists to numpy arrays
                    X = np.array(all_sequences)
                    y = np.array(all_targets)
                    
                    print(f"\n✓ Lote procesado exitosamente!")
                    print(f"Forma del dataset de entrada: {X.shape}")
                    print(f"Forma del dataset objetivo: {y.shape}")
                    
                    # Initialize or load trainer
                    if trainer is None:
                        print("\n=== Inicializando entrenador ===")
                        print("Configurando el modelo con los siguientes parámetros:")
                        print(f"- Unidades LSTM: 32")
                        print(f"- Número de capas: 2")
                        print(f"- Tasa de dropout: 0.2")
                        print(f"- Tasa de aprendizaje: 0.001")
                        
                        trainer = MusicTrainer(
                            input_shape=(X.shape[1], X.shape[2]),
                            units=32,
                            num_layers=2,
                            dropout_rate=0.2,
                            learning_rate=0.001
                        )
                        print("✓ Entrenador inicializado correctamente")
                        
                        # Try to load existing weights if they exist
                        if os.path.exists(model_weights_path):
                            print("\n=== Cargando pesos existentes ===")
                            trainer.model.load_weights(model_weights_path)
                            print("✓ Pesos cargados correctamente")
                    else:
                        print("\n=== Cargando modelo existente ===")
                        trainer.model.load_weights(model_weights_path)
                        print("✓ Modelo cargado correctamente")
                    
                    # Train on current batch
                    print("\n=== Iniciando entrenamiento del lote ===")
                    print(f"Entrenando con {len(batch_files)} canciones...")
                    
                    history = trainer.train(
                        X, y,
                        epochs=20,
                        batch_size=4,
                        validation_split=0.2,
                        checkpoint_dir=models_dir
                    )
                    
                    print(f"\n✓ Modelo actualizado guardado en: {model_weights_path}")
                    print(f"Pérdida final de entrenamiento: {history['train_loss'][-1]:.4f}")
                    print(f"Pérdida final de validación: {history['val_loss'][-1]:.4f}")
                    
                    # Clear memory after training
                    del X, y, history
                    gc.collect()
                    
                except Exception as e:
                    print(f"\nError durante el procesamiento del lote: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print("\nEntrenamiento completado exitosamente!")
            print(f"Modelo final guardado en: {model_weights_path}")
        
        else:  # generate mode
            # Create and launch Gradio interface
            interface = create_gradio_interface()
            interface.launch(share=True)  # Make the interface public
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 