#!/usr/bin/env python3
import os
import sys
import argparse
from music_ai.training.trainer import MusicTrainer
from music_ai.data.preprocessor import MusicPreprocessor
from music_ai.data.genres import get_genre_path, create_genre_directories, get_genre_name, GENRE_STRUCTURE
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
import gc

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
    
    # Check for both model file types
    model_weights_path = os.path.join('generos', genre_path, 'models', 'model.weights.h5')
    model_path = os.path.join('generos', genre_path, 'models', 'model.h5')
    
    # Determine which model file to use
    if os.path.exists(model_path):
        use_model_path = model_path
    elif os.path.exists(model_weights_path):
        use_model_path = model_weights_path
    else:
        return None, f"❌ No se encontró el modelo para el género {get_genre_name(genre_path)}"
    
    try:
        # Initialize generator with the genre-specific model
        generator = MusicGenerator(model_path=use_model_path)
        
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
        
        # Enable eager execution
        tf.config.run_functions_eagerly(True)
        
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
        parser.add_argument('--model_path', type=str, help='Ruta al modelo existente para continuar entrenamiento')
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
            project_path = get_genre_path(genre_path)
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
            model_path = os.path.join(models_dir, 'model.h5')
            
            # Ask if user wants to load an existing model
            load_existing = input("\n¿Deseas cargar un modelo existente? (s/n): ").strip().lower()
            
            # Check if we should load an existing model
            if args.model_path:
                model_path = args.model_path
                load_existing = 's'
                print(f"\nUtilizando ruta de modelo especificada: {model_path}")
            
            if load_existing == 's':
                custom_path = input("\n¿Quieres especificar una ruta diferente para el modelo? (s/n): ").strip().lower()
                if custom_path == 's':
                    custom_model_path = input("\nIngresa la ruta completa del modelo o carpeta: ").strip()
                    
                    # Check if it's a directory and find model files
                    if os.path.isdir(custom_model_path):
                        possible_models = [
                            os.path.join(custom_model_path, 'model.h5'),
                            os.path.join(custom_model_path, 'model.weights.h5'),
                            os.path.join(custom_model_path, 'model.keras')
                        ]
                        
                        found = False
                        for model_file in possible_models:
                            if os.path.exists(model_file):
                                custom_model_path = model_file
                                found = True
                                print(f"\nEncontrado archivo de modelo: {custom_model_path}")
                                break
                        
                        if not found:
                            print(f"\n❌ No se encontraron archivos de modelo en la carpeta: {custom_model_path}")
                            print(f"Se usará la ruta predeterminada: {model_path}")
                            custom_model_path = model_path
                    
                    # Check if the model exists at the specified path
                    if os.path.exists(custom_model_path):
                        model_path = custom_model_path
                    else:
                        print(f"\n❌ No se encontró el modelo en: {custom_model_path}")
                        print(f"Se usará la ruta predeterminada: {model_path}")
                
                # Check if model exists at selected path
                if os.path.exists(model_path):
                    print(f"\n=== Cargando modelo existente desde: {model_path} ===")
                    
                    # Try loading the model in a loop to allow retries
                    loading_success = False
                    while not loading_success:
                        try:
                            # Try loading complete model first
                            model = load_model(model_path)
                            trainer = MusicTrainer(model=model)
                            print("✓ Modelo cargado correctamente")
                            loading_success = True
                            
                            # Store the original model path to save back to the same location
                            original_model_path = model_path
                        except Exception as e:
                            print(f"\n❌ Error al cargar el modelo completo: {str(e)}")
                            
                            # If that fails, try loading weights only
                            if model_path.endswith('.weights.h5'):
                                try:
                                    print("Intentando cargar solo los pesos del modelo...")
                                    trainer = MusicTrainer(input_shape=(50, 128))
                                    trainer.model.load_weights(model_path)
                                    print("✓ Pesos del modelo cargados correctamente")
                                    loading_success = True
                                    
                                    # Store the original model path to save back to the same location
                                    original_model_path = model_path
                                    continue
                                except Exception as e2:
                                    print(f"❌ Error al cargar los pesos: {str(e2)}")
                            
                            # Ask user what to do next
                            retry_option = input("\n¿Qué deseas hacer?\n1. Intentar con otra ruta\n2. Crear un nuevo modelo\nSelecciona una opción (1/2): ").strip()
                            
                            if retry_option == '1':
                                new_path = input("\nIngresa la nueva ruta del modelo: ").strip()
                                if os.path.exists(new_path):
                                    model_path = new_path
                                    print(f"\n=== Intentando cargar modelo desde: {model_path} ===")
                                    # Continue to the next iteration of the loop
                                else:
                                    print(f"\n❌ No se encontró el modelo en: {new_path}")
                                    # Ask again in the next iteration
                            else:
                                print("\nInicializando nuevo modelo...")
                                trainer = MusicTrainer(input_shape=(50, 128))
                                print("✓ Modelo inicializado correctamente")
                                loading_success = True
                else:
                    print(f"\n❌ No se encontró el modelo en: {model_path}")
                    print("Inicializando nuevo modelo...")
                    trainer = None
            
            # Initialize trainer if it doesn't exist
            if trainer is None:
                print("\n=== Inicializando nuevo modelo ===")
                # Use (50, 128) for input shape to match the processed data
                trainer = MusicTrainer(
                    input_shape=(50, 128),
                    units=64,  # Increased units for better learning
                    num_layers=2,
                    dropout_rate=0.2,
                    learning_rate=0.002  # Slightly higher learning rate
                )
                print("✓ Modelo inicializado correctamente")
            
            # Process and train in batches
            batch_size = args.songs_per_batch
            total_batches = (len(audio_files) + batch_size - 1) // batch_size
            
            # Loop through all batches
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(audio_files))
                batch_files = audio_files[start_idx:end_idx]
                
                print(f"\n=== Procesando lote {batch_num + 1} de {total_batches} ===")
                print(f"Procesando canciones: {', '.join(os.path.basename(f) for f in batch_files)}")
                
                # Process current batch to get training data
                X, y = preprocessor.process_batch(batch_files)
                
                if X is not None and y is not None:
                    print(f"✓ Datos procesados correctamente")
                    print(f"✓ Forma de los datos de entrada: {X.shape}")
                    print(f"✓ Forma de los datos objetivo: {y.shape}")
                    
                    # Limit the number of sequences to avoid excessive training time
                    max_sequences = 1500  # Increased from 1000 to get more training examples
                    if X.shape[0] > max_sequences:
                        print(f"\n⚠️ Limitando número de secuencias de {X.shape[0]} a {max_sequences} para reducir tiempo de entrenamiento")
                        indices = np.random.choice(X.shape[0], max_sequences, replace=False)
                        X = X[indices]
                        y = y[indices]
                    
                    # Train the model with a few more epochs per batch to improve learning
                    print("\n=== Entrenando modelo ===")
                    
                    # Use original path for checkpoints if we loaded an existing model
                    checkpoint_dir = os.path.dirname(original_model_path) if 'original_model_path' in locals() and original_model_path else models_dir
                    
                    history = trainer.train(
                        X, y,
                        epochs=3,  # Train for 3 epochs per batch instead of 1
                        batch_size=32,  # Increased batch size for better gradient estimation
                        validation_split=0.2,
                        checkpoint_dir=checkpoint_dir
                    )
                    
                    # If we loaded an existing model, save back to the original location
                    if 'original_model_path' in locals() and original_model_path:
                        print(f"\n=== Guardando modelo actualizado en la ubicación original ===")
                        if original_model_path.endswith('.h5'):
                            trainer.model.save(original_model_path)
                            print(f"✓ Modelo guardado en: {original_model_path}")
                        
                        # Also save weights if it was a weights file
                        if original_model_path.endswith('.weights.h5'):
                            trainer.model.save_weights(original_model_path)
                            print(f"✓ Pesos guardados en: {original_model_path}")
                    
                    # Display training metrics
                    print("\n=== Métricas de entrenamiento ===")
                    print(f"✓ Pérdida de entrenamiento: {history['train_loss'][-1]:.4f}")
                    print(f"✓ Pérdida de validación: {history['val_loss'][-1]:.4f}")
                    
                    # Clean up memory
                    del X, y
                    gc.collect()
                    print("✓ Memoria liberada")
                else:
                    print("❌ No se pudieron procesar las canciones en este lote")
            
            # Use the original model path in the final message if we loaded an existing model
            final_model_dir = os.path.dirname(original_model_path) if 'original_model_path' in locals() and original_model_path else models_dir
            
            # Save the final model
            if 'original_model_path' in locals() and original_model_path:
                # Save to the original location
                final_model_path = original_model_path
            else:
                # Save to the models directory
                final_model_path = os.path.join(models_dir, 'model.h5')
                
            # Always save both the complete model and weights
            print("\n=== Guardando modelo final ===")
            trainer.model.save(final_model_path)
            trainer.model.save_weights(final_model_path.replace('.h5', '.weights.h5'))
            
            print("\n=== Entrenamiento completado exitosamente! ===")
            print(f"El modelo final se guardó en: {final_model_dir}")
        
        else:  # generate mode
            # Create and launch Gradio interface
            interface = create_gradio_interface()
            interface.launch(share=True)  # Make the interface public
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 