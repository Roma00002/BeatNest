#!/usr/bin/env python3
import os
import gradio as gr
from music_ai.data.genres import get_genre_name, GENRE_STRUCTURE

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

def create_interface():
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
                    interactive=bool(genre_choices)
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
                
                generate_btn = gr.Button("🎵 Generar Beat", interactive=bool(genre_choices))
            
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

if __name__ == "__main__":
    # Force TensorFlow to use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Create and launch interface
    interface = create_interface()
    interface.launch() 