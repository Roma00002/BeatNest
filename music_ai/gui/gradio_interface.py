import gradio as gr
import os
import numpy as np
from music_ai.generation.generator import MusicGenerator
from music_ai.data.genres import get_genre_path, get_genre_name, GENRE_STRUCTURE

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

def create_interface():
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

if __name__ == "__main__":
    interface = create_interface()
    interface.launch() 