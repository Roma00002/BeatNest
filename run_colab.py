#!/usr/bin/env python3
import os
import sys
import gradio as gr
import numpy as np
import tensorflow as tf

# Añadir el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from music_ai.data.preprocessor import MusicPreprocessor
from music_ai.models.music_generator import MusicGenerator as Model
from music_ai.generation.generator import MusicGenerator

def print_training_instructions():
    """Print instructions for training the model in Colab."""
    instructions = """
🎵 BeatNest - Instrucciones de Entrenamiento 🎵

1. Primero, sube tus archivos MIDI a Google Colab:
   - En el panel izquierdo, haz clic en el icono de carpeta
   - Crea una carpeta llamada 'midi_files'
   - Sube tus archivos MIDI a esa carpeta

2. Ejecuta estos comandos para preparar el entorno:
   ```python
   # Clonar el repositorio
   !git clone https://github.com/Roma00002/BeatNest.git
   %cd BeatNest
   
   # Instalar dependencias
   !pip install -r requirements.txt
   ```

3. Para entrenar el modelo, simplemente ejecuta:
   ```python
   !python train_colab.py
   ```

4. Una vez entrenado, puedes iniciar la interfaz web:
   ```python
   !python run_colab.py
   ```

Para más información, visita: https://github.com/Roma00002/BeatNest
"""
    print(instructions)

class BeatGeneratorApp:
    def __init__(self):
        self.model_params = {
            'sample_rate': 44100,
            'hop_length': 512,
            'n_mels': 128,
            'sequence_length': 64,
            'units': 256,
            'num_layers': 3,
            'checkpoint_dir': 'checkpoints',
            'temperature': 1.0
        }
        
        # Create necessary directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('generated', exist_ok=True)
        
        # Initialize model and preprocessor
        try:
            self.preprocessor = MusicPreprocessor(
                sample_rate=self.model_params['sample_rate'],
                hop_length=self.model_params['hop_length'],
                n_mels=self.model_params['n_mels']
            )
            
            self.model = Model(
                input_shape=(self.model_params['n_mels'], self.model_params['sequence_length']),
                units=self.model_params['units'],
                num_layers=self.model_params['num_layers']
            )
            
            # Load latest checkpoint
            checkpoint = tf.train.Checkpoint(model=self.model)
            status = checkpoint.restore(tf.train.latest_checkpoint(self.model_params['checkpoint_dir']))
            status.expect_partial()
            
            self.generator = MusicGenerator(
                self.model,
                self.preprocessor,
                temperature=self.model_params['temperature']
            )
            
            print("✅ Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {str(e)}")
            print("Por favor, asegúrate de entrenar el modelo primero.")
            self.model = None
            self.preprocessor = None
            self.generator = None
    
    def generate_beat(self, description, temperature=1.0):
        """Generate a new beat based on the description."""
        try:
            if self.model is None:
                return None, "❌ Error: Primero debes entrenar el modelo. Ejecuta las instrucciones de entrenamiento en la consola de Colab."
            
            # Adjust parameters based on description
            description = description.lower()
            
            # Adjust temperature based on creativity desired
            if any(word in description for word in ["experimental", "creativo", "único", "original"]):
                temperature = 1.2
            elif any(word in description for word in ["conservador", "simple", "básico"]):
                temperature = 0.8
            
            # Update generator temperature
            self.generator.temperature = temperature
            
            # Generate beat
            output_path = os.path.join("generated", f"beat_{len(os.listdir('generated'))}.mid")
            self.generator.generate_beat(
                sequence_length=self.model_params['sequence_length'],
                output_path=output_path
            )
            
            return output_path, f"✅ Beat generado exitosamente: {os.path.basename(output_path)}"
            
        except Exception as e:
            return None, f"❌ Error durante la generación: {str(e)}"

def main():
    # Print training instructions
    print_training_instructions()
    
    # Initialize app
    app = BeatGeneratorApp()
    
    # Create interface
    with gr.Blocks(title="AI Music Beat Generator") as interface:
        gr.Markdown("# 🎵 AI Music Beat Generator")
        gr.Markdown("""
        ## Generación de Beats con IA
        
        Este es un generador de beats musicales basado en IA. Para usarlo:
        1. Asegúrate de haber entrenado el modelo primero usando las instrucciones de la consola
        2. Describe cómo quieres que sea tu beat
        3. ¡Haz clic en Generar Beat!
        """)
        
        with gr.Row():
            with gr.Column():
                description = gr.Textbox(
                    label="Describe cómo quieres que sea tu beat",
                    placeholder="Ejemplo: Un beat de hip-hop rápido y experimental con influencias electrónicas..."
                )
                generate_btn = gr.Button("🎵 Generar Beat")
                generation_output = gr.Textbox(label="Estado de la generación")
                audio_output = gr.Audio(label="Beat Generado", type="filepath")
        
        # Connect components
        generate_btn.click(
            app.generate_beat,
            inputs=[description],
            outputs=[audio_output, generation_output]
        )
    
    # Launch the interface
    interface.launch(share=True)

if __name__ == "__main__":
    main() 