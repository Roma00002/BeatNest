#!/usr/bin/env python3
import os
import gradio as gr
import numpy as np
import tensorflow as tf
from music_ai.data.preprocessor import MusicPreprocessor
from music_ai.models.music_generator import MusicGenerator as Model
from music_ai.training.trainer import MusicTrainer
from music_ai.generation.generator import MusicGenerator

class BeatGeneratorApp:
    def __init__(self):
        self.model_params = {
            'sample_rate': 44100,
            'hop_length': 512,
            'n_mels': 128,
            'sequence_length': 64,
            'units': 256,
            'num_layers': 3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'checkpoint_dir': 'checkpoints',
            'temperature': 1.0
        }
        
        # Create necessary directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('generated', exist_ok=True)
        
        self.preprocessor = None
        self.model = None
        self.trainer = None
        
    def train_model(self, files, genre, epochs, batch_size, progress=gr.Progress()):
        """Train the model with the uploaded MIDI files."""
        try:
            # Save uploaded files
            data_dir = "uploaded_midi"
            os.makedirs(data_dir, exist_ok=True)
            for file in files:
                os.system(f'cp "{file.name}" "{data_dir}"')
            
            # Update parameters
            self.model_params['epochs'] = epochs
            self.model_params['batch_size'] = batch_size
            
            # Initialize preprocessor
            self.preprocessor = MusicPreprocessor(
                sample_rate=self.model_params['sample_rate'],
                hop_length=self.model_params['hop_length'],
                n_mels=self.model_params['n_mels']
            )
            
            # Load dataset
            X, y = self.preprocessor.load_dataset(
                data_dir,
                sequence_length=self.model_params['sequence_length']
            )
            
            # Split dataset
            split_idx = int(len(X) * 0.9)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Initialize model
            input_shape = (self.model_params['n_mels'], X.shape[2])
            self.model = Model(
                input_shape=input_shape,
                units=self.model_params['units'],
                num_layers=self.model_params['num_layers']
            )
            
            # Initialize trainer
            self.trainer = MusicTrainer(
                self.model,
                learning_rate=self.model_params['learning_rate'],
                checkpoint_dir=self.model_params['checkpoint_dir']
            )
            
            # Train model
            history = []
            for epoch in range(epochs):
                metrics = self.trainer.train(
                    train_data=(X_train, y_train),
                    validation_data=(X_val, y_val),
                    batch_size=batch_size,
                    epochs=1
                )
                progress(epoch / epochs, f"Epoch {epoch + 1}/{epochs}")
                history.append(metrics)
            
            return f"✅ Entrenamiento completado exitosamente.\nGénero: {genre}\nÉpocas: {epochs}\nBatch size: {batch_size}"
            
        except Exception as e:
            return f"❌ Error durante el entrenamiento: {str(e)}"
    
    def generate_beat(self, description, temperature=1.0):
        """Generate a new beat based on the description."""
        try:
            if self.model is None:
                return None, "❌ Error: Primero debes entrenar el modelo."
            
            # Adjust parameters based on description
            description = description.lower()
            
            # Adjust temperature based on creativity desired
            if any(word in description for word in ["experimental", "creativo", "único", "original"]):
                temperature = 1.2
            elif any(word in description for word in ["conservador", "simple", "básico"]):
                temperature = 0.8
            
            # Initialize generator
            generator = MusicGenerator(
                self.model,
                self.preprocessor,
                temperature=temperature
            )
            
            # Generate beat
            output_path = os.path.join("generated", f"beat_{len(os.listdir('generated'))}.mid")
            generator.generate_beat(
                sequence_length=self.model_params['sequence_length'],
                output_path=output_path
            )
            
            return output_path, f"✅ Beat generado exitosamente: {os.path.basename(output_path)}"
            
        except Exception as e:
            return None, f"❌ Error durante la generación: {str(e)}"

def main():
    app = BeatGeneratorApp()
    
    with gr.Blocks(title="AI Music Beat Generator") as interface:
        gr.Markdown("# 🎵 AI Music Beat Generator")
        
        with gr.Tab("Entrenamiento"):
            gr.Markdown("## 1. Entrenamiento del Modelo")
            with gr.Row():
                with gr.Column():
                    files = gr.File(
                        file_count="multiple",
                        label="Archivos MIDI para entrenamiento",
                        file_types=[".mid", ".midi"]
                    )
                    genre = gr.Dropdown(
                        choices=[
                            "Hip-Hop",
                            "Electrónica",
                            "Rock",
                            "Jazz",
                            "Pop",
                            "Clásica",
                            "Latina",
                            "Experimental",
                            "Otro"
                        ],
                        label="Género Musical",
                        value="Hip-Hop"
                    )
                    epochs = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=100,
                        step=1,
                        label="Épocas de entrenamiento"
                    )
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=32,
                        step=1,
                        label="Tamaño de batch"
                    )
                    train_btn = gr.Button("🚀 Iniciar Entrenamiento")
                    train_output = gr.Textbox(label="Estado del entrenamiento")
        
        with gr.Tab("Generación"):
            gr.Markdown("## 2. Generación de Beats")
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
        train_btn.click(
            app.train_model,
            inputs=[files, genre, epochs, batch_size],
            outputs=[train_output]
        )
        
        generate_btn.click(
            app.generate_beat,
            inputs=[description],
            outputs=[audio_output, generation_output]
        )
    
    # Launch the interface
    interface.launch(share=True)

if __name__ == "__main__":
    main() 