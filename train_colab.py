#!/usr/bin/env python3
import os
import sys
import argparse

# Añadir el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from music_ai.main import train

def main():
    """Main function for training in Colab."""
    print("🎵 Iniciando entrenamiento de BeatNest...")
    
    # Verificar que existen los directorios necesarios
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('generated', exist_ok=True)
    os.makedirs('midi_files', exist_ok=True)
    
    # Verificar que hay archivos MIDI
    midi_files = [f for f in os.listdir('midi_files') if f.endswith(('.mid', '.midi'))]
    if not midi_files:
        print("❌ Error: No se encontraron archivos MIDI en la carpeta 'midi_files'")
        print("Por favor, sube tus archivos MIDI a la carpeta 'midi_files' antes de entrenar")
        return
    
    print(f"✅ Encontrados {len(midi_files)} archivos MIDI para entrenamiento")
    
    # Configurar parámetros
    args = argparse.Namespace(
        data_dir='midi_files',           # Carpeta donde están tus archivos MIDI
        checkpoint_dir='checkpoints',     # Carpeta donde se guardará el modelo
        sample_rate=44100,               # Tasa de muestreo (no cambiar)
        hop_length=512,                  # Longitud de salto (no cambiar)
        n_mels=128,                      # Número de bandas mel (no cambiar)
        sequence_length=64,              # Longitud de secuencia (no cambiar)
        units=256,                       # Unidades LSTM (no cambiar)
        num_layers=3,                    # Capas LSTM (no cambiar)
        learning_rate=0.001,             # Velocidad de aprendizaje
        batch_size=32,                   # Tamaño del batch
        epochs=100                       # Número de épocas
    )
    
    # Iniciar entrenamiento
    try:
        train(args)
        print("✅ Entrenamiento completado exitosamente")
        print("\nAhora puedes generar beats usando:")
        print("!python run_colab.py")
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        print("\nPor favor, verifica:")
        print("1. Que los archivos MIDI sean válidos")
        print("2. Que tengas suficiente memoria disponible")
        print("3. Que la GPU esté habilitada en Colab")

if __name__ == "__main__":
    main() 