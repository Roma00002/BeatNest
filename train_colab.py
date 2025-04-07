#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import argparse

# Añadir el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from music_ai.main import train

def find_and_move_midi_files():
    """Find MIDI files in common locations and move them to midi_files directory."""
    # Create midi_files directory if it doesn't exist
    os.makedirs('midi_files', exist_ok=True)
    
    # Common locations to look for MIDI files
    search_paths = [
        'midi_files',           # Current midi_files directory
        '.',                    # Current directory
        '..',                   # Parent directory
        '/content',             # Colab root directory
        '/content/BeatNest/midi_files',  # Project midi_files directory
        '/content/sample_data'  # Colab sample data directory
    ]
    
    midi_files_found = []
    
    # Search for MIDI files in all locations
    for path in search_paths:
        if os.path.exists(path):
            # Search for both .mid and .midi files
            for pattern in ['*.mid', '*.midi']:
                midi_pattern = os.path.join(path, pattern)
                midi_files_found.extend(glob.glob(midi_pattern))
    
    if not midi_files_found:
        print("\n❌ No se encontraron archivos MIDI en ninguna ubicación común.")
        print("\nPor favor, verifica:")
        print(f"1. Que los archivos estén en: {os.path.abspath('midi_files')}")
        print("2. Que los archivos terminen en .mid o .midi")
        print("3. Que tengas permisos de lectura en los archivos")
        print("\nUbicaciones buscadas:")
        for path in search_paths:
            if os.path.exists(path):
                print(f"  ✓ {os.path.abspath(path)}")
            else:
                print(f"  ✗ {path} (no existe)")
        return False
    
    # Move all found MIDI files to midi_files directory
    files_moved = 0
    for file_path in midi_files_found:
        file_name = os.path.basename(file_path)
        destination = os.path.join('midi_files', file_name)
        if os.path.abspath(file_path) != os.path.abspath(destination):
            try:
                shutil.copy2(file_path, destination)
                print(f"✓ Copiado: {file_name}")
                files_moved += 1
            except Exception as e:
                print(f"✗ Error al copiar {file_name}: {str(e)}")
    
    if files_moved == 0 and len(midi_files_found) > 0:
        print("\n✓ Los archivos MIDI ya están en la ubicación correcta")
    
    return True

def main():
    """Main function for training in Colab."""
    print("🎵 Iniciando entrenamiento de BeatNest...")
    print("\nVerificando archivos MIDI...")
    
    # Verificar y mover archivos MIDI si es necesario
    if not find_and_move_midi_files():
        return
    
    # Verificar que hay archivos MIDI en la carpeta correcta
    midi_files = [f for f in os.listdir('midi_files') if f.endswith(('.mid', '.midi'))]
    if not midi_files:
        print("❌ Error: No se encontraron archivos MIDI en la carpeta 'midi_files'")
        return
    
    print(f"\n✅ Encontrados {len(midi_files)} archivos MIDI para entrenamiento:")
    for file in midi_files:
        print(f"  - {file}")
    
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
    
    # Crear directorio para checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Iniciar entrenamiento
    try:
        print("\n🚀 Iniciando entrenamiento...")
        train(args)
        print("\n✅ Entrenamiento completado exitosamente")
        print("\nAhora puedes generar beats usando:")
        print("!python run_colab.py")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        print("\nPor favor, verifica:")
        print("1. Que los archivos MIDI sean válidos")
        print("2. Que tengas suficiente memoria disponible")
        print("3. Que la GPU esté habilitada en Colab")
        print("\nPara habilitar la GPU en Colab:")
        print("1. Ve a 'Entorno de ejecución' en el menú")
        print("2. Selecciona 'Cambiar tipo de entorno de ejecución'")
        print("3. Elige 'GPU' en el desplegable")
        print("4. Haz clic en 'Guardar'")

if __name__ == "__main__":
    main() 