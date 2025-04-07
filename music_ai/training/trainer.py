import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict
from tqdm import tqdm
import os

class MusicTrainer:
    def __init__(
        self,
        model: tf.keras.Model,
        learning_rate: float = 0.001,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Asegurar ruta absoluta para checkpoints
        if not os.path.isabs(checkpoint_dir):
            # Si estamos en Colab, usar la ruta completa
            if os.path.exists('/content/BeatNest'):
                checkpoint_dir = os.path.join('/content/BeatNest', checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"✓ Creado directorio de checkpoints: {os.path.abspath(checkpoint_dir)}")
        
        # Initialize checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model,
            epoch=tf.Variable(0)
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.checkpoint_dir,
            max_to_keep=3
        )
        
        # Load existing checkpoint if available
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"✓ Restaurado checkpoint: {self.checkpoint_manager.latest_checkpoint}")
            
            # Cargar modelo .h5 si existe
            h5_path = os.path.join(self.checkpoint_dir, 'modelo_entrenado.h5')
            if os.path.exists(h5_path):
                try:
                    self.model.load_weights(h5_path)
                    print(f"✓ Restaurado modelo desde: {h5_path}")
                except:
                    print("⚠️ No se pudo cargar el modelo .h5 anterior")
        else:
            print("✓ Iniciando entrenamiento desde cero")

    def prepare_batch(self, x: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Prepara un batch asegurando dimensiones consistentes."""
        # Asegurar que x e y tengan la misma longitud de secuencia
        seq_len = min(x.shape[2], y.shape[2])
        x = x[:, :, :seq_len]
        y = y[:, :, :seq_len]
        
        # Convertir a tensores
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        return x, y

    @tf.function
    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Ejecuta un paso de entrenamiento."""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x, training=True)
            
            # Calcular pérdida
            loss = self.loss_fn(y, predictions)
        
        # Calcular gradientes
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Aplicar gradientes
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def save_complete_model(self):
        """Guarda el modelo completo en formato .h5"""
        try:
            h5_path = os.path.join(self.checkpoint_dir, 'modelo_entrenado.h5')
            self.model.save(h5_path)
            print(f"✓ Modelo guardado en: {h5_path}")
        except Exception as e:
            print(f"⚠️ Error al guardar el modelo .h5: {str(e)}")

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 32,
        epochs: int = 1
    ) -> Dict[str, list]:
        """Entrena el modelo por un número específico de épocas."""
        X_train, y_train = train_data
        X_val, y_val = validation_data
        
        print(f"\nIniciando entrenamiento con {X_train.shape[0]} muestras")
        print(f"Forma de los datos: {X_train.shape}")
        print(f"Directorio de checkpoints: {os.path.abspath(self.checkpoint_dir)}")
        
        # Preparar datos de entrenamiento
        X_train, y_train = self.prepare_batch(X_train, y_train, batch_size)
        
        # Preparar datos de validación
        X_val, y_val = self.prepare_batch(X_val, y_val, batch_size)
        
        # Historiales de pérdida
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Entrenar por el número especificado de épocas
        for epoch in range(epochs):
            # Entrenar
            train_loss = self.train_step(X_train, y_train)
            
            # Validar
            val_predictions = self.model(X_val, training=False)
            val_loss = self.loss_fn(y_val, val_predictions)
            
            # Actualizar historiales
            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            
            # Actualizar época actual
            self.checkpoint.epoch.assign(epoch + 1)
            
            # Guardar checkpoint y modelo completo
            save_path = self.checkpoint_manager.save()
            self.save_complete_model()
            
            # Mostrar progreso
            print(f"Época {epoch + 1}/{epochs}")
            print(f"  - Pérdida entrenamiento: {float(train_loss):.4f}")
            print(f"  - Pérdida validación: {float(val_loss):.4f}")
            print(f"  - Checkpoint guardado en: {save_path}")
        
        # Verificar checkpoints al final del entrenamiento
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('ckpt-')]
        print(f"\n✓ Total de checkpoints guardados: {len(checkpoints)}")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
        
        # Verificar modelo .h5
        h5_path = os.path.join(self.checkpoint_dir, 'modelo_entrenado.h5')
        if os.path.exists(h5_path):
            print(f"✓ Modelo .h5 guardado en: {h5_path}")
        
        return history

    def load_latest_checkpoint(self) -> bool:
        """Load the latest checkpoint if available."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"✓ Restaurado checkpoint: {self.checkpoint_manager.latest_checkpoint}")
            print(f"  - Época actual: {int(self.checkpoint.epoch)}")
            
            # También intentar cargar el modelo .h5
            h5_path = os.path.join(self.checkpoint_dir, 'modelo_entrenado.h5')
            if os.path.exists(h5_path):
                try:
                    self.model.load_weights(h5_path)
                    print(f"✓ Restaurado modelo desde: {h5_path}")
                except:
                    print("⚠️ No se pudo cargar el modelo .h5")
            return True
        return False 