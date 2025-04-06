import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
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
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Initialize checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.checkpoint_dir,
            max_to_keep=3
        )

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

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 32,
        epochs: int = 1
    ) -> float:
        """Entrena el modelo por un número específico de épocas."""
        X_train, y_train = train_data
        X_val, y_val = validation_data
        
        # Preparar datos de entrenamiento
        X_train, y_train = self.prepare_batch(X_train, y_train, batch_size)
        
        # Preparar datos de validación
        X_val, y_val = self.prepare_batch(X_val, y_val, batch_size)
        
        # Entrenar por el número especificado de épocas
        for epoch in range(epochs):
            # Entrenar
            train_loss = self.train_step(X_train, y_train)
            
            # Validar
            val_predictions = self.model(X_val, training=False)
            val_loss = self.loss_fn(y_val, val_predictions)
            
            # Guardar checkpoint
            self.checkpoint_manager.save()
        
        return train_loss.numpy()

    def load_latest_checkpoint(self) -> bool:
        """Load the latest checkpoint if available."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restored from checkpoint: {self.checkpoint_manager.latest_checkpoint}")
            return True
        return False 