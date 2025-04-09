import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict
from tqdm import tqdm
import os

class MusicTrainer:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        units: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        """
        Initialize the MusicTrainer.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data (n_mels, sequence_length)
            units (int): Number of units in LSTM layers
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.units = units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Build model
        self.model = self._build_model()
        
        # Initialize checkpoint manager
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"✓ Checkpoint directory: {os.path.abspath(self.checkpoint_dir)}")
        
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=3
        )
        
        # Restore latest checkpoint if available
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"✓ Restored checkpoint from {latest_checkpoint}")
    
    def _build_model(self) -> tf.keras.Model:
        """Build the LSTM model."""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        
        # LSTM layers
        for _ in range(self.num_layers - 1):
            model.add(tf.keras.layers.LSTM(
                self.units,
                return_sequences=True,
                dropout=self.dropout_rate
            ))
        
        # Final LSTM layer
        model.add(tf.keras.layers.LSTM(
            self.units,
            dropout=self.dropout_rate
        ))
        
        # Output layer
        model.add(tf.keras.layers.Dense(self.input_shape[0]))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            X (np.ndarray): Input sequences
            y (np.ndarray): Target sequences
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            Dict[str, list]: Training history
        """
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'model_{epoch:02d}.h5'),
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save(os.path.join(self.checkpoint_dir, 'model_final.h5'))
        
        return {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    
    def load_latest_checkpoint(self) -> bool:
        """
        Load the latest checkpoint.
        
        Returns:
            bool: True if checkpoint was loaded, False otherwise
        """
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"Loaded checkpoint from {latest_checkpoint}")
            return True
        return False 