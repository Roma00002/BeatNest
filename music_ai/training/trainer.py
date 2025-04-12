import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import os

class MusicTrainer:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        units: int = 256,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize the music trainer.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data (n_mels, sequence_length)
            units (int): Number of units in LSTM layers
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the optimizer
        """
        self.input_shape = input_shape
        self.units = units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the LSTM model architecture."""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        
        # LSTM layers
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            model.add(tf.keras.layers.LSTM(
                self.units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate
            ))
        
        # Output layer
        model.add(tf.keras.layers.Dense(self.input_shape[0], activation='sigmoid'))
        
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
        validation_split: float = 0.2,
        checkpoint_dir: str = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Target data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            checkpoint_dir (str): Directory to save checkpoints
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Create checkpoint callback
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-{epoch:02d}.weights.h5')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                save_weights_only=True,
                verbose=1
            )
            callbacks = [checkpoint_callback]
        else:
            callbacks = []
        
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        } 