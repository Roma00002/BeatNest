import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import os

class MusicTrainer:
    def __init__(self, input_shape: tuple, units: int = 128, num_layers: int = 2, 
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize the MusicTrainer.
        
        Args:
            input_shape (tuple): Shape of input data (n_mels, sequence_length)
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
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the LSTM model."""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        
        # LSTM layers with reduced units
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            model.add(tf.keras.layers.LSTM(
                self.units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
        
        # Output layer
        model.add(tf.keras.layers.Dense(self.input_shape[0], activation='sigmoid'))
        
        # Compile model with reduced learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 16,
             validation_split: float = 0.2, checkpoint_dir: str = None) -> dict:
        """
        Train the model on the given data.
        
        Args:
            X: Input data
            y: Target data
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            dict: Training history
        """
        # Create checkpoint callback if directory is provided
        callbacks = []
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.weights.h5')
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ))
        
        # Add early stopping
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ))
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        if checkpoint_dir:
            self.model.save_weights(os.path.join(checkpoint_dir, 'model.weights.h5'))
        
        return {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        } 