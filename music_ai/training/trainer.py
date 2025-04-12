import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import os

class MusicTrainer:
    """Handles the training of the music generation model."""
    
    def __init__(
        self,
        input_shape: tuple,
        units: int = 64,  # Reduced from 128
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """Initialize the trainer with model parameters.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of units in LSTM layers
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.units = units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the LSTM model architecture."""
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
        model.add(tf.keras.layers.Dense(self.input_shape[1], activation='softmax'))
        
        # Compile model with reduced learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,  # Reduced from 50
        batch_size: int = 8,  # Reduced from 16
        validation_split: float = 0.2,
        checkpoint_dir: str = None
    ) -> dict:
        """Train the model on the provided data.
        
        Args:
            X: Input data
            y: Target data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dictionary containing training history
        """
        # Create checkpoint directory if specified
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.weights.h5')
            
            # Callback to save best model
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            
            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,  # Reduced from 5
                restore_best_weights=True,
                verbose=1
            )
            
            callbacks = [checkpoint_callback, early_stopping]
        else:
            callbacks = []
        
        # Train model with reduced batch size
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model weights if checkpoint directory provided
        if checkpoint_dir:
            final_model_path = os.path.join(checkpoint_dir, 'model.weights.h5')
            self.model.save_weights(final_model_path)
        
        return {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        } 