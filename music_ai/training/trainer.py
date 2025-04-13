import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, InputLayer, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure TensorFlow to use CPU
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('CPU')
if physical_devices:
    try:
        # Set memory growth for CPU devices
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except ValueError:
        # If memory growth is not supported, just continue
        pass

class MusicTrainer:
    """Handles the training of the music generation model."""
    
    def __init__(
        self,
        input_shape: tuple,
        units: int = 32,
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
    
    def _build_model(self) -> Model:
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # Input layer
        model.add(InputLayer(shape=self.input_shape))
        
        # LSTM layers with reduced units
        for i in range(self.num_layers):
            return_sequences = True  # Always return sequences to maintain shape
            model.add(LSTM(
                self.units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
        
        # Output layer with TimeDistributed to maintain sequence length
        model.add(TimeDistributed(Dense(self.input_shape[1], activation='softmax')))
        
        # Compile model with reduced learning rate
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.2,
        checkpoint_dir: str = None
    ) -> dict:
        """Train the model on the given data."""
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'model.weights.h5')
            model_path = os.path.join(checkpoint_dir, 'model.h5')
            
            # Save complete model
            self.model.save(model_path)
            
            # Save weights
            self.model.save_weights(checkpoint_path)
            
            # Create checkpoint callback
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                verbose=1
            )
            
            # Create early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            
            callbacks = [checkpoint, early_stopping]
        else:
            callbacks = []
        
        # Train the model
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