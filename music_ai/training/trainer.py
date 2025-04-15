import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, InputLayer, TimeDistributed, Input
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
        input_shape: tuple = None,
        units: int = 32,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        model = None
    ):
        """Initialize the trainer with model parameters.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of units in LSTM layers
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model: An existing model to use instead of creating a new one
        """
        self.units = units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        if model is not None:
            # Use the provided model
            self.model = model
            self.input_shape = model.input_shape[1:]  # Extract input shape from model
            
            # Recompile the model to reset optimizer state
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            # Create a new model
            if input_shape is None:
                raise ValueError("input_shape must be provided when not loading an existing model")
            self.input_shape = input_shape
            self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """Build the LSTM model architecture."""
        model = Sequential([
            # Input layer
            Input(shape=self.input_shape),
            
            # First LSTM layer with increased units
            LSTM(self.units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate/2),
            
            # BatchNormalization can help with faster and more stable learning
            tf.keras.layers.BatchNormalization(),
            
            # Second LSTM layer with increased units
            LSTM(self.units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate/2),
            
            # Another BatchNormalization layer
            tf.keras.layers.BatchNormalization(),
            
            # Third LSTM layer (if num_layers >= 3)
            LSTM(self.units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate/2) if self.num_layers >= 3 else tf.keras.layers.Lambda(lambda x: x),
            
            # Final BatchNormalization
            tf.keras.layers.BatchNormalization(),
            
            # Output layer - use input_shape[1] for the output dimension to match the target shape
            TimeDistributed(Dense(self.input_shape[1], activation='softmax'))
        ])
        
        # Compile model with adjusted learning rate
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
        callbacks = []
        
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'model.weights.h5')
            model_path = os.path.join(checkpoint_dir, 'model.h5')
            
            # Create checkpoint callback with improved performance monitoring
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                verbose=1
            )
            
            # Create early stopping callback with adjusted patience
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,  # Increased patience for a better chance at finding global minima
                restore_best_weights=True,
                verbose=1
            )
            
            # Add learning rate reduction on plateau
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Reduce learning rate by half
                patience=3,   # Wait 3 epochs before reducing
                min_lr=0.00001,
                verbose=1
            )
            
            # Add a TensorBoard callback
            tensor_board = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(checkpoint_dir, 'logs'),
                histogram_freq=1,
                write_graph=True
            )
            
            callbacks = [checkpoint, early_stopping, reduce_lr, tensor_board]
        
        # Reshape inputs if necessary to avoid dimension mismatch
        if len(X.shape) == 3 and X.shape[1] == self.input_shape[0] and X.shape[2] == self.input_shape[1]:
            print("✓ Dimensiones de entrada correctas")
        else:
            print(f"⚠️ Ajustando dimensiones: {X.shape} a forma esperada: (n_samples, {self.input_shape[0]}, {self.input_shape[1]})")
            try:
                X = X.reshape(-1, self.input_shape[0], self.input_shape[1])
                y = y.reshape(-1, self.input_shape[0], self.input_shape[1])
            except ValueError as e:
                print(f"❌ Error al ajustar dimensiones: {e}")
        
        # Apply class weights for better balance if needed
        class_weights = None
        if validation_split > 0:
            print("✓ Usando validación durante el entrenamiento")
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights
        )
        
        return {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_accuracy': history.history.get('accuracy', []),
            'val_accuracy': history.history.get('val_accuracy', [])
        } 