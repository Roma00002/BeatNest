import tensorflow as tf
from typing import Tuple

class MusicGenerator(tf.keras.Model):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        units: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.3
    ):
        super(MusicGenerator, self).__init__()
        
        self.input_shape = input_shape
        self.units = units
        self.num_layers = num_layers
        
        # Input layer with explicit shape
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        
        # LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                stateful=False,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate/2
            )
            self.lstm_layers.append(lstm)
            
        # Batch normalization layers
        self.batch_norms = [
            tf.keras.layers.BatchNormalization()
            for _ in range(num_layers)
        ]
        
        # Output layer to match input dimensions
        self.output_dense = tf.keras.layers.Dense(
            input_shape[0],  # Número de bandas mel
            activation='sigmoid'
        )
        
    def call(self, inputs, training=False):
        # Process through LSTM layers
        x = inputs
        
        for lstm, batch_norm in zip(self.lstm_layers, self.batch_norms):
            x = lstm(x)
            x = batch_norm(x, training=training)
        
        # Generate output
        return self.output_dense(x)
    
    def get_config(self):
        return {
            'input_shape': self.input_shape,
            'units': self.units,
            'num_layers': self.num_layers
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def generate(self, initial_sequence: tf.Tensor, sequence_length: int) -> tf.Tensor:
        """Generate a new sequence of specified length."""
        # Ensure initial sequence has correct shape
        if len(initial_sequence.shape) == 2:
            initial_sequence = tf.expand_dims(initial_sequence, 0)
        
        generated = initial_sequence
        
        # Generate one step at a time
        for _ in range(sequence_length):
            # Predict next step
            next_step = self(generated, training=False)
            
            # Take last timestep prediction
            next_step = next_step[:, :, -1:]
            
            # Concatenate with generated sequence
            generated = tf.concat([generated, next_step], axis=2)
            
        return generated 

    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build and compile the model."""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Capa LSTM bidireccional
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)
        )(inputs)
        
        # Capas densas con dropout para regularización
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Capa de salida - genera un único valor para cada banda mel
        outputs = tf.keras.layers.Dense(input_shape[0], activation='linear')(x[:, -1, :])
        outputs = tf.keras.layers.Reshape((input_shape[0], 1))(outputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar el modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        model.summary()
        return model 