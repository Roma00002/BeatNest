import sys
import os
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QFileDialog, QProgressBar, QLabel,
    QSpinBox, QHBoxLayout, QMessageBox, QListWidget,
    QTextEdit, QLineEdit, QScrollArea, QFrame,
    QDialog, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor
from ..data.preprocessor import MusicPreprocessor
from ..models.music_generator import MusicGenerator as Model
from ..training.trainer import MusicTrainer
from ..generation.generator import MusicGenerator

class GenreDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar Género Musical")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Género selector
        layout.addWidget(QLabel("Selecciona el género principal de los archivos MIDI:"))
        
        self.genre_combo = QComboBox()
        self.genre_combo.addItems([
            "Hip-Hop",
            "Electrónica",
            "Rock",
            "Jazz",
            "Pop",
            "Clásica",
            "Latina",
            "Experimental",
            "Otro"
        ])
        layout.addWidget(self.genre_combo)
        
        # Botones
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Aceptar")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancelar")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
    
    def get_genre(self):
        return self.genre_combo.currentText()

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    time_remaining = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, data_dir, model_params):
        super().__init__()
        self.data_dir = data_dir
        self.model_params = model_params
        self.start_time = None
        self.should_stop = False
        
    def stop(self):
        self.should_stop = True
        
    def run(self):
        try:
            self.start_time = time.time()
            self.should_stop = False
            
            # Initialize preprocessor
            preprocessor = MusicPreprocessor(
                sample_rate=self.model_params['sample_rate'],
                hop_length=self.model_params['hop_length'],
                n_mels=self.model_params['n_mels']
            )
            
            # Load dataset
            X, y = preprocessor.load_dataset(
                self.data_dir,
                sequence_length=self.model_params['sequence_length']
            )
            
            # Split dataset
            split_idx = int(len(X) * 0.9)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Initialize model with correct input shape
            input_shape = (self.model_params['n_mels'], X.shape[2])
            model = Model(
                input_shape=input_shape,
                units=self.model_params['units'],
                num_layers=self.model_params['num_layers']
            )
            
            # Initialize trainer
            trainer = MusicTrainer(
                model,
                learning_rate=self.model_params['learning_rate'],
                checkpoint_dir=self.model_params['checkpoint_dir']
            )
            
            # Train model
            for epoch in range(self.model_params['epochs']):
                if self.should_stop:
                    self.error.emit("Entrenamiento detenido por el usuario")
                    return
                    
                trainer.train(
                    train_data=(X_train, y_train),
                    validation_data=(X_val, y_val),
                    batch_size=self.model_params['batch_size'],
                    epochs=1
                )
                
                # Calculate progress and time remaining
                progress = int((epoch + 1) / self.model_params['epochs'] * 100)
                self.progress.emit(progress)
                
                # Calculate time remaining
                elapsed_time = time.time() - self.start_time
                time_per_epoch = elapsed_time / (epoch + 1)
                remaining_epochs = self.model_params['epochs'] - (epoch + 1)
                remaining_time = remaining_epochs * time_per_epoch
                
                # Format time remaining
                hours = int(remaining_time // 3600)
                minutes = int((remaining_time % 3600) // 60)
                seconds = int(remaining_time % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                self.time_remaining.emit(time_str)
            
            if not self.should_stop:
                self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

class GenerationThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, model_params, seed_path=None):
        super().__init__()
        self.model_params = model_params
        self.seed_path = seed_path
        
    def run(self):
        try:
            # Initialize preprocessor and model
            preprocessor = MusicPreprocessor(
                sample_rate=self.model_params['sample_rate'],
                hop_length=self.model_params['hop_length'],
                n_mels=self.model_params['n_mels']
            )
            
            model = Model(
                input_shape=(self.model_params['n_mels'], self.model_params['sequence_length']),
                units=self.model_params['units'],
                num_layers=self.model_params['num_layers']
            )
            
            # Initialize generator
            generator = MusicGenerator(
                model,
                preprocessor,
                temperature=self.model_params['temperature']
            )
            
            # Generate beat
            output_path = os.path.join("generated", f"beat_{len(os.listdir('generated'))}.mid")
            generator.generate_beat(
                seed_path=self.seed_path,
                sequence_length=self.model_params['sequence_length'],
                output_path=output_path
            )
            
            self.finished.emit(output_path)
            
        except Exception as e:
            self.error.emit(str(e))

class ChatWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMinimumHeight(150)
        layout.addWidget(self.chat_history)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Describe cómo quieres que sea tu beat...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("Enviar")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Initial message
        self.add_message("AI", "¡Hola! Describe cómo quieres que sea tu beat. Por ejemplo:\n"
                        "- Género musical (hip-hop, electrónico, etc.)\n"
                        "- Tempo (lento, medio, rápido)\n"
                        "- Estilo (minimalista, complejo, etc.)\n"
                        "- Referencias ('algo como [artista/canción]')")
    
    def add_message(self, sender: str, message: str):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_history.setTextCursor(cursor)
        
        if sender == "User":
            self.chat_history.insertHtml(f"<p style='color: #2196F3'><b>Tú:</b> {message}</p>")
        else:
            self.chat_history.insertHtml(f"<p style='color: #4CAF50'><b>AI:</b> {message}</p>")
        
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum()
        )
    
    def send_message(self):
        message = self.input_field.text().strip()
        if message:
            self.add_message("User", message)
            self.input_field.clear()
            return message
        return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Music Beat Generator")
        self.setMinimumSize(800, 600)
        
        # Initialize model parameters
        self.model_params = {
            'sample_rate': 44100,
            'hop_length': 512,
            'n_mels': 128,
            'sequence_length': 64,
            'units': 256,
            'num_layers': 3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'checkpoint_dir': 'checkpoints',
            'temperature': 1.0,
            'genre': None
        }
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create split layout
        split_layout = QHBoxLayout()
        
        # Left panel (original controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.create_control_elements(left_layout)
        split_layout.addWidget(left_panel)
        
        # Right panel (chat interface)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.create_chat_interface(right_layout)
        split_layout.addWidget(right_panel)
        
        main_layout.addLayout(split_layout)
        
        # Create generated directory if it doesn't exist
        os.makedirs('generated', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
    
    def create_chat_interface(self, layout):
        chat_label = QLabel("Chat con la IA")
        chat_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(chat_label)
        
        self.chat_widget = ChatWidget()
        layout.addWidget(self.chat_widget)
    
    def create_control_elements(self, layout):
        # Data loading section
        data_label = QLabel("1. Load Training Data")
        data_label.setStyleSheet("font-weight: bold")
        layout.addWidget(data_label)
        
        self.load_data_btn = QPushButton("Select MIDI Files Directory")
        self.load_data_btn.clicked.connect(self.load_data)
        layout.addWidget(self.load_data_btn)
        
        # Training parameters section
        params_label = QLabel("2. Training Parameters")
        params_label.setStyleSheet("font-weight: bold")
        layout.addWidget(params_label)
        
        params_layout = QHBoxLayout()
        
        # Epochs
        epochs_layout = QVBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.model_params['epochs'])
        epochs_layout.addWidget(self.epochs_spin)
        params_layout.addLayout(epochs_layout)
        
        # Batch size
        batch_layout = QVBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(self.model_params['batch_size'])
        batch_layout.addWidget(self.batch_spin)
        params_layout.addLayout(batch_layout)
        
        layout.addLayout(params_layout)
        
        # Training section
        train_label = QLabel("3. Train Model")
        train_label.setStyleSheet("font-weight: bold")
        layout.addWidget(train_label)
        
        # Training buttons layout
        train_buttons = QHBoxLayout()
        
        # Start training button
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.train_btn.setMinimumWidth(120)
        train_buttons.addWidget(self.train_btn)
        
        # Stop training button
        self.stop_train_btn = QPushButton("Stop Training")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.setMinimumWidth(120)
        self.stop_train_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff5252;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ff1744;
            }
            QPushButton:disabled {
                background-color: #ffcdd2;
            }
        """)
        train_buttons.addWidget(self.stop_train_btn)
        
        layout.addLayout(train_buttons)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Time remaining label
        self.time_label = QLabel("Tiempo restante: --:--:--")
        layout.addWidget(self.time_label)
        
        # Generation section
        gen_label = QLabel("4. Generate Beats")
        gen_label.setStyleSheet("font-weight: bold")
        layout.addWidget(gen_label)
        
        self.generate_btn = QPushButton("Generate New Beat")
        self.generate_btn.clicked.connect(self.generate_beat)
        self.generate_btn.setEnabled(False)
        layout.addWidget(self.generate_btn)
        
        # Generated files list
        self.files_list = QListWidget()
        self.files_list.itemDoubleClicked.connect(self.open_file)
        layout.addWidget(self.files_list)
        
        self.update_files_list()
        
    def load_data(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory with MIDI Files"
        )
        if dir_path:
            self.model_params['data_dir'] = dir_path
            self.train_btn.setEnabled(True)
            QMessageBox.information(
                self,
                "Success",
                f"Selected directory: {dir_path}\nReady to start training!"
            )
    
    def start_training(self):
        # Show genre selection dialog
        genre_dialog = GenreDialog(self)
        if genre_dialog.exec() == QDialog.DialogCode.Accepted:
            self.model_params['genre'] = genre_dialog.get_genre()
            
            # Update parameters from UI
            self.model_params['epochs'] = self.epochs_spin.value()
            self.model_params['batch_size'] = self.batch_spin.value()
            
            # Create and start training thread
            self.training_thread = TrainingThread(
                self.model_params['data_dir'],
                self.model_params
            )
            self.training_thread.progress.connect(self.update_progress)
            self.training_thread.time_remaining.connect(self.update_time)
            self.training_thread.finished.connect(self.training_finished)
            self.training_thread.error.connect(self.show_error)
            
            self.training_thread.start()
            self.train_btn.setEnabled(False)
            self.load_data_btn.setEnabled(False)
            
            # Update chat with genre information
            self.chat_widget.add_message(
                "AI",
                f"Comenzando entrenamiento con archivos de {self.model_params['genre']}..."
            )
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_time(self, time_str):
        self.time_label.setText(f"Tiempo restante: {time_str}")
    
    def training_finished(self):
        self.train_btn.setEnabled(True)
        self.load_data_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.time_label.setText("Tiempo restante: Completado")
        
        genre = self.model_params['genre']
        self.chat_widget.add_message(
            "AI",
            f"¡Entrenamiento completado exitosamente!\n"
            f"Ahora puedo generar beats de {genre}. "
            "Describe cómo quieres que sea tu beat."
        )
        
    def generate_beat(self):
        # Get user description from chat
        description = self.chat_widget.input_field.text().strip()
        if not description:
            self.chat_widget.add_message("AI", "Por favor, describe cómo quieres que sea el beat antes de generarlo.")
            return
            
        # Adjust generation parameters based on description
        self.adjust_parameters_from_description(description)
        
        # Start generation
        self.generation_thread = GenerationThread(self.model_params)
        self.generation_thread.finished.connect(self.generation_finished)
        self.generation_thread.error.connect(self.show_error)
        self.generation_thread.start()
        self.generate_btn.setEnabled(False)
        
        # Add confirmation message to chat
        self.chat_widget.add_message("AI", "Generando beat según tu descripción... Por favor espera.")
    
    def adjust_parameters_from_description(self, description):
        """Ajusta los parámetros de generación según la descripción del usuario."""
        description = description.lower()
        
        # Ajustar temperatura según la creatividad deseada
        if any(word in description for word in ["experimental", "creativo", "único", "original"]):
            self.model_params['temperature'] = 1.2
        elif any(word in description for word in ["conservador", "simple", "básico"]):
            self.model_params['temperature'] = 0.8
            
        # Ajustar sequence_length según el tempo
        if any(word in description for word in ["rápido", "energético", "veloz"]):
            self.model_params['sequence_length'] = 128
        elif any(word in description for word in ["lento", "suave", "tranquilo"]):
            self.model_params['sequence_length'] = 256
            
        # Considerar el género en la generación
        if self.model_params['genre']:
            self.chat_widget.add_message(
                "AI",
                f"Generando un beat de {self.model_params['genre']} según tu descripción..."
            )
    
    def generation_finished(self, output_path):
        self.generate_btn.setEnabled(True)
        self.update_files_list()
        self.chat_widget.add_message(
            "AI",
            f"¡Beat generado con éxito! Puedes encontrarlo en la lista de archivos generados como: {os.path.basename(output_path)}\n"
            "¿Quieres generar otro beat? Describe cómo lo quieres."
        )
        
    def update_files_list(self):
        self.files_list.clear()
        if os.path.exists('generated'):
            for file in sorted(os.listdir('generated')):
                if file.endswith('.mid'):
                    self.files_list.addItem(file)
                    
    def open_file(self, item):
        file_path = os.path.join('generated', item.text())
        if sys.platform == 'linux':
            os.system(f'xdg-open "{file_path}"')
            
    def show_error(self, message):
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred: {message}"
        )

    def stop_training(self):
        """Detiene el proceso de entrenamiento de forma segura."""
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirmar Detención",
                "¿Estás seguro de que quieres detener el entrenamiento?\nSe guardará el progreso hasta este punto.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.chat_widget.add_message(
                    "AI",
                    "Deteniendo el entrenamiento... Por favor espera."
                )
                self.training_thread.stop()
                self.stop_train_btn.setEnabled(False)
                self.train_btn.setEnabled(True)
                self.load_data_btn.setEnabled(True)
                self.time_label.setText("Tiempo restante: Detenido")
                self.chat_widget.add_message(
                    "AI",
                    "Entrenamiento detenido. Puedes iniciar un nuevo entrenamiento cuando quieras."
                ) 