#!/usr/bin/env python3
from music_ai.gui.gradio_interface import create_interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)  # share=True para crear un enlace público 