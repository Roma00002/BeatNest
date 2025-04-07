from setuptools import setup, find_packages

setup(
    name="music-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "numpy>=1.24.3",
        "librosa>=0.10.1",
        "pretty_midi>=0.2.10",
        "midiutil>=1.2.1",
        "gradio>=4.19.2",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "python-dotenv>=0.19.0",
        "soundfile>=0.12.1"
    ],
    python_requires=">=3.8",
) 