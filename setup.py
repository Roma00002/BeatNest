from setuptools import setup, find_packages

setup(
    name="music-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "tensorflow>=2.4.0",
        "librosa>=0.8.0",
        "soundfile>=0.10.3",
        "PyQt5>=5.15.2",
        "matplotlib>=3.3.2",
        "scipy>=1.5.2"
    ],
    python_requires=">=3.7",
) 