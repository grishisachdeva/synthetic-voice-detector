# Audio Deepfake & Synthetic Voice Detector

A machine-learning-based cybersecurity application that analyzes audio files and classifies them as either **BONAFIDE** (genuine human speech) or **SPOOF** (AI-generated, cloned, or synthetically manipulated speech). Built as a B.Tech Artificial Intelligence and Data Science final-year minor project, this tool leverages deep learning on Mel-spectrogram representations to detect synthetic voices with high accuracy, providing confidence scores and visual explanations for each prediction.

## Technology Stack

- **Python** — Core programming language
- **PyTorch** — Deep learning framework
- **Librosa** — Audio analysis and feature extraction
- **NumPy** — Numerical computation
- **Pandas** — Data manipulation and analysis
- **Scikit-learn** — Classical ML utilities and metrics
- **Matplotlib** — Data visualization
- **Streamlit** — Interactive web application
- **SoundFile** — Audio file I/O
- **Torchvision** — Vision utilities for spectrogram processing

## Planned Pipeline

```
Audio Input
  → Audio Preprocessing
    → Mel-Spectrogram Extraction
      → Deep Learning Model
        → Real / Synthetic Classification
          → Confidence Score
            → Visualization
```

## Project Status

**Step 1** — Project structure created.
