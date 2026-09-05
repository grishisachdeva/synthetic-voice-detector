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
**Step 2** — Python environment configured.
**Step 3** — ASVspoof 2021 DF evaluation dataset inspected.
**Step 4** — Training/development data prepared.
**Step 5** — Audio preprocessing pipeline verified.
**Step 6** — Mel-spectrogram feature extraction verified.
**Step 7** — PyTorch Dataset and DataLoaders verified.
**Step 8** — Baseline CNN Architecture verified.
**Step 9** — CNN Training Pipeline verified.
**Step 10** — CNN Evaluation Pipeline verified.

## Feature Extraction

### What is a Spectrogram?
A spectrogram is a visual representation of the frequency content of a signal over time. It transforms a one-dimensional audio waveform into a two-dimensional image where the x-axis represents time, the y-axis represents frequency, and pixel intensity represents energy.

### What is the STFT?
The Short-Time Fourier Transform (STFT) divides the audio signal into short overlapping windows and computes the Fourier transform of each window, revealing how the frequency content evolves over time. Key parameters:
- **n_fft** (1024): The FFT window size controlling frequency resolution. A 1024-point FFT at 16 kHz gives a 64 ms analysis window.
- **hop_length** (256): The number of samples between successive STFT frames. 256 samples at 16 kHz gives a 16 ms hop, producing overlapping frames.
- **win_length** (1024): The actual window length applied before FFT, typically equal to n_fft.

### What is the Mel Scale?
The Mel scale is a perceptual scale of pitch that approximates how humans perceive differences in frequency. Lower frequencies are spaced more finely than higher frequencies, matching the non-linear frequency sensitivity of the human ear. A Mel filter bank maps the linear-frequency STFT output onto the Mel scale.

### Why Mel Spectrograms for Speech?
Mel spectrograms compress the frequency axis in a way that emphasizes the frequency ranges most relevant to human speech perception. This makes them highly effective as input features for speech-related deep learning tasks, including deepfake detection, because they capture timbral and prosodic characteristics while discarding less perceptually relevant detail.

### Why Convert to Decibels?
Human perception of loudness is approximately logarithmic. Converting the power spectrogram to decibels compresses the dynamic range, making quiet and loud components more equally visible to both human viewers and neural networks.

### Why Normalize?
Per-sample normalization (zero mean, unit variance) ensures that the feature values are on a consistent scale across different audio recordings, which helps the neural network train more stably and converge faster.

### Parameters Used

| Parameter    | Value   | Description                              |
|-------------|---------|------------------------------------------|
| Sample rate | 16000 Hz| Standard speech sampling rate            |
| N_FFT       | 1024    | FFT window size                          |
| Hop length  | 256     | Samples between STFT frames              |
| Win length  | 1024    | Window length for STFT                   |
| Mel bands   | 128     | Number of Mel filter banks               |
| F_min       | 20 Hz   | Lowest Mel filter frequency              |
| F_max       | 8000 Hz | Highest Mel filter frequency (Nyquist)   |
| Power       | 2.0     | Power spectrogram exponent               |
| Top dB      | 80      | Dynamic range clipping for power_to_db   |

## PyTorch Dataset Pipeline

### Why a Custom Dataset?
A custom `torch.utils.data.Dataset` is needed because our pipeline involves multiple transformations (loading, preprocessing, feature extraction) that must happen consistently for every sample. The built-in PyTorch datasets do not support audio deepfake detection workflows out of the box.

### On-Demand Feature Extraction
Features are extracted inside `__getitem__` rather than precomputed, keeping RAM usage low regardless of dataset size. This is essential because the full ASVspoof dataset contains hundreds of thousands of files.

### Expected Tensor Shape
Each sample produces a feature tensor of shape `[1, 128, 251]`:
- **1** — single channel (grayscale-like, for CNN compatibility)
- **128** — Mel frequency bands
- **251** — time frames (from 64000 samples with hop_length=256)

Batched shape: `[batch_size, 1, 128, 251]`

### Label Mapping
- **BONAFIDE** (genuine human speech) = `0`
- **SPOOF** (synthetic/manipulated speech) = `1`

Labels are stored as `torch.float32` for compatibility with `BCEWithLogitsLoss`.

### Training Augmentation
Mild waveform augmentations are applied **before** feature extraction during training only:
- Additive Gaussian noise (p=0.25, scale=0.005)
- Random gain (p=0.25, range 0.8-1.2x)
- Random circular time shift (p=0.20, max 0.1s)

Optional SpecAugment-style masking is available but disabled for the baseline experiment.

### Validation/Evaluation Determinism
Development and evaluation datasets produce identical outputs for the same index across calls, ensuring reproducible metrics. Training augmentation makes training samples intentionally stochastic.

### Evaluation Set Isolation
The ASVspoof 2021 DF evaluation set is never mixed with training or development data. It is loaded from a separate metadata partition and used only for final model benchmarking.

### Speaker Overlap Check
A diagnostic function verifies that no speakers appear in both training and development sets (speaker-disjoint), preventing the model from memorizing voice characteristics instead of learning deepfake detection.

### DataLoader Batching
- Training: shuffled each epoch for better generalization
- Development/Evaluation: deterministic order for reproducible metrics
- `num_workers=0` for Windows/CPU compatibility
- `pin_memory=True` (automatically ignored if no GPU is available)

## Baseline CNN Architecture

The baseline model is a standard Convolutional Neural Network designed to extract local time-frequency patterns from the log-Mel spectrogram.

### Input
- **Shape**: `[batch_size, 1, 128, 251]` (1 channel, 128 Mel bands, 251 time frames for a 4.0s clip).

### Convolutional Blocks
The model consists of four convolutional blocks, progressively increasing the channel depth while reducing the spatial dimensions via max pooling.

1. **Block 1**: Conv2d(1 → 32) → BatchNorm2d → ReLU → MaxPool2d(2)
2. **Block 2**: Conv2d(32 → 64) → BatchNorm2d → ReLU → MaxPool2d(2)
3. **Block 3**: Conv2d(64 → 128) → BatchNorm2d → ReLU → MaxPool2d(2)
4. **Block 4**: Conv2d(128 → 256) → BatchNorm2d → ReLU (No MaxPool here)

**Key Components:**
- **Convolution**: Extracts local translation-invariant features (e.g., edges, formants) using a 3x3 kernel.
- **Batch Normalization**: Normalizes activations to stabilize and accelerate training.
- **ReLU**: Introduces non-linearity to learn complex patterns.
- **Max Pooling**: Reduces spatial dimensions, promoting invariance to small shifts in time or frequency.

### Global Pooling and Classification
- **Adaptive Global Average Pooling**: Reduces the remaining spatial dimensions (H, W) to exactly 1x1 per channel. This crucial step decouples the fully-connected layers from the input time dimension, allowing the model to accept variable-length audio clips without architectural changes or magic numbers.
- **Flatten**: Reshapes the pooled features from `[B, 256, 1, 1]` to `[B, 256]`.
- **Fully Connected (Linear)**: A hidden layer (256 → 128) with ReLU.
- **Dropout**: Randomly zeroes out 40% of the activations to prevent overfitting.
- **Output (Linear)**: The final layer (128 → 1) produces a single, unnormalized scalar logit.

### Why No Sigmoid?
We explicitly omit a Sigmoid activation at the end of the network. Instead, the model outputs raw logits which are passed directly to `torch.nn.BCEWithLogitsLoss()`. This PyTorch loss function mathematically combines the Sigmoid and Binary Cross Entropy steps into a single operation, which is significantly more numerically stable and avoids issues like vanishing gradients or precision loss near 0 and 1.

### Architecture Diagram
```
Input: [B, 1, 128, 251]
       ↓
[Conv Block 1]  (32 channels, pool/2)
       ↓
[Conv Block 2]  (64 channels, pool/2)
       ↓
[Conv Block 3]  (128 channels, pool/2)
       ↓
[Conv Block 4]  (256 channels, no pool)
       ↓
[Adaptive Avg Pool]  (1x1 spatial)
       ↓
    [Flatten]   (256 features)
       ↓
 [Linear 128] + ReLU + Dropout(0.4)
       ↓
  [Linear 1]    (Raw Logit)
```

## Training Pipeline

The training pipeline orchestrates data loading, model optimization, and performance tracking.

### Data Loading
The training dataloader feeds batches of audio features (extracted on-the-fly) to the network. The mock dataset we are currently using (6 training / 6 development files) is intentionally tiny; it exists **solely to verify the pipeline technically** and does not yield real-world deepfake detection metrics.

### Forward Pass & Loss
1. **CNN Forward Pass:** The model processes the batch and outputs raw binary classification logits.
2. **Loss Computation:** `torch.nn.BCEWithLogitsLoss` computes the binary cross entropy loss. This function combines a sigmoid activation and BCE into a single, numerically stable operation. 

### Optimization & Backpropagation
3. **Backpropagation:** The loss is backpropagated to compute gradients.
4. **Gradient Clipping:** Gradients are clipped to a maximum norm to prevent exploding gradients.
5. **Optimizer:** We use `torch.optim.AdamW`, which provides Adam optimization with decoupled weight decay for better regularization.

### Learning Rate Scheduling & Early Stopping
6. **Learning-Rate Scheduler:** `ReduceLROnPlateau` monitors validation loss and reduces the learning rate when improvement stalls.
7. **Early Stopping:** If the validation metric stops improving for a set patience (e.g., 5 epochs), training halts early to prevent overfitting.

### Validation & Checkpointing
8. **Validation:** After each epoch, the model is evaluated on the development set without calculating gradients.
9. **Checkpointing:** The model yielding the best validation ROC-AUC (or F1/Loss if ROC-AUC is undefined) is saved to disk as a checkpoint. Checkpoints include the model state, optimizer state, scheduler state, and configuration to allow seamless resuming.

## Evaluation

The final evaluation pipeline provides a rigorous assessment of the deepfake detection model on a held-out dataset. 

### Core Principles
- **Held-out Evaluation**: The model is evaluated on the ASVspoof 2021 DF set, which is kept entirely separate from the training and development sets.
- **Zero Leakage**: Strict code-level checks ensure that evaluation samples are never used for training, threshold tuning, or calculating preprocessing statistics. This ensures an unbiased estimate of generalization.

### Metrics Computed
- **Accuracy**: The overall percentage of correctly classified audio clips.
- **Precision**: The proportion of predicted SPOOF samples that were actually SPOOF (minimizes false alarms).
- **Recall**: The proportion of actual SPOOF samples correctly identified (minimizes missed deepfakes).
- **F1 Score**: The harmonic mean of Precision and Recall, providing a single robust metric for imbalanced data.
- **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to distinguish between classes across all possible thresholds.
- **EER (Equal Error Rate)**: The point on the ROC curve where the False Positive Rate equals the False Negative Rate. A lower EER indicates a better performing biometric/detection system.
- **Confusion Matrix**: A visualization of the absolute counts of True Positives, True Negatives, False Positives, and False Negatives at the default threshold of 0.5.

### Current Limitation: Pipeline Validation Only
**IMPORTANT**: The dataset currently used in this repository is a tiny 5-file mock representation of the ASVspoof 2021 DF evaluation set. Because of this small sample size, the generated accuracy, ROC-AUC, and EER values are **not representative** of the actual deepfake detection performance of this system. They exist solely to verify that the evaluation code, metric calculations, and visualization pipelines function correctly.
