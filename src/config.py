# config.py — Centralized configuration: hyperparameters, file paths, constants, and training settings.

# ============================================================
# Audio Preprocessing Configuration
# ============================================================

SAMPLE_RATE = 16000            # Target sampling rate in Hz (speech standard)
TARGET_DURATION = 4.0          # Fixed duration in seconds for all audio clips
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)  # = 64000 samples
SILENCE_TOP_DB = 30            # Threshold (dB) below peak for silence trimming
NORMALIZE_AUDIO = True         # Whether to apply amplitude normalization
CROP_MODE = "center"           # Cropping strategy: "center" removes equal amounts from both ends
PAD_MODE = "constant"          # Padding strategy: "constant" zero-pads

SUPPORTED_AUDIO_EXTENSIONS = [
    ".wav",
    ".flac",
    ".mp3"
]

# ============================================================
# Mel-Spectrogram Feature Extraction Configuration
# ============================================================

N_FFT = 1024                   # FFT window size — controls frequency resolution (1024 / 16000 = 64ms window)
HOP_LENGTH = 256               # Number of samples between successive STFT frames (256 / 16000 = 16ms hop)
WIN_LENGTH = 1024              # Window length for STFT — typically equal to N_FFT
N_MELS = 128                   # Number of Mel filter banks — frequency axis dimension of the spectrogram
F_MIN = 20                     # Lowest frequency (Hz) for the Mel filter bank — below typical speech content
F_MAX = 8000                   # Highest frequency (Hz) for the Mel filter bank — Nyquist for 16kHz is 8000
POWER = 2.0                    # Exponent for the magnitude spectrogram (2.0 = power spectrogram)
TOP_DB = 80                    # Dynamic range in dB for power_to_db clipping

# Optional MFCC configuration
N_MFCC = 40                    # Number of MFCC coefficients to extract

# ============================================================
# Dataset Sources
# ============================================================

TRAIN_DATASET = "ASVspoof2019_LA"        # Source dataset for training partition
DEVELOPMENT_DATASET = "ASVspoof2019_LA"  # Source dataset for development partition
EVALUATION_DATASET = "ASVspoof2021_DF"   # Source dataset for final evaluation (isolated)

TRAIN_METADATA = "data/metadata/training_metadata.csv"
DEV_METADATA = "data/metadata/development_metadata.csv"
EVAL_METADATA = "data/metadata/dataset_manifest.csv"  # eval rows filtered by partition == "eval"

# ============================================================
# DataLoader Configuration
# ============================================================

BATCH_SIZE = 8                 # Number of samples per batch
NUM_WORKERS = 0                # Subprocesses for data loading (0 = main thread, safest on Windows/CPU)
PIN_MEMORY = True              # Faster host-to-device transfer when using GPU
SHUFFLE_TRAIN = True           # Shuffle training data each epoch
SHUFFLE_VALIDATION = False     # Keep validation order deterministic
SHUFFLE_TEST = False           # Keep test/evaluation order deterministic

# ============================================================
# Training Augmentation Configuration
# ============================================================

USE_AUGMENTATION = True        # Enable waveform augmentation during training
NOISE_PROB = 0.25              # Probability of additive Gaussian noise
NOISE_SCALE = 0.005            # Standard deviation of additive noise
GAIN_PROB = 0.25               # Probability of random gain
GAIN_RANGE = (0.8, 1.2)       # Min/max multiplier for random gain
TIME_SHIFT_PROB = 0.20         # Probability of random circular time shift
TIME_SHIFT_MAX = 1600          # Maximum shift in samples (0.1 s at 16 kHz)

# ============================================================
# Spectrogram Augmentation Configuration
# ============================================================

USE_SPEC_AUGMENTATION = False  # Disabled by default for baseline experiment
FREQ_MASK_PARAM = 15           # Maximum number of Mel bins to mask
TIME_MASK_PARAM = 25           # Maximum number of time frames to mask

# ============================================================
# Baseline CNN Configuration
# ============================================================

MODEL_NAME = "AudioDeepfakeCNN"
INPUT_CHANNELS = 1             # Grayscale-like spectrogram
NUM_CLASSES = 1                # Binary classification (BCEWithLogitsLoss)
DROPOUT = 0.4                  # Dropout rate for the fully connected layer

# ============================================================
# Training Configuration
# ============================================================

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_NORM = 1.0
USE_CLASS_WEIGHT = False
CHECKPOINT_PATH = "models/best_cnn.pth"
RANDOM_SEED = 42
PREDICTION_THRESHOLD = 0.5

# Learning Rate Scheduler
LR_FACTOR = 0.5
LR_PATIENCE = 2
LR_MIN = 1e-7
