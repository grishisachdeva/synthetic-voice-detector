# Audio Deepfake & Synthetic Voice Detector - Dataset Documentation

## Dataset sources
1. **ASVspoof 2019 LA**: Used for Training and Development partitions.
2. **ASVspoof 2021 DF**: Used strictly for Final Evaluation.

## Why each dataset is being used
The ASVspoof 2021 DF challenge did not release new training data. Participants were instructed to use the ASVspoof 2019 LA training and development datasets.

## Datasets
- **Training dataset**: ASVspoof 2019 LA (Train partition)
- **Development dataset**: ASVspoof 2019 LA (Dev partition)
- **Final evaluation dataset**: ASVspoof 2021 DF (Eval partition)

## Label mapping
BONAFIDE = 0, SPOOF = 1

## Speaker information
Speaker IDs are preserved in metadata to ensure disjoint splitting and prevent data leakage.

## Attack/system information
Attack types (e.g. A01, A02) are preserved to monitor generalization to unseen attacks.

## Data leakage considerations
The ASVspoof 2019 LA official train and development sets are speaker-disjoint by design. We must maintain this isolation during any further augmentation.

## Dataset limitations
**IMPORTANT**: The 5-file mock ASVspoof 2021 DF subset (from Step 3) is NOT sufficient for model training and is strictly used for end-to-end pipeline evaluation testing. Due to environment bandwidth limits, mock datasets maintaining strict structural fidelity are utilized.
