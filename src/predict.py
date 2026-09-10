import torch
import numpy as np
import librosa
from src.preprocessing import preprocess_audio
from src.features import extract_mel_spectrogram
from src.model import AudioDeepfakeCNN

def get_audio_metadata(audio_path):
    # Safe metadata extraction
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)
        channels = 1 if len(y.shape) == 1 else y.shape[0]
        return {
            "duration": duration,
            "sample_rate": sr,
            "channels": channels
        }
    except Exception as e:
        raise ValueError("Could not read audio metadata.")

def load_frozen_model(device):
    model = AudioDeepfakeCNN().to(device)
    checkpoint = torch.load('models/cnn_real/best_cnn_real.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def run_inference(audio_path, model, device, threshold=0.5):
    try:
        meta = get_audio_metadata(audio_path)
        out = preprocess_audio(audio_path)
        waveform = out['waveform']
        mel = extract_mel_spectrogram(waveform)
        
        if mel.shape != (128, 251):
            raise ValueError(f"Extracted Log-Mel shape {mel.shape} is invalid for the model.")
            
        feat = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit = model(feat)
            prob = float(torch.sigmoid(logit).cpu().numpy()[0][0])
            
        spoof_prob = prob
        bonafide_prob = 1.0 - prob
        predicted_label = 1 if spoof_prob >= threshold else 0
        
        return {
            "logit": float(logit.cpu().numpy()[0][0]),
            "spoof_probability": spoof_prob,
            "bonafide_probability": bonafide_prob,
            "predicted_label": predicted_label,
            "waveform": waveform,
            "mel_spectrogram": mel,
            "metadata": meta,
            "tensor_input": feat
        }
    except Exception as e:
        raise ValueError(f"Inference failed: {str(e)}")
