import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import json
from src.predict import load_frozen_model, run_inference
from src.explainability import generate_gradcam

st.set_page_config(page_title="Audio Deepfake Detector", layout="wide")

# Cached Model Loading
@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_frozen_model(device)
    return model, device

model, device = get_model()

# Header
st.title("Audio Deepfake & Synthetic Voice Detector")
st.subheader("AI-powered analysis of speech authenticity using deep-learning-based audio classification")
st.markdown("**Disclaimer: Model output is probabilistic and should not be treated as forensic certainty.**")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Analyze Audio", "Model Performance", "About"])

with tab1:
    st.header("Analyze Audio")
    
    uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, FLAC)", type=['wav', 'mp3', 'flac'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Analyze Audio"):
            with st.spinner("Analyzing..."):
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Run Inference
                    results = run_inference(tmp_path, model, device, threshold=0.5)
                    os.unlink(tmp_path)
                    
                    # Extract Data
                    meta = results['metadata']
                    waveform = results['waveform']
                    mel = results['mel_spectrogram']
                    spoof_prob = results['spoof_probability']
                    bonafide_prob = results['bonafide_probability']
                    pred_label = results['predicted_label']
                    tensor_input = results['tensor_input']
                    
                    # Display Metadata
                    st.markdown(f"**Filename:** {uploaded_file.name} | **Duration:** {meta['duration']:.2f}s | **Sample Rate:** {meta['sample_rate']}Hz | **Channels:** {meta['channels']}")
                    
                    # Prediction Result Card
                    st.markdown("---")
                    if pred_label == 1:
                        st.error(f"### Prediction: Possible Synthetic / Spoofed Speech")
                    else:
                        st.success(f"### Prediction: Bona Fide / Human Speech")
                        
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Synthetic/Spoof Probability", f"{spoof_prob*100:.2f}%")
                    col2.metric("Bona Fide Probability", f"{bonafide_prob*100:.2f}%")
                    col3.metric("Decision Threshold", "50.00%")
                    
                    st.markdown("---")
                    
                    # Visualizations
                    st.subheader("Audio Visualizations")
                    colA, colB = st.columns(2)
                    
                    with colA:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        time_axis = np.linspace(0, meta['duration'], num=len(waveform))
                        ax.plot(time_axis, waveform, color='blue', alpha=0.7)
                        ax.set_title("Normalized Preprocessed Waveform")
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Amplitude")
                        st.pyplot(fig)
                        
                    with colB:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        im = ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
                        ax.set_title("Log-Mel Spectrogram Used for Model Input")
                        ax.set_xlabel("Time Frames")
                        ax.set_ylabel("Mel Frequency Bins")
                        fig.colorbar(im, ax=ax)
                        st.pyplot(fig)
                        
                    # Grad-CAM
                    st.markdown("---")
                    st.subheader("Grad-CAM Model Attention Visualization")
                    st.info("This visualization shows regions of the input that influenced the model prediction. It is not forensic ground truth and should not be interpreted as proof of synthetic or human origin.")
                    
                    # Run Grad-CAM
                    tensor_input.requires_grad = True
                    cam = generate_gradcam(model, tensor_input, target_class=pred_label)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(mel, aspect='auto', origin='lower', cmap='gray')
                    im = ax.imshow(cam, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
                    ax.set_title("Grad-CAM Overlay")
                    fig.colorbar(im, ax=ax)
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    st.markdown("### Model Information")
                    st.markdown("- **Architecture**: AudioDeepfakeCNN\n- **Input**: 128 × 251 Log-Mel spectrogram\n- **Sample rate**: 16 kHz\n- **Analysis duration**: 4 seconds\n- **Decision threshold**: 0.5\n- **Framework**: PyTorch")

                except Exception as e:
                    st.error("The uploaded audio could not be processed. Please try another WAV, MP3, or FLAC file.")
                    # st.error(str(e)) # Do not expose raw traces
                    
with tab2:
    st.header("Model Performance")
    st.markdown("These results reflect the completely frozen, final evaluation performed purely on zero-shot unseen data from the ASVspoof 2021 DF dataset.")
    
    st.markdown("### Dataset Details")
    st.markdown("- **Total ASVspoof 2021 DF evaluation files**: 60,176\n- **Successfully decoded/scored files**: 34,481\n- **Excluded undecodable files**: 25,695")
    st.info("Note: The final metrics below are based strictly on the 34,481 successfully decoded files. Files were excluded due to standard audio codec corruption in the distributed dataset, not model failures.")
    
    metrics_path = 'outputs/metrics/final_results.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            res = json.load(f)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{res.get('accuracy', 0)*100:.2f}%")
        col2.metric("Precision", f"{res.get('precision', 0)*100:.2f}%")
        col3.metric("Recall", f"{res.get('recall', 0)*100:.2f}%")
        col4.metric("F1 Score", f"{res.get('f1', 0)*100:.2f}%")
        
    st.markdown("### Interpretation")
    st.warning("The model exhibits extremely high precision but substantially lower recall. This highlights a classic domain-shift challenge: the model is highly confident when classifying a sample as fake, but fails to generalize to many novel deepfake attacks in the 2021 DF dataset.")
    
    colA, colB = st.columns(2)
    with colA:
        if os.path.exists('outputs/plots/final_evaluation/roc_curve.png'):
            st.image('outputs/plots/final_evaluation/roc_curve.png', caption="ROC Curve")
        if os.path.exists('outputs/plots/final_evaluation/confusion_matrix.png'):
            st.image('outputs/plots/final_evaluation/confusion_matrix.png', caption="Confusion Matrix")
    with colB:
        if os.path.exists('outputs/plots/final_evaluation/score_distribution_detailed.png'):
            st.image('outputs/plots/final_evaluation/score_distribution_detailed.png', caption="Detailed Score Distribution")
        elif os.path.exists('outputs/plots/final_evaluation/score_distribution.png'):
            st.image('outputs/plots/final_evaluation/score_distribution.png', caption="Score Distribution")
            
        if os.path.exists('outputs/plots/final_evaluation/per_attack_f1.png'):
            st.image('outputs/plots/final_evaluation/per_attack_f1.png', caption="Performance by Attack")

with tab3:
    st.header("About")
    st.markdown("""
    ### Problem
    Synthetic and deepfake speech can be misused for impersonation, fraud, misinformation, and security threats. Identifying AI-generated audio is a critical challenge.
    
    ### Pipeline
    `Audio Upload → Preprocessing → Log-Mel Features → CNN → Prediction → Grad-CAM`
    
    ### Model
    - **Architecture**: AudioDeepfakeCNN (Custom PyTorch CNN with Global Average Pooling)
    - **Input**: 128 × 251 Log-Mel spectrogram
    - **Sample rate**: 16 kHz
    - **Analysis duration**: 4 seconds
    
    ### Limitations
    The final held-out evaluation demonstrated a significant **domain-shift and generalization challenge**. While the model performs exceptionally well on data similar to its training distribution (ASVspoof 2019 LA), it shows strong precision but substantially lower recall on the ASVspoof 2021 DF evaluation samples. Because of this, it may miss highly sophisticated or heavily compressed synthetic speech. 
    
    **This system is an academic research prototype and does not provide forensic certainty.**
    """)
