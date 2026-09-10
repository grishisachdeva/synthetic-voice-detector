import os

def update_readme_with_explainability():
    print("Updating README with Explainability section...")
    with open("README.md", "r") as f:
        readme = f.read()
        
    if "## Explainability" not in readme:
        readme += "\n\n## Explainability\n"
        readme += "To understand what the model is looking at, we implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)**.\n\n"
        readme += "### What is Grad-CAM?\n"
        readme += "Grad-CAM visualizes the internal activations of the final convolutional layer (`model.block4[0]`) by weighting them with the gradients backpropagated from the target class prediction (either SPOOF or BONAFIDE).\n\n"
        readme += "### What the Heatmap Represents\n"
        readme += "The resulting heatmap shows exactly which time-frequency regions of the Log-Mel Spectrogram contributed most strongly to the model's decision.\n\n"
        readme += "### Limitations & Correct Interpretation\n"
        readme += "> [!WARNING]\n"
        readme += "> This visualization highlights **model attention**, NOT ground-truth forensic artifacts. If the model is wrong (e.g., False Positives or False Negatives), the highlighted regions show what fooled the model, not what a human should consider a true deepfake artifact.\n\n"
        readme += "### How it helps\n"
        readme += "By observing True Positives and False Negatives, we can qualitatively understand which types of acoustic textures the model learned to associate with 'fake' speech during its training on the 2019 dataset, and why it fails to generalize to certain novel 2021 attacks.\n"
        
        with open("README.md", "w") as f:
            f.write(readme)
            
    print("README updated successfully.")

if __name__ == "__main__":
    update_readme_with_explainability()
