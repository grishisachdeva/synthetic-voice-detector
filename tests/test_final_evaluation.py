import pytest
import os
import json
import torch
from src.model import AudioDeepfakeCNN

def test_frozen_configuration_loading():
    config_path = 'outputs/metrics/final_frozen_configuration.json'
    if not os.path.exists(config_path):
        pytest.skip("Frozen config not generated yet.")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    assert 'model_name' in config
    assert 'classification_threshold' in config
    assert config['classification_threshold'] == 0.5
    assert 'checkpoint' in config
    assert os.path.exists(config['checkpoint'])

def test_model_instantiation_from_frozen_config():
    config_path = 'outputs/metrics/final_frozen_configuration.json'
    if not os.path.exists(config_path):
        pytest.skip()
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model = AudioDeepfakeCNN()
    # Should load without errors
    checkpoint = torch.load(config['checkpoint'], map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    assert not model.training  # Needs to be explicitly put in eval, we test if it runs

def test_prediction_generation():
    pred_path = 'outputs/predictions/final_evaluation/final_df_predictions.csv'
    if not os.path.exists(pred_path):
        pytest.skip()
        
    import pandas as pd
    df = pd.read_csv(pred_path)
    assert 'file_id' in df.columns
    assert 'spoof_probability' in df.columns
    assert 'predicted_label' in df.columns
    assert df['spoof_probability'].min() >= 0.0
    assert df['spoof_probability'].max() <= 1.0

def test_final_results_json():
    res_path = 'outputs/metrics/final_results.json'
    if not os.path.exists(res_path):
        pytest.skip()
        
    with open(res_path, 'r') as f:
        data = json.load(f)
        
    assert 'accuracy' in data
    assert 'roc_auc' in data
    assert 'eer' in data
    assert data['total_official_files'] == 60176
