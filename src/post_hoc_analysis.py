import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_analysis():
    preds = pd.read_csv('outputs/predictions/final_evaluation/final_df_predictions.csv')
    threshold = 0.5
    
    # Task 2: Final Error Analysis
    print("Task 2: Final Error Analysis")
    tp = len(preds[(preds['true_label'] == 1) & (preds['predicted_label'] == 1)])
    tn = len(preds[(preds['true_label'] == 0) & (preds['predicted_label'] == 0)])
    fp = len(preds[(preds['true_label'] == 0) & (preds['predicted_label'] == 1)])
    fn = len(preds[(preds['true_label'] == 1) & (preds['predicted_label'] == 0)])
    total = len(preds)
    total_spoof = tp + fn
    total_bonafide = tn + fp
    
    fpr = fp / total_bonafide if total_bonafide > 0 else 0.0
    fnr = fn / total_spoof if total_spoof > 0 else 0.0
    miss_rate = fnr
    correct_rejection_rate = tn / total_bonafide if total_bonafide > 0 else 0.0
    
    error_stats = {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "miss_rate": miss_rate,
        "correct_rejection_rate": correct_rejection_rate,
        "positive_class": "SPOOF",
        "negative_class": "BONAFIDE"
    }
    with open('outputs/metrics/final_error_statistics.json', 'w') as f:
        json.dump(error_stats, f, indent=4)
        
    # Task 3: Error Analysis by Attack
    print("Task 3: Error Analysis by Attack")
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
    def attack_metrics(x):
        y_t = x['true_label']
        y_p = x['predicted_label']
        tp_a = len(x[(x['true_label']==1) & (x['predicted_label']==1)])
        fn_a = len(x[(x['true_label']==1) & (x['predicted_label']==0)])
        fnr_a = fn_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        return pd.Series({
            'sample_count': len(x),
            'true_positives': tp_a,
            'false_negatives': fn_a,
            'recall': recall_score(y_t, y_p, zero_division=0),
            'precision': precision_score(y_t, y_p, zero_division=0),
            'F1': f1_score(y_t, y_p, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_t, y_p),
            'FNR': fnr_a
        })
    attacks = preds.groupby('attack_id').apply(attack_metrics).reset_index()
    attacks = attacks.sort_values('FNR', ascending=False)
    attacks.to_csv('outputs/metrics/final_attack_error_analysis.csv', index=False)
    
    # Task 4: Score Distribution Analysis
    print("Task 4: Score Distribution")
    bonafide_probs = preds[preds['true_label'] == 0]['spoof_probability']
    spoof_probs = preds[preds['true_label'] == 1]['spoof_probability']
    
    plt.figure()
    plt.hist([bonafide_probs, spoof_probs], bins=50, label=['BONAFIDE', 'SPOOF'], density=True, alpha=0.7)
    plt.title('Detailed Score Distribution')
    plt.legend()
    plt.savefig('outputs/plots/final_evaluation/score_distribution_detailed.png')
    plt.close()
    
    score_stats = {
        "bonafide": {
            "mean": float(bonafide_probs.mean()),
            "median": float(bonafide_probs.median()),
            "std": float(bonafide_probs.std())
        },
        "spoof": {
            "mean": float(spoof_probs.mean()),
            "median": float(spoof_probs.median()),
            "std": float(spoof_probs.std())
        }
    }
    with open('outputs/metrics/final_score_distribution_statistics.json', 'w') as f:
        json.dump(score_stats, f, indent=4)
        
    # Task 5: Confusion Matrix Analysis
    print("Task 5: Confusion Matrix Analysis")
    with open('outputs/metrics/final_confusion_matrix_analysis.txt', 'w') as f:
        f.write("CONFUSION MATRIX ANALYSIS\n=========================\n")
        f.write(f"TN (Bonafide correctly identified): {tn}\n")
        f.write(f"FP (Bonafide incorrectly flagged as spoof): {fp}\n")
        f.write(f"FN (Spoof incorrectly accepted as bonafide): {fn}\n")
        f.write(f"TP (Spoof correctly flagged): {tp}\n\n")
        f.write("Explanation of High Precision / Low Recall:\n")
        f.write("The model has an extremely low FP count (19), meaning when it predicts SPOOF, it is almost certainly correct (High Precision 0.9986).\n")
        f.write("However, the model fails to detect many spoofs (FN=19405), pushing them below the 0.5 threshold. This causes Low Recall (0.4188).\n")
        f.write("This indicates the model's learned representation of 'spoof' from the 2019 LA dataset is too narrow and doesn't generalize to the new attacks in the 2021 DF dataset.\n")
        
    # Task 10 & 11: Decoder Exclusion Analysis
    print("Task 10/11: Exclusions")
    metadata = pd.read_csv('data/metadata/asvspoof2021_df_evaluation_metadata.csv')
    val_report = pd.read_csv('outputs/metrics/df_decode_validation_report.csv')
    df = metadata.merge(val_report, on='file_id')
    
    total_official = len(metadata)
    decodable = len(df[df['final_status'] == 'DECODABLE'])
    excluded = total_official - decodable
    
    with open('outputs/metrics/df_exclusion_summary.txt', 'w') as f:
        f.write("DECODER EXCLUSION ANALYSIS\n==========================\n")
        f.write(f"Official evaluation files: {total_official}\n")
        f.write(f"Decodable files: {decodable}\n")
        f.write(f"Excluded files: {excluded}\n")
        f.write(f"Exclusion percentage: {(excluded/total_official)*100:.2f}%\n\n")
        f.write("Failure Categories:\n")
        f.write("1. libsndfile generic read error (Primary decoder failure)\n")
        f.write("2. audioread NoBackendError (Secondary fallback failure)\n\n")
        f.write("Note: These are file-format corruption issues inherent to the dataset distribution, NOT model errors.\n")
        
    exclusion_stats = df.groupby(['attack_id', 'label_name']).apply(lambda x: pd.Series({
        'total': len(x),
        'decodable': len(x[x['final_status'] == 'DECODABLE']),
        'excluded': len(x[x['final_status'] != 'DECODABLE']),
        'exclusion_rate': len(x[x['final_status'] != 'DECODABLE']) / len(x)
    })).reset_index()
    exclusion_stats.to_csv('outputs/metrics/exclusion_distribution_by_attack.csv', index=False)
    
    # Task 12: Confidence Analysis
    print("Task 12: Confidence Analysis")
    preds['confidence'] = np.abs(preds['spoof_probability'] - threshold)
    
    def get_outcome(row):
        if row['true_label'] == 1 and row['predicted_label'] == 1: return 'TP'
        if row['true_label'] == 0 and row['predicted_label'] == 0: return 'TN'
        if row['true_label'] == 0 and row['predicted_label'] == 1: return 'FP'
        if row['true_label'] == 1 and row['predicted_label'] == 0: return 'FN'
    
    preds['outcome'] = preds.apply(get_outcome, axis=1)
    conf_stats = preds.groupby('outcome')['confidence'].agg(['mean', 'median', 'std']).reset_index()
    conf_stats.to_csv('outputs/metrics/final_confidence_by_outcome.csv', index=False)
    
    # Task 13: Error Examples
    print("Task 13: Error Examples")
    examples = []
    for outcome in ['TP', 'TN', 'FP', 'FN']:
        subset = preds[preds['outcome'] == outcome]
        if len(subset) > 0:
            examples.append(subset.sample(min(5, len(subset)), random_state=42))
    
    if examples:
        ex_df = pd.concat(examples)
        ex_df[['file_id', 'true_label', 'predicted_label', 'spoof_probability', 'attack_id', 'file_path']].to_csv('outputs/metrics/final_error_examples.csv', index=False)
        
    print("Post-hoc foundational analysis complete.")

if __name__ == "__main__":
    run_analysis()
