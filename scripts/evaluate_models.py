import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

from geosituation.perception.classifier import IncidentClassifier
from geosituation.config import INCIDENT_CATEGORIES

def run_evaluation_benchmark():
    print("================================================================================")
    print("   GEOSITUATION-NLP: CLASSIFICATION PERFORMANCE & ABLATION EVALUATION BENCHMARK  ")
    print("================================================================================\n")

    classifier = IncidentClassifier()
    texts, labels = classifier._get_seed_training_data()

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    accuracies = []
    macro_f1s = []
    weighted_f1s = []
    precisions = []
    recalls = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Fit model on fold
        classifier.pipeline.fit(train_texts, train_labels)
        preds = classifier.pipeline.predict(val_texts)

        acc = accuracy_score(val_labels, preds)
        mf1 = f1_score(val_labels, preds, average='macro', zero_division=0)
        wf1 = f1_score(val_labels, preds, average='weighted', zero_division=0)
        prec = precision_score(val_labels, preds, average='macro', zero_division=0)
        rec = recall_score(val_labels, preds, average='macro', zero_division=0)

        accuracies.append(acc)
        macro_f1s.append(mf1)
        weighted_f1s.append(wf1)
        precisions.append(prec)
        recalls.append(rec)

        print(f"Fold {fold}: Accuracy = {acc:.3f} | Macro-F1 = {mf1:.3f} | Precision = {prec:.3f} | Recall = {rec:.3f}")

    print("\n--- 4-FOLD STRATIFIED CROSS-VALIDATION SUMMARY ---")
    print(f"Mean Classification Accuracy : {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"Mean Macro-Averaged F1 Score : {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")
    print(f"Mean Weighted-Averaged F1    : {np.mean(weighted_f1s):.4f} +/- {np.std(weighted_f1s):.4f}")
    print(f"Mean Precision               : {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}")
    print(f"Mean Recall                  : {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}")

    # Retrain on full dataset
    classifier.train_seed_model()
    full_preds = classifier.pipeline.predict(texts)
    print("\n--- DETAILED CLASS-WISE REPORT ---")
    print(classification_report(labels, full_preds))

if __name__ == "__main__":
    run_evaluation_benchmark()
