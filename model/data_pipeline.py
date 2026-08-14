import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ARTIFACT_DIR)
RANDOM_SEED = 7
HOLDOUT_FRACTION = 0.3


def load_tumor_dataset():
    bundle = load_breast_cancer(as_frame=True)
    frame = bundle.frame.rename(columns={"target": "diagnosis"})
    predictor_frame = frame.drop(columns=["diagnosis"])
    outcome_series = frame["diagnosis"]
    return predictor_frame, outcome_series


def build_train_test_split():
    predictor_frame, outcome_series = load_tumor_dataset()

    feats_train, feats_test, labels_train, labels_test = train_test_split(
        predictor_frame,
        outcome_series,
        test_size=HOLDOUT_FRACTION,
        random_state=RANDOM_SEED,
        stratify=outcome_series,
    )

    standardizer = StandardScaler()
    feats_train_std = standardizer.fit_transform(feats_train)
    feats_test_std = standardizer.transform(feats_test)

    joblib.dump(standardizer, os.path.join(ARTIFACT_DIR, "feature_scaler.joblib"))
    joblib.dump(list(predictor_frame.columns), os.path.join(ARTIFACT_DIR, "feature_columns.joblib"))

    if not os.path.exists(os.path.join(REPO_ROOT, "test_data.csv")):
        export_frame = feats_test.copy()
        export_frame["diagnosis"] = labels_test.values
        export_frame.to_csv(os.path.join(REPO_ROOT, "test_data.csv"), index=False)

    return feats_train_std, feats_test_std, labels_train, labels_test


def score_predictions(labels_true, labels_pred, probability_scores):
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
    )

    return {
        "Accuracy": round(accuracy_score(labels_true, labels_pred), 4),
        "AUC": round(roc_auc_score(labels_true, probability_scores), 4),
        "Precision": round(precision_score(labels_true, labels_pred), 4),
        "Recall": round(recall_score(labels_true, labels_pred), 4),
        "F1": round(f1_score(labels_true, labels_pred), 4),
        "MCC": round(matthews_corrcoef(labels_true, labels_pred), 4),
    }


def persist_metrics(model_label, score_dict):
    import json

    summary_path = os.path.join(ARTIFACT_DIR, "performance_log.json")
    existing = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as fh:
            existing = json.load(fh)
    existing[model_label] = score_dict
    with open(summary_path, "w") as fh:
        json.dump(existing, fh, indent=2)
