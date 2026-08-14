import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

CLASSIFIER_REGISTRY = {
    "Logistic Regression": "logreg_clf.joblib",
    "Decision Tree": "dtree_clf.joblib",
    "kNN": "knn_clf.joblib",
    "Naive Bayes": "gnb_clf.joblib",
    "Random Forest": "rf_clf.joblib",
}

st.set_page_config(page_title="Tumor Diagnosis Classifier Dashboard", layout="wide")

st.title("Tumor Diagnosis Classifier Dashboard")
st.write(
    "Upload the held-out test split and pick a trained classifier to inspect "
    "how it performs on unseen breast tumor measurements."
)


@st.cache_resource
def fetch_preprocessing_assets():
    standardizer = joblib.load(os.path.join(ARTIFACT_DIR, "feature_scaler.joblib"))
    predictor_columns = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.joblib"))
    return standardizer, predictor_columns


@st.cache_resource
def fetch_classifier(model_key):
    artifact_path = os.path.join(ARTIFACT_DIR, CLASSIFIER_REGISTRY[model_key])
    return joblib.load(artifact_path)


standardizer, predictor_columns = fetch_preprocessing_assets()

csv_upload = st.file_uploader("Upload test_data.csv", type=["csv"])
chosen_model_key = st.selectbox("Select a Classifier", list(CLASSIFIER_REGISTRY.keys()))

if csv_upload is not None:
    incoming_frame = pd.read_csv(csv_upload)

    if "diagnosis" not in incoming_frame.columns:
        st.error("Uploaded CSV must contain a 'diagnosis' target column.")
    else:
        absent_columns = [c for c in predictor_columns if c not in incoming_frame.columns]
        if absent_columns:
            st.error(f"Uploaded CSV is missing required feature columns: {absent_columns}")
        else:
            feature_block = incoming_frame[predictor_columns]
            ground_truth = incoming_frame["diagnosis"]
            feature_block_std = standardizer.transform(feature_block)

            active_model = fetch_classifier(chosen_model_key)
            predicted_labels = active_model.predict(feature_block_std)

            if hasattr(active_model, "predict_proba"):
                positive_class_scores = active_model.predict_proba(feature_block_std)[:, 1]
            else:
                positive_class_scores = predicted_labels

            st.subheader("Preview of Uploaded Data")
            st.dataframe(incoming_frame.head())

            perf_accuracy = accuracy_score(ground_truth, predicted_labels)
            perf_auc = roc_auc_score(ground_truth, positive_class_scores)
            perf_precision = precision_score(ground_truth, predicted_labels)
            perf_recall = recall_score(ground_truth, predicted_labels)
            perf_f1 = f1_score(ground_truth, predicted_labels)
            perf_mcc = matthews_corrcoef(ground_truth, predicted_labels)

            st.subheader(f"Evaluation Metrics: {chosen_model_key}")
            metric_slots = st.columns(6)
            metric_slots[0].metric("Accuracy", f"{perf_accuracy:.4f}")
            metric_slots[1].metric("AUC", f"{perf_auc:.4f}")
            metric_slots[2].metric("Precision", f"{perf_precision:.4f}")
            metric_slots[3].metric("Recall", f"{perf_recall:.4f}")
            metric_slots[4].metric("F1 Score", f"{perf_f1:.4f}")
            metric_slots[5].metric("MCC", f"{perf_mcc:.4f}")

            left_pane, right_pane = st.columns(2)

            with left_pane:
                st.subheader("Confusion Matrix")
                conf_mat = confusion_matrix(ground_truth, predicted_labels)
                fig, ax = plt.subplots()
                sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Purples", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with right_pane:
                st.subheader("Classification Report")
                report_dict = classification_report(ground_truth, predicted_labels, output_dict=True)
                report_frame = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_frame.round(3))

            st.subheader("Compare Every Classifier on This Upload")
            if st.button("Run All Classifiers"):
                leaderboard_rows = []
                for model_key in CLASSIFIER_REGISTRY.keys():
                    candidate_model = fetch_classifier(model_key)
                    candidate_preds = candidate_model.predict(feature_block_std)
                    if hasattr(candidate_model, "predict_proba"):
                        candidate_scores = candidate_model.predict_proba(feature_block_std)[:, 1]
                    else:
                        candidate_scores = candidate_preds
                    leaderboard_rows.append(
                        {
                            "Model": model_key,
                            "Accuracy": round(accuracy_score(ground_truth, candidate_preds), 4),
                            "AUC": round(roc_auc_score(ground_truth, candidate_scores), 4),
                            "Precision": round(precision_score(ground_truth, candidate_preds), 4),
                            "Recall": round(recall_score(ground_truth, candidate_preds), 4),
                            "F1": round(f1_score(ground_truth, candidate_preds), 4),
                            "MCC": round(matthews_corrcoef(ground_truth, candidate_preds), 4),
                        }
                    )
                st.dataframe(pd.DataFrame(leaderboard_rows))
else:
    st.info("Please upload the test_data.csv file to begin.")
