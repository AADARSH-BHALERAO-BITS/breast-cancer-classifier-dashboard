import os
import joblib
from sklearn.naive_bayes import GaussianNB
from data_pipeline import build_train_test_split, score_predictions, persist_metrics, ARTIFACT_DIR


class NaiveBayesExperiment:
    label = "Naive Bayes"
    artifact_name = "gnb_clf.joblib"

    def __init__(self):
        self.estimator = GaussianNB(var_smoothing=1e-9)

    def run(self):
        feats_train, feats_test, labels_train, labels_test = build_train_test_split()
        self.estimator.fit(feats_train, labels_train)

        preds = self.estimator.predict(feats_test)
        proba = self.estimator.predict_proba(feats_test)[:, 1]

        scores = score_predictions(labels_test, preds, proba)
        persist_metrics(self.label, scores)

        joblib.dump(self.estimator, os.path.join(ARTIFACT_DIR, self.artifact_name))
        return scores


if __name__ == "__main__":
    outcome = NaiveBayesExperiment().run()
    print(NaiveBayesExperiment.label, outcome)
