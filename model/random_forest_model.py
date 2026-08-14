import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from data_pipeline import build_train_test_split, score_predictions, persist_metrics, ARTIFACT_DIR


class RandomForestExperiment:
    label = "Random Forest"
    artifact_name = "rf_clf.joblib"

    def __init__(self, tree_count=300):
        self.estimator = RandomForestClassifier(
            n_estimators=tree_count, max_depth=None, random_state=11, n_jobs=-1
        )

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
    outcome = RandomForestExperiment().run()
    print(RandomForestExperiment.label, outcome)
