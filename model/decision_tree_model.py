import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from data_pipeline import build_train_test_split, score_predictions, persist_metrics, ARTIFACT_DIR


class DecisionTreeExperiment:
    label = "Decision Tree"
    artifact_name = "dtree_clf.joblib"

    def __init__(self):
        self.estimator = DecisionTreeClassifier(criterion="gini", max_depth=6, random_state=11)

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
    outcome = DecisionTreeExperiment().run()
    print(DecisionTreeExperiment.label, outcome)
