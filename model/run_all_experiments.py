import os
import pandas as pd

from logistic_regression_model import LogRegExperiment
from decision_tree_model import DecisionTreeExperiment
from knn_model import KNearestExperiment
from naive_bayes_model import NaiveBayesExperiment
from random_forest_model import RandomForestExperiment
from data_pipeline import ARTIFACT_DIR

EXPERIMENT_CLASSES = [
    LogRegExperiment,
    DecisionTreeExperiment,
    KNearestExperiment,
    NaiveBayesExperiment,
    RandomForestExperiment,
]


def main():
    leaderboard = {}
    for experiment_cls in EXPERIMENT_CLASSES:
        experiment = experiment_cls()
        leaderboard[experiment_cls.label] = experiment.run()

    leaderboard_frame = pd.DataFrame(leaderboard).T
    leaderboard_frame.index.name = "ML Model Name"
    leaderboard_frame.to_csv(os.path.join(ARTIFACT_DIR, "leaderboard.csv"))

    print(leaderboard_frame)
    print("All five classifiers trained and saved to the model/ directory.")


if __name__ == "__main__":
    main()
