import json

import numpy as np
import pandas as pd
from feature_extraction.feature_extraction import (
    get_all_liwc_features,
    get_bi_gram_features,
    get_LDA_features,
    get_liwc_features,
    get_uni_gram_features,
)
from scipy.sparse import hstack
from scipy.stats import uniform
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

model_params = {
    "logistic_regression": {
        "model": LogisticRegression(),
        "params": {
            "penalty": ["l1", "l2", "elasticnet"],
            "C": np.logspace(-4, 4, 20),  # smaller values mean stronger regularization
            "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
            "max_iter": [2000],
        },
    },
    # "random_forest": {
    #     "model": RandomForestClassifier(),
    #     "params": {
    #         "n_estimators": [int(x) for x in np.linspace(start=10, stop=80, num=10)],
    #         "max_features": [
    #             "auto",
    #             "sqrt",
    #         ],
    #         "max_depth": [2, 4],
    #         "min_samples_split": [2, 5],
    #         "min_samples_leaf": [1, 2],
    #         "bootstrap": [
    #             True,
    #             False,
    #         ],  # method of selecting samples for training each true
    #     },
    # },
    # "svm": {
    #     "model": SVC(),
    #     "params": {
    #         "C": uniform(loc=0.1, scale=10),
    #         "kernel": ["rbf", "poly", "sigmoid", "linear"],
    #         "degree": [1, 2, 3, 4, 5, 6],
    #     },
    # },
    # "mlp": {
    #     "model": MLPClassifier(),
    #     "params": {
    #         "hidden_layer_sizes": [
    #             (4, 16)
    #         ],  # Two hidden layers with 4 and 16 perceptrons
    #         "activation": ["relu"],  # Non-linear activation function
    #         "alpha": [0.0001],  # Regularization parameter
    #     },
    # },
    # "adaboost": {
    #     "model": AdaBoostClassifier(),
    #     "params": {
    #         "n_estimators": [50, 100, 200],
    #         "learning_rate": [0.0001, 0.01, 0.1, 1.0],
    #     },
    # },
}


# also save precision, f1 and recall.
def save_results(
    results,
    model_name,
    feature_set_name,
    param_grid,
    accuracy_best,
    precision,
    recall,
    f1,
    datasetname,
):
    if model_name == "logistic_regression":
        results[model_name + " " + feature_set_name + " " + datasetname] = {
            "C": param_grid["C"],
            "max_iter": param_grid["max_iter"],
            "penalty": param_grid["penalty"],
            "solver": param_grid["solver"],
            "accuracy_best": accuracy_best,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    elif model_name == "svm":
        results[model_name + " " + feature_set_name + " " + datasetname] = {
            "C": param_grid["C"],
            "kernel": param_grid["kernel"],
            "degree": param_grid["degree"],
            "accuracy_best": accuracy_best,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    elif model_name == "random_forest":
        results[model_name + " " + feature_set_name + " " + datasetname] = {
            "n_estimators": param_grid["n_estimators"],
            "max_features": param_grid["max_features"],
            "max_depth": param_grid["max_depth"],
            "min_samples_split": param_grid["min_samples_split"],
            "min_samples_leaf": param_grid["min_samples_leaf"],
            "bootstrap": param_grid["bootstrap"],
            "accuracy_best": accuracy_best,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    elif model_name == "mlp":
        results[model_name + " " + feature_set_name + " " + datasetname] = {
            "hidden_layer_sizes": param_grid["hidden_layer_sizes"],
            "activation": param_grid["activation"],
            "alpha": param_grid["alpha"],
            "accuracy_best": accuracy_best,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    elif model_name == "adaboost":
        results[model_name + " " + feature_set_name + " " + datasetname] = {
            "n_estimators": param_grid["n_estimators"],
            "learning_rate": param_grid["learning_rate"],
            "accuracy_best": accuracy_best,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return results


# can use RandomizedSearchCV where
# just give n_iter=value when get larger dataset.

# TODO
# play around amount of n-grams extracted as well as num topics

if __name__ == "__main__":
    # download the data.
    datasetname = "angs"

    # EXTRA
    full = pd.read_csv(
        "/home/penguin/Projects/SKRIPSIE/DATA/Small/mhrs_small_anxiety.csv"
    )
    full.rename(columns={"Unnamed: 0": "id"}, inplace=True)
    full = full.drop("id", axis=1)
    full = full.drop(columns=["Subreddit"])
    full = full.dropna()
    # # ________________________________________________________________________

    results = {}
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        full, full["label"], test_size=0.15, random_state=42
    )
    X_train = X_train_full["Text"]
    X_test = X_test_full["Text"]

    (
        X_train_unigram,
        X_test_unigram,
    ) = get_uni_gram_features(1500, X_train, X_test)
    (
        X_train_bigram,
        X_test_bigram,
    ) = get_bi_gram_features(1500, X_train, X_test)
    # (X_train_liwc, X_test_liwc) = get_liwc_features(full)
    X_train_liwc, X_test_liwc = get_all_liwc_features(X_train_full, X_test_full)
    X_train_liwc = X_train_liwc.astype(np.float64)
    X_test_liwc = X_test_liwc.astype(np.float64)

    (X_train_lda, X_test_lda) = get_LDA_features(X_train, X_test)

    # Combine the feature matrices
    X_train_liwc_lda_uni = hstack([X_train_liwc, X_train_lda, X_train_unigram])
    X_test_liwc_lda_uni = hstack([X_test_liwc, X_test_lda, X_test_unigram])
    X_train_liwc_lda_bi = hstack([X_train_liwc, X_train_lda, X_train_bigram])
    X_test_liwc_lda_bi = hstack([X_test_liwc, X_test_lda, X_test_bigram])

    feature_sets = {
        "bigram": (X_train_bigram, X_test_bigram),
        "unigram": (X_train_unigram, X_test_unigram),
        "LIWC": (X_train_liwc, X_test_liwc),
        "LDA": (X_train_lda, X_test_lda),
        "LIWC-LDA-UNIGRAM": (X_train_liwc_lda_uni, X_test_liwc_lda_uni),
        "LIWC-LDA-BIGRAM": (X_train_liwc_lda_bi, X_test_liwc_lda_bi),
    }
    # print(feature_sets)
    for feature_set_name, (X_train_feat, X_test_feat) in feature_sets.items():
        for model_name, mp in model_params.items():
            clf = RandomizedSearchCV(
                mp["model"],
                mp["params"],
                n_iter=20,
                cv=5,
                verbose=10,
                return_train_score=True,
                n_jobs=-1,
                random_state=42,
            )
            best_clf = clf.fit(X_train_feat, y_train)

            param_grid = best_clf.best_params_
            best_model = mp["model"]
            best_model.set_params(**param_grid)

            best_model.fit(X_train_feat, y_train)
            standard = mp["model"]
            standard.fit(X_train_feat, y_train)

            accuracy_best = best_model.score(X_test_feat, y_test)
            accuracy_standard = standard.score(X_test_feat, y_test)

            # Calculate precision, recall, and F1 score
            y_pred = best_model.predict(X_test_feat)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results = save_results(
                results,
                model_name,
                feature_set_name,
                param_grid,
                accuracy_best,
                precision,
                recall,
                f1,
                datasetname,
            )

        # Save results to a JSON file
        with open(
            f"Results_tuning/{model_name}/{datasetname}_{model_name}_tuned_hyperparameters.json",
            "w",
        ) as f:
            json.dump(results, f, indent=4)
