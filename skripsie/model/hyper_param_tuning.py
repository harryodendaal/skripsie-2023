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

# Define the model parameters
model_params = {
    "log": {
        "model": LogisticRegression(),
        "params": {
            "penalty": ["l1", "l2", "elasticnet"],
            "C": np.logspace(-4, 4, 20),  # smaller values mean stronger regularization
            "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
            "max_iter": [2000],
        },
    },
    "rfc": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [int(x) for x in np.linspace(start=10, stop=80, num=10)],
            "max_features": [
                "auto",
                "sqrt",
            ],
            "max_depth": [2, 4],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [
                True,
                False,
            ],  # method of selecting samples for training each true
        },
    },
    "svm": {
        "model": SVC(),
        "params": {
            "C": uniform(loc=0.1, scale=10),
            "kernel": ["rbf", "poly", "sigmoid", "linear"],
            "degree": [1, 2, 3, 4, 5, 6],
        },
    },
    "mlp": {
        "model": MLPClassifier(),
        "params": {
            "hidden_layer_sizes": [
                (4, 16)
            ],  # Two hidden layers with 4 and 16 perceptrons
            "activation": ["relu"],  # Non-linear activation function
            "alpha": [0.0001],  # Regularization parameter
        },
    },
    "ada": {
        "model": AdaBoostClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.0001, 0.01, 0.1, 1.0],
        },
    },
}


# def save_results(
#     results, model_name, feature_set_name, accuracy_best, precision, recall, f1, datasetname
# ):
#     results[model_name + " " + feature_set_name + " " + datasetname] = {
#         "accuracy_best": accuracy_best,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#     }
#     return results
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
    if model_name == "log":
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
    elif model_name == "rfc":
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
    elif model_name == "ada":
        results[model_name + " " + feature_set_name + " " + datasetname] = {
            "n_estimators": param_grid["n_estimators"],
            "learning_rate": param_grid["learning_rate"],
            "accuracy_best": accuracy_best,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return results


if __name__ == "__main__":
    # Create dictionaries to store results for different models
    results_dict = {
        "log": {},
        "rfc": {},
        "svm": {},
        "mlp": {},
        "ada": {},
    }

    dataset_paths_default = [
        "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_anxiety.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_autism.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_bipolar.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_bpd.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_depression.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_schizophrenia.csv",
    ]
    dataset_path_small = [
        "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_anxiety.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_autism.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_bipolar.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_bpd.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_depression.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_schizophrenia.csv",
    ]
    dataset_path_medium = [
        "/home/harry/Documents/skripsie_workspace/DATA/medium/mhrs_medium_anxiety.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/medium/mhrs_medium_autism.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/medium/mhrs_medium_bipolar.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/medium/mhrs_medium_bpd.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/medium/mhrs_medium_depression.csv",
        "/home/harry/Documents/skripsie_workspace/DATA/medium/mhrs_medium_schizophrenia.csv",
    ]
    dataset_names = [
        "anxiety",
        "autism",
        "bipolar",
        "bpd",
        "depression",
        "schizophrenia",
    ]

    sub_folder = "sml"

    # # dreaddit toggle.
    data = "DREADDIT"
    # data=""
    datasetpath = ["/home/penguin/Projects/SKRIPSIE/DATA/processed.csv"]
    dataset_names = ["DREADDIT_new"]

    for path, datasetname in zip(datasetpath, dataset_names):
        print("starting")
        full = pd.read_csv(path)
        full.rename(columns={"Unnamed: 0": "id"}, inplace=True)
        full = full.drop("id", axis=1)
        if data == "DREADDIT":
            full = full.drop("post_id", axis=1)
            full = full.drop("sentence_range", axis=1)
        full = full.drop(columns=["subreddit"])
        full = full.dropna()

        X = full.drop(columns=["label"])
        y = full["label"]
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        X_train = X_train_full["text"]
        X_test = X_test_full["text"]

        (
            X_train_unigram,
            X_test_unigram,
        ) = get_uni_gram_features(1500, X_train, X_test)
        (
            X_train_bigram,
            X_test_bigram,
        ) = get_bi_gram_features(1500, X_train, X_test)
        # X_train_liwc, X_test_liwc = get_liwc_features(full)
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

        for feature_set_name, (X_train_feat, X_test_feat) in feature_sets.items():
            for model_name, mp in model_params.items():
                clf = RandomizedSearchCV(
                    mp["model"],
                    mp["params"],
                    n_iter=5,
                    cv=3,
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

                accuracy_best = best_model.score(X_test_feat, y_test)

                # Calculate precision, recall, and F1 score
                y_pred = best_model.predict(X_test_feat)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                results_dict[model_name] = save_results(
                    results_dict[model_name],
                    model_name,
                    feature_set_name,
                    param_grid,
                    accuracy_best,
                    precision,
                    recall,
                    f1,
                    datasetname,
                )

            # Save results to a JSON file for each model
        if data == "DREADDIT":
            print("hello")
            for model_name, results in results_dict.items():
                with open(
                    f"Results_default/Dreaddit/new/{model_name}_{datasetname}_results.json",
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)
        else:
            for model_name, results in results_dict.items():
                with open(
                    f"Results_default/{model_name}/{sub_folder}/{model_name}_{datasetname}_default_results.json",
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)
