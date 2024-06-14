import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# TODO
# split dataset
# apply smote (only on training set.)
# 80% train and 20% test.
# XGboost with convolutinonal neural networks.
# Exclude thos who wrote over multiple subreddits.

# 1 for XGBOOST: convert into numerical representation using TF-IDF vectorizer from scikit learn.


def xgboost_algorithm(data):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)

    X_train_tfidf = tfidf_vectorizer.fit_transform(data["Text"])
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_tfidf, data["label"], test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier()

    sm = SMOTE()
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_class_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_class_1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"F1-Score for Class 0: {f1_class_0}")
    print(f"F1-Score for Class 1: {f1_class_1}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    # download the data.
    depression_dataset = pd.read_csv(
        "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_depression.csv"
    )
    depression_dataset.rename(columns={"Unnamed: 0": "id"}, inplace=True)

    # # ________________________________________________________________________
    depression_dataset = depression_dataset.drop("id", axis=1)
    depression_dataset = depression_dataset.dropna()

    # # Anxiety_dataset = pd.read_csv(
    # #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_anxiety.csv"
    # # )
    # # bpd_dataset = pd.read_csv(
    # #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_bpd.csv"
    # # )
    # # autism_dataset = pd.read_csv(
    # #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_autism.csv"
    # # )
    # # mentalhealth_dataset = pd.read_csv(
    # #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_mentalhealth.csv"
    # # )
    # # bipolar_dataset = pd.read_csv(
    # #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_bipolar.csv"
    # # )

    xgboost_algorithm(depression_dataset)

    # Research question: Can we identify whether a user’s post belongs to mental illnesses on social media?
