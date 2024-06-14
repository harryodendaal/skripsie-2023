import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# logistic regression, support vector machines, adaboost, multi layered perceptron,


def evaluate_classifier_result(classifier, X_train, X_test, y_train, y_test):
    model = classifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return accuracy, f1, precision, recall
