import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# for now copy pasta it then generalize
def lda_topic_proportions(stress_posts, non_stress_posts):
    stress_text = stress_posts["text"]
    non_stress_text = non_stress_posts["text"]

    vectorizer = CountVectorizer(max_features=1000)
    X_stress = vectorizer.fit_transform(stress_text)
    X_non_stress = vectorizer.transform(non_stress_text)

    # LDA performed on stress and non_stress datasets
    n_topics = 20
    lda_stress = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_proportions_stress = lda_stress.fit_transform(X_stress)

    lda_non_stress = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_proportions_non_stress = lda_non_stress.fit_transform(X_non_stress)

    stress_topic_names = []
    non_stress_topic_names = []
    for lda_model, topic_names in [
        (lda_stress, stress_topic_names),
        (lda_non_stress, non_stress_topic_names),
    ]:
        n_topics = lda_model.n_components
        feature_names = vectorizer.get_feature_names_out()
        n_top_words = 10

        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[: -n_top_words - 1 : -1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_name = f"Topic {topic_idx + 1}: {' '.join(top_words)}"
            topic_names.append(topic_name)

    return (
        topic_proportions_stress,
        topic_proportions_non_stress,
        stress_topic_names,
        non_stress_topic_names,
    )


def print_log_acc(
    topic_proportions_stress,
    topic_proportions_non_stress,
    stress_posts,
    non_stress_posts,
):
    # Concatenate topic proportions and labels for stress and non-stress subsets
    X_combined = np.concatenate(
        (topic_proportions_stress, topic_proportions_non_stress), axis=0
    )
    y_combined = np.concatenate(
        (stress_posts["label"], non_stress_posts["label"]), axis=0
    )

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42
    )

    # Initialize and train the classification model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    full = pd.read_csv("/home/penguin/Projects/SKIPSIE/DATA/processed.csv")

    grouped = full.groupby("label")
    stress_posts = grouped.get_group(1)
    non_stress_posts = grouped.get_group(0)
    (
        topic_proportions_stress,
        topic_proportions_non_stress,
        stress_topic_names,
        non_stress_topic_names,
    ) = lda_topic_proportions(stress_posts, non_stress_posts)

    print(stress_topic_names)
    print(non_stress_topic_names)
    print(topic_proportions_non_stress)
    print(topic_proportions_stress)
    print_log_acc(
        topic_proportions_stress,
        topic_proportions_non_stress,
        stress_posts,
        non_stress_posts,
    )
