import numpy as np
from feature_extraction.LIWC import (
    liwc_extract_features_correlation_matrix,
    liwc_get_correlation_matrix,
)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split


def get_all_liwc_features(X_train_full, X_test_full):
    # Drop the "Text" column to get all other columns
    X_train_liwc = X_train_full.drop(columns=["text"])
    X_test_liwc = X_test_full.drop(columns=["text"])
    return X_train_liwc, X_test_liwc


def get_uni_gram_features(max_features, train, test):
    uni_vectorizer = TfidfVectorizer(
        ngram_range=(1, 1), max_features=max_features, stop_words="english"
    )
    X_train_tfidf_unigram = uni_vectorizer.fit_transform(train)
    X_test_tfidf_unigram = uni_vectorizer.transform(test)

    return X_train_tfidf_unigram, X_test_tfidf_unigram


def get_bi_gram_features(max_features, train, test):
    bi_vectorizer = TfidfVectorizer(
        ngram_range=(2, 2), max_features=max_features, stop_words="english"
    )
    X_train_tfidf_bigram = bi_vectorizer.fit_transform(train)
    X_test_tfidf_bigram = bi_vectorizer.transform(test)

    return X_train_tfidf_bigram, X_test_tfidf_bigram


def get_liwc_features(full):
    correlation_matrix = liwc_get_correlation_matrix(full=full)
    # liwc_print_correlation_matrix(correlation_matrix=correlation_matrix)
    reduced_features = liwc_extract_features_correlation_matrix(
        correlation_matrix=correlation_matrix, full=full
    )
    X_train_liwc, X_test_liwc, y_train_liwc, y_test_liwc = train_test_split(
        reduced_features, full["label"], test_size=0.15, random_state=42
    )
    return X_train_liwc, X_test_liwc


def get_LDA_features(X_train, X_test, num_topics=10):
    # create and fit the countvectorizer
    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # apply laten dirichlet allocationa (LDA)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    X_train_lda = lda.fit_transform(X_train_vectorized)
    X_test_lda = lda.transform(X_test_vectorized)

    return X_train_lda, X_test_lda


# testing.
if __name__ == "__main__":
    # text_data
    text_data = [
        "I am interested in NLP",
        "This is a good tutorial with good topic",
        "Feature extraction is very important topic",
    ]
    # TODO
    # play around with hyperparamters amount of features etc etc.
