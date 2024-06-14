import random
from collections import Counter

import gensim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# TODO:
# Word cloud visualizations
# vocabluary and word clouds
# STILL LEFT
# topic modeling.
# word2vec
# TOP2vec


def get_n_gram(text_list, n):
    concatenated_text = " ".join(text_list)
    tokens = concatenated_text.split()
    ngram_list = list(ngrams(tokens, n))
    ngram_counter = Counter(ngram_list)

    ngram_dict = {" ".join(ngram): freq for ngram, freq in ngram_counter.items()}

    return ngram_dict


def remove_common_words(ngram_dict_list):
    common_words = set.intersection(
        *(set(ngram_dict.keys()) for ngram_dict in ngram_dict_list)
    )

    for ngram_dict in ngram_dict_list:
        for common_word in common_words:
            ngram_dict.pop(common_word, None)


def print_basic_data_info(data):
    print("column names:")
    print(data.columns)
    print("______________")
    print("subreddits and counts:")

    subreddit_counts = data["Subreddit"].value_counts()
    print(subreddit_counts)

    print("______________")
    print("unique words in each subreddit:")


def unique_words_subreddits(data):
    subreddit_counts = data["Subreddit"].value_counts()
    for subreddit in subreddit_counts.index:
        subreddit_data = data[data["Subreddit"] == subreddit]
        subreddit_data["Text"] = subreddit_data["Text"].astype(str)
        subreddit_text = " ".join(subreddit_data["Text"])

        # Tokenize the text and calculate the number of unique words
        words = word_tokenize(subreddit_text)
        unique_words = set(words)

        print(f"Subreddit: {subreddit}, Unique Words Count: {len(unique_words)}")


def create_word_cloud(n_gram_dict, position, title):
    # Create a word cloud
    wordcloud = WordCloud(
        width=800, height=400, max_words=100, background_color="white"
    ).generate_from_frequencies(n_gram_dict)

    # Add the subplot
    plt.subplot(3, 3, position)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")


def create_word_cloud_plots(data, n):
    plt.figure(figsize=(12, 8))
    grouped_data = data[data["Subreddit"] != "mentalhealth"].groupby("Subreddit")
    n_grams = []
    names = []
    for s, (subreddit_name, subreddit_data) in enumerate(grouped_data):
        # print(f"Subreddit: {subreddit_name}")
        n_gram = get_n_gram(subreddit_data["Text"].astype(str), n=n)
        n_grams.append(n_gram)
        names.append(subreddit_name)

        if s == 5:
            remove_common_words(n_grams)
            for p, (n_gram) in enumerate(n_grams):
                create_word_cloud(
                    n_gram_dict=n_gram,
                    position=p + 1,
                    title=f"{n}-gram {names[p]}",
                )

    plt.savefig(
        f"/home/penguin/Projects/SKRIPSIE/exploratory_data_analysis/new_word_clouds/{n}-gram word cloud",
        dpi=300,
    )

    # now we have the n_grams now create plots.
    # for subreddit_name in grouped_data:


def post_lengths_plots(data):
    plt.figure(figsize=(15, 10))
    grouped_data = data.groupby("Subreddit")

    for p, (subreddit_name, subreddit_data) in enumerate(grouped_data):
        print(subreddit_name)
        seq_len = [len(i.split()) for i in subreddit_data["Text"]]
        # Calculate the 95th percentile
        percentile_95 = np.percentile(seq_len, 95)

        # Filter the values to include only the top 95%
        filtered_seq_len = [length for length in seq_len if length <= percentile_95]

        plt.subplot(3, 3, p + 1)
        plt.hist(filtered_seq_len, bins=50)
        plt.title(subreddit_name)
        # plt.axis("off")
    plt.savefig(
        f"/home/penguin/Projects/SKIPSIE/two_exploratory_data_analysis/images/ disorder average post length",
        dpi=300,
    )


def lda_topic_proportions(posts):
    text = posts["Text"]

    vectorizer = CountVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(text)

    n_topics = 20
    lda_text = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_proportions_text = lda_text.fit_transform(X_text)

    topic_names = []
    for lda_model, topic_names in [
        (lda_text, topic_names),
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
        topic_proportions_text,
        topic_names,
    )


def topic_visualization(
    topic_proportions,
    topic_names,
):
    num_topics = len(topic_names)
    random.seed(42)  # Set a seed for reproducibility
    topic_colors = random.sample(list(mcolors.CSS4_COLORS.values()), num_topics)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=50, n_iter=500)
    X_tsne = tsne.fit_transform(topic_proportions)

    plt.figure(figsize=(12, 6))
    total = 0
    for i in range(num_topics):
        num_data_points_in_topic = (topic_proportions.argmax(axis=1) == i).sum()
        for j in range(
            num_data_points_in_topic
        ):  # Loop over data points within a topic
            index = j + total
            plt.scatter(
                X_tsne[index, 0],  # Correct index calculation
                X_tsne[index, 1],  # Correct index calculation
                color=topic_colors[i],
            )
        total = total + num_data_points_in_topic

    for i, topic_name in enumerate(topic_names):
        first_word = topic_name.split()[0]  # Extract the first word from the topic name
        plt.annotate(first_word, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.title("t-SNE Visualization of Stress-Related Topics")
    plt.colorbar()
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"/home/penguin/Projects/SKIPSIE/two_exploratory_data_analysis/images/ thing",
        dpi=300,
    )


# Define preprocessing functions
def preprocess_text(text):
    # Tokenize the text using NLTK
    tokens = word_tokenize(text)
    # You can add more preprocessing steps here (e.g., removing stop words, lemmatization)

    return tokens


def lda_words():
    data = pd.read_csv("/home/penguin/Projects/SKRIPSIE/DATA/Default/mhrs_default.csv")
    grouped_data = data[data["Subreddit"] != "mentalhealth"].groupby("Subreddit")
    depression_text = (
        grouped_data.get_group("depression")["Text"].fillna("").astype(str)
    )
    anxiety_text = grouped_data.get_group("Anxiety")["Text"].fillna("").astype(str)
    bpd_text = grouped_data.get_group("BPD")["Text"].fillna("").astype(str)
    autism_text = grouped_data.get_group("autism")["Text"].fillna("").astype(str)
    schizophrenia_text = (
        grouped_data.get_group("schizophrenia")["Text"].fillna("").astype(str)
    )
    bipolar_text = grouped_data.get_group("bipolar")["Text"].fillna("").astype(str)

    # Define common parameters
    n_topics = 10
    n_top_words = 10

    # Initialize dictionaries to store results
    topic_proportions = {}
    topic_names = {}

    for subreddit_text in [
        depression_text,
        anxiety_text,
        bpd_text,
        autism_text,
        schizophrenia_text,
        bipolar_text,
    ]:
        # Create and fit CountVectorizer
        vectorizer = CountVectorizer(max_features=1000)
        X_subreddit = vectorizer.fit_transform(subreddit_text)

        # Perform LDA
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        topic_proportions = lda_model.fit_transform(X_subreddit)

        # Get the feature names (words) from the CountVectorizer
        feature_names = vectorizer.get_feature_names_out()

        # Initialize a list to store topic names
        topic_names = []

        # Loop through each topic and assign names to the top words
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[: -n_top_words - 1 : -1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_name = f"Topic {topic_idx + 1}: {' '.join(top_words)}"
            topic_names.append(topic_name)

        for topic_name in topic_names:
            print(topic_name)


if __name__ == "__main__":
    data = pd.read_csv("/home/penguin/Projects/SKRIPSIE/DATA/Default/mhrs_default.csv")

    lda_words()

    # depression_dataset = pd.read_csv(
    #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_mentalhealth.csv"
    # )
    # depression_dataset = pd.read_csv(
    #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_depression.csv"
    # )
    # depression_dataset = pd.read_csv(
    #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_anxiety.csv"
    # )
    # depression_dataset = pd.read_csv(
    #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_bipolar.csv"
    # )
    # depression_dataset = pd.read_csv(
    #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_bpd.csv"
    # )
    # depression_dataset = pd.read_csv(
    #     "/home/penguin/Projects/SKIPSIE/DATA/Default/mhrs_default_depression.csv"
    # )
