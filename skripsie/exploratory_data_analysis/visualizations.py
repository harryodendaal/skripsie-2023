import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from wordcloud import WordCloud


def create_word_cloud(n_gram_dict, position, title):
    # Create a word cloud
    wordcloud = WordCloud(
        width=800, height=400, max_words=100, background_color="white"
    ).generate_from_frequencies(n_gram_dict)

    # Add the subplot
    plt.subplot(2, 2, position)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")


def topic_visualization(
    topic_proportions_stress,
    topic_proportions_non_stress,
    stress_topic_names,
    non_stress_topic_names,
):
    # Perform t-SNE for stress-related topics
    tsne_stress = TSNE(n_components=2, perplexity=50, n_iter=500)
    X_tsne_stress = tsne_stress.fit_transform(topic_proportions_stress)

    # Perform t-SNE for non-stress-related topics
    tsne_non_stress = TSNE(n_components=2, perplexity=50, n_iter=500)
    X_tsne_non_stress = tsne_non_stress.fit_transform(topic_proportions_non_stress)

    # Create scatter plots to visualize t-SNE representations
    plt.figure(figsize=(12, 6))

    # Stress-related topics
    plt.subplot(1, 2, 1)
    plt.scatter(
        X_tsne_stress[:, 0],
        X_tsne_stress[:, 1],
        c="red",
        label="Stress",
        cmap="viridis",
    )
    for i, topic_name in enumerate(stress_topic_names):
        first_word = topic_name.split()[0]  # Extract the first word from the topic name
        plt.annotate(first_word, (X_tsne_stress[i, 0], X_tsne_stress[i, 1]))

    plt.title("t-SNE Visualization of Stress-Related Topics")
    plt.colorbar()
    plt.legend()

    # Non-stress-related topics
    plt.subplot(1, 2, 2)
    plt.scatter(
        X_tsne_non_stress[:, 0],
        X_tsne_non_stress[:, 1],
        c="blue",
        label="Non-Stress",
        cmap="viridis",
    )
    for i, topic_name in enumerate(non_stress_topic_names):
        first_word = topic_name.split()[0]  # Extract the first word from the topic name
        plt.annotate(first_word, (X_tsne_non_stress[i, 0], X_tsne_non_stress[i, 1]))

    plt.title("t-SNE Visualization of Non-Stress-Related Topics")
    plt.colorbar()
    plt.legend()

    plt.tight_layout()
    plt.savefig("sample_plot2.png", dpi=300)
    plt.show()
