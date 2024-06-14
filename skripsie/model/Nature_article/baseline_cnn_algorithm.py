import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import TextClassificationDataset
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import CNNTextClassifier

# Device configuration
# will do cuda after cpu working.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cuda:0")


# Define model parameters
num_filters = 128
filter_sizes = [5]
dropout_rate = 0.25
max_pooling_dim = 128
max_sequence_length = 20


# 1) SMOTE to fix class imbalance
# 2) word-embedding procedures from the pre-processed texts using the word2vec API of Python Package, Gensim
def cnn_algo(df, datasetname):
    # X = df["Text"]
    y = df["label"]

    word2vec_model = Word2Vec.load(
        "/home/penguin/Projects/SKRIPSIE/model/Nature_article/dreaddit.model"
    )
    vocab = word2vec_model.wv
    word_vectors = vocab.vectors
    embedding_dim = word_vectors.shape[1]
    vocab_size = len(vocab)

    sentences = df["Text"].tolist()
    tokenized_posts = [post.split() for post in sentences]
    post_embeddings = []

    # Assuming 'word2vec_model' is your pre-trained Word2Vec model
    for post in tokenized_posts:
        # Initialize an empty array to store word embeddings for this post
        post_embedding = np.zeros((150, 20))  # Initialize with zeros

        for i, word in enumerate(post[:150]):
            if word in word2vec_model.wv:
                post_embedding[i] = word2vec_model.wv[
                    word
                ]  # Use the word2vec embeddings

        post_embeddings.append(post_embedding)

    # format data for smote

    post_embeddings_array = np.stack(post_embeddings, axis=0)  # 3553 - (3553,150,20)
    X_resampled = post_embeddings_array.reshape(
        post_embeddings_array.shape[0], -1
    )  # (3553, 3000)

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_resampled, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    X_train = np.array(X_train)  # 2974 3000
    X_test = np.array(X_test)
    y_train = np.array(y_train)  # 2971
    y_test = np.array(y_test)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # Create DataLoader instances for training and testing
    batch_size = 64

    train_dataset = TextClassificationDataset(X_train, y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TextClassificationDataset(X_test, y_test)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = CNNTextClassifier(
        embedding_dim,
        num_filters,
        filter_sizes,
        dropout_rate,
        max_pooling_dim,
    )

    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_total_steps = len(train_data_loader)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (post_embedding, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            post_embedding = post_embedding.to(device)
            labels = labels.unsqueeze(1).to(device).float()
            outputs = model(post_embedding)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # if (i + 1) % 5 == 0:
            #     print(
            #         f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
            #     )
    print("finished Training")
    y_true = []
    y_pred = []
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted_labels = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=["class 0", "class 1"])
    # Parse the classification report to get individual metrics
    report_dict = classification_report(
        y_true, y_pred, target_names=["class 0", "class 1"], output_dict=True
    )

    # Access metrics for both classes
    class_0_metrics = report_dict["class 0"]
    class_1_metrics = report_dict["class 1"]

    # Access precision, recall, and f1 score for class 0 and class 1

    accuracy = accuracy_score(y_true, y_pred)

    print(report)
    print(f"Accuracy: {accuracy}")
    # Store results in a dictionary
    results = {}
    results[datasetname] = {
        "accuracy_best": accuracy,
        "precision_class_0": class_0_metrics["precision"],
        "recall_class_0": class_0_metrics["recall"],
        "f1_class_0": class_0_metrics["f1-score"],
        "precision_class_1": class_1_metrics["precision"],
        "recall_class_1": class_1_metrics["recall"],
        "f1_class_1": class_1_metrics["f1-score"],
    }

    return results


if __name__ == "__main__":
    # dataset_paths_default = [
    #     "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_anxiety.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_autism.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_bipolar.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_bpd.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_depression.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Default/mhrs_default_schizophrenia.csv",
    # ]
    # dataset_path_small = [
    #     "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_anxiety.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_autism.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_bipolar.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_bpd.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_depression.csv",
    #     "/home/harry/Documents/skripsie_workspace/DATA/Small/mhrs_small_schizophrenia.csv",
    # ]
    dataset_path_medium = [
        "/home/penguin/Projects/SKRIPSIE/DATA/medium/mhrs_medium_anxiety.csv",
        "/home/penguin/Projects/SKRIPSIE/DATA/medium/mhrs_medium_autism.csv",
        "/home/penguin/Projects/SKRIPSIE/DATA/medium/mhrs_medium_bipolar.csv",
        "/home/penguin/Projects/SKRIPSIE/DATA/medium/mhrs_medium_bpd.csv",
        "/home/penguin/Projects/SKRIPSIE/DATA/medium/mhrs_medium_depression.csv",
        "/home/penguin/Projects/SKRIPSIE/DATA/medium/mhrs_medium_schizophrenia.csv",
    ]
    dataset_names = [
        "anxiety",
        "autism",
        "bipolar",
        "bpd",
        "depression",
        "schizophrenia",
    ]

    dreaddit_name = ["depression"]
    dreaddit_path = [
        "/home/penguin/Projects/SKRIPSIE/DATA/test/mhrs_medium_depression.csv"
    ]

    for path, datasetname in zip(dreaddit_path, dreaddit_name):
        data = pd.read_csv(path)
        data.rename(columns={"Unnamed: 0": "id"}, inplace=True)

        # # ________________________________________________________________________
        data = data.drop("id", axis=1)
        data = data.dropna()
        # data.drop(["Subreddit"], axis=1)
        results = cnn_algo(data, datasetname)
        print(results)
        with open(f"Results_default/{datasetname}_results.json", "w") as f:
            json.dump(results, f, indent=4)
