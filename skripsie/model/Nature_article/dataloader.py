import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, post_embeddings, labels):
        self.post_embeddings = post_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.post_embeddings)

    def __getitem__(self, idx):
        post_embedding = self.post_embeddings[idx]
        label = self.labels[idx]

        # Transpose the post_embedding tensor to shape [20, 150] (or [embedding_dim, sequence_length])
        post_embedding = torch.tensor(post_embedding.T, dtype=torch.float32)

        label = torch.tensor(label, dtype=torch.float32)

        return post_embedding, label
