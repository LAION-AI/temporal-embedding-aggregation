import os
import glob

import clip
import torch
import numpy as np

from sklearn.linear_model import LogisticRegression

def concat_features(dataloader, aggr, labs):
    all_features = []
    all_labels = []
    for b in dataloader:
        b_labs = b["text"]
        aggr_embeds = aggr(b["embeddings"])
        all_features.append(aggr_embeds)
        for i, lab in enumerate(b_labs):
            all_labels.append(labs.index(b_labs[i]))

    return np.concatenate(all_features), np.array(all_labels)



class LinearProbeClassification:
    def __init__(self,
        train_dataloader,
        val_dataloader,
        embedding_aggregator,
        labels,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.embedding_aggregator = embedding_aggregator
        self.labels = labels

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def evaluate(self):
        train_feat, train_lab = concat_features(self.train_dataloader, self.embedding_aggregator, self.labels)
        val_feat, val_lab = concat_features(self.val_dataloader, self.embedding_aggregator, self.labels)

        # Perform logistic regression
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        classifier.fit(train_feat, train_lab)

        # Evaluate using the logistic regression classifier
        predictions = classifier.predict(val_feat)
        accuracy = np.mean((val_lab == predictions).astype(np.float)) * 100.
        print(f"Accuracy = {accuracy:.3f}")
