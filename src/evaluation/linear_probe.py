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
        if b["embeddings"].shape[1] > 0:
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

    def evaluate(self):
        train_feat, train_lab = concat_features(self.train_dataloader, self.embedding_aggregator, self.labels)
        val_feat, val_lab = concat_features(self.val_dataloader, self.embedding_aggregator, self.labels)

        # Perform logistic regression
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        classifier.fit(train_feat, train_lab)

        # Evaluate using the logistic regression classifier
        predictions = classifier.predict_proba(val_feat)
        top_n = np.argsort(predictions)[:,:-15-1:-1]

        results = {
            "top1": 0,
            "top5": 0,
            "top15": 0,
        }

        for i in range(len(val_lab)):
            results["top1"] += (val_lab[i] in top_n[i, :1])
            results["top5"] += (val_lab[i] in top_n[i, :5])
            results["top15"] += (val_lab[i] in top_n[i])

        for k in results.keys():
            results[k] /= len(val_lab)
        return results
