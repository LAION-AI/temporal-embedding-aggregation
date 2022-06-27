import os
import glob

import clip
import torch
import numpy as np

class ZeroShotClassification:
    def __init__(self,
        dataloader,
        labels,
        embedding_aggregator,
        prompt_func=lambda x: x
    ):

        self.dataloader = dataloader
        self.embedding_aggregator = embedding_aggregator
        self.labels = labels

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)

        labels_ = [prompt_func(l) for l in self.labels] # prompt engineering
        with torch.no_grad():
            self.lab_embeds = self.model.encode_text(clip.tokenize(labels_).to(self.device))

    def evaluate(self):
        results = {
            "top1": 0,
            "top5": 0,
            "top15": 0,
        }

        count = 0
        with torch.no_grad():
            for batch in self.dataloader:
                emb = batch["embeddings"].to(self.device)
                labs = batch["text"]
                
                emb_agg = self.embedding_aggregator(emb)
                scores = (emb_agg @ self.lab_embeds.T).cpu().numpy() # (batch_size x n_classes)

                for i, lab in enumerate(labs):
                    best_15_inds = np.argpartition(scores[i], -15)[-15:]
                    best_15_labs = [self.labels[i] for i in best_15_inds]

                    count += emb.shape[0]
                    results["top1"] += (lab in best_15_labs[:1])
                    results["top5"] += (lab in best_15_labs[:5])
                    results["top15"] += (lab in best_15_labs[:15])

        for key in results.keys():
            results[key] /= count
        return results

            



