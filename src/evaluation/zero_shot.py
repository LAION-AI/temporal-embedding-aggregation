import os
import glob

import clip
import torch
import numpy as np


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


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

        # Create zero-shot classifier weights
        with open("evaluation/zs_templates.txt", "r") as f:
            templates = f.read().splitlines()

        with torch.no_grad():
            self.zeroshot_weights = []
            for classname in self.labels:
                texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts).to(self.device)
                class_embeddings = self.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                self.zeroshot_weights.append(class_embedding)
            self.zeroshot_weights = torch.stack(self.zeroshot_weights, dim=1).to(self.device)


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
                emb /= emb.norm(dim=-1, keepdim=True)
                labs = torch.Tensor([self.labels.index(l) for l in batch["text"]]).to(self.device)

                similarity = 100.0 * emb @ self.zeroshot_weights
                similarity = similarity.softmax(dim=-1) 
                scores = similarity.mean(dim=1)

                acc1, acc5, acc15 = accuracy(scores, labs, topk=(1, 5, 15))
                results["top1"] += acc1
                results["top5"] += acc5
                results["top15"] += acc15
                count += len(emb)

        for key in results.keys():
            results[key] /= count
        return results
