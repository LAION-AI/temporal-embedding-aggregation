import os
import glob

import clip
import torch
import numpy as np


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def zero_shot_eval(dataloader, labels, embedding_aggregator, prompt_func=labmda x: x)
    dataloader = dataloader
    embedding_aggregator = embedding_aggregator
    labels = labels

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: generalize this to any text model, ideally pass in
    model, _ = clip.load("ViT-B/32", device=device)

    # Create zero-shot classifier weights
    with open("evaluation/zs_templates.txt", "r") as f:
        templates = f.read().splitlines()

    with torch.no_grad():
        zeroshot_weights = []
        for classname in labels:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    results = {
        "top1": 0,
        "top5": 0,
        "top15": 0,
    }

    count = 0
    with torch.no_grad():
        for batch in dataloader:
            emb = batch["embeddings"].to(device)
            emb /= emb.norm(dim=-1, keepdim=True)
            labs = torch.Tensor([labels.index(l) for l in batch["text"]]).to(device)

            similarity = 100.0 * emb @ zeroshot_weights
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
