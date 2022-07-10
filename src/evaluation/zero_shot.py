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

        
        embed_list = []
        for template in templates:#Done as a loop over lists because model can't handle 700 x 30
            texts = [template.format(classname) for classname in self.labels]
            with torch.no_grad():
                embed_list.append(self.model.encode_text(clip.tokenize(texts).to(self.device)))
        
        self.lab_embeds = torch.stack(embed_list, dim = 0)
        self.lab_embeds /= self.lab_embeds.norm(dim=-1, keepdim=True)
        self.lab_embeds = self.lab_embeds.to(torch.float32)

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

                emb /= emb.norm(dim=-1, keepdim=True)
                emb_agg = self.embedding_aggregator(emb)
                emb_agg = emb_agg.to(torch.float32)
                
                
                scores = (100.0 * emb_agg @ self.lab_embeds.T).softmax(dim=-1)

                total_scores = []
                for label in self.lab_embeds:
                    scores = (100.0 * emb_agg @ label.T).softmax(dim=-1)
                    total_scores.append(scores)
                 
                total_scores = torch.stack(total_scores, dim = 0)
                scores = torch.max(total_scores, dim = 0)[0]
                
                for i, lab in enumerate(labs):

                    values, best_15_inds = scores[i].topk(15)
                    best_15_labs = [self.labels[i] for i in best_15_inds]

                    count += emb.shape[0]
                    results["top1"] += (lab in best_15_labs[:1])
                    results["top5"] += (lab in best_15_labs[:5])
                    results["top15"] += (lab in best_15_labs[:15])

        for key in results.keys():
            results[key] /= count
        return results
    
templates = [
    'a photo of {}.',
    'a photo of a person {}.',
    'a photo of a person using {}.',
    'a photo of a person doing {}.',
    'a photo of a person during {}.',
    'a photo of a person performing {}.',
    'a photo of a person practicing {}.',
    'a video of {}.',
    'a video of a person {}.',
    'a video of a person using {}.',
    'a video of a person doing {}.',
    'a video of a person during {}.',
    'a video of a person performing {}.',
    'a video of a person practicing {}.',
    'a example of {}.',
    'a example of a person {}.',
    'a example of a person using {}.',
    'a example of a person doing {}.',
    'a example of a person during {}.',
    'a example of a person performing {}.',
    'a example of a person practicing {}.',
    'a demonstration of {}.',
    'a demonstration of a person {}.',
    'a demonstration of a person using {}.',
    'a demonstration of a person doing {}.',
    'a demonstration of a person during {}.',
    'a demonstration of a person performing {}.',
    'a demonstration of a person practicing {}.',
]
