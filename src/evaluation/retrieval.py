import open_clip
import numpy as np
import torch
from einops import rearrange

def retrieval_evaluation(model_video, model_text, data):
    if type(data) == dict:
        dataloader = data["val"].dataloader
    else:
        dataloader = data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_video_features, all_text_features = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            embeddings = batch["embeddings"]

            toks = []
            # TODO: does this require batch_size = 1 ??
            for cap in batch["text"]:
                for c in cap.split(";"): # multiple captions separated by ;
                    toks.append(open_clip.tokenize(c))
            
            toks = torch.cat(toks)
            embeddings = embeddings.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            video_embeddings = model_video(embeddings)
            text_embeddings = model_text(toks)

            all_video_features.append(video_embeddings.cpu())
            all_text_features.append(text_embeddings.cpu())
        
        video_features = torch.cat(all_video_features)
        text_features = torch.stack(all_text_features) # maintain the 3d structure of text_features

        val_metrics = get_metrics(
            video_features=video_features,
            text_features=text_features,
            logit_scale=100.0,
        )
    return val_metrics,

def get_metrics(video_features, text_features, logit_scale):
    ''' 
    Assumptions for this eval:

    - len(text_features) % len(video_features) == 0, i.e. there's a constant number 
    of captions per video (that constant can be 1).

    - video_features and text_features are arranged such that if we repeat
    each element of video_features N times along axis=0, with N being the nunber
    of captions per video, video_features[i] has ground truth label 
    text_features[i] for 0 <= i <= len(text_features), i.e. each index of
    video_features and text_features gives you a matching video-text pair 

    - both model_video and model_text spit out NORMALIZED embeddings

    Einops notation:
    n = num captions per video
    v = num videos
    t = num captions
    '''
    metrics = {}
    video_features = video_features.float()
    text_features = text_features.float()
    
    logits_per_text = (logit_scale * text_features @ video_features.T)
    logits_per_text = rearrange(logits_per_text, 'v n t -> n v t')
    
    min_rank, _ = torch.max(logits_per_text, dim=0)
    logits_per_video = min_rank.t().unsqueeze(0)
    
    logits = {'video_to_text': logits_per_video, 'text_to_video': logits_per_text}
    
    for name, logit in logits.items():
      argsorted = torch.argsort(logit, dim=-1, descending=True)
      ranks_on_diagonals = torch.argsort(argsorted, dim=-1, descending=False)
      
      rankings = torch.diagonal(ranks_on_diagonals, dim1=-1, dim2=-2).flatten().float()

      metrics[f"{name}_mean_rank"] = (rankings.mean() + 1).item()
      metrics[f"{name}_median_rank"] = (torch.floor(torch.median(rankings)) + 1).item()
      for k in [1, 5, 10]:
        metrics[f'{name}_R@{k}'] = torch.mean((rankings < k).float()).item()

    return metrics