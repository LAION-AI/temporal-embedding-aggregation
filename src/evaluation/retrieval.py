import open_clip
import numpy as np
import torch
from einops import rearrange

def retrieval_evaluation(model_video, data, multicaption=False):
    if type(data) == dict:
        dataloader = data["val"].dataloader
    else:
        dataloader = data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_video_features, all_text_features = [], []
    max_txt_len = 1
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            embeddings = batch["embeddings"]
            toks = []
            # TODO: does this require batch_size = 1 ??
            for cap in batch["text"]:
                if multicaption:
                    caps_list = cap.split(';')
                    max_txt_len = max(max_txt_len, len(caps_list))
            
                    for c in caps_list: # multiple captions separated by ;
                        toks.append(open_clip.tokenize(c))
                else:
                    toks.append(open_clip.tokenize(cap))

            toks = torch.cat(toks)
            embeddings = embeddings.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            video_embeddings, text_embeddings, _ = model_video(embeddings, toks, prenorm=True, postnorm=True)

            all_video_features.append(video_embeddings.cpu())
            all_text_features.append(text_embeddings.cpu())

        dim_model = all_video_features[0].shape[-1]

        text_features = torch.stack(
            zero_pad_text_features(all_text_features, max_txt_len, dim_model=dim_model)
        ) if multicaption else torch.cat(all_text_features).unsqueeze(1)
        video_features = torch.cat(all_video_features)
        
        val_metrics = get_metrics(
            video_features=video_features,
            text_features=text_features,
            logit_scale=100.0,
        )
    return val_metrics

def zero_pad_text_features(text_features, max_txt_len, dim_model=512):
    out = []
    for mat in text_features:
        padded = torch.zeros(size=(max_txt_len, dim_model))
        padded[0:len(mat), :] = mat

        out.append(padded)

    return out
    
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
        
        diagonal_logits = torch.diagonal(logit, dim1=-1, dim2=-2).flatten().float()
        mask = ~(diagonal_logits == 0) # make sure we take out zero padded logits
        rankings = rankings[mask]
            
        metrics[f"{name}_mean_rank"] = (rankings.mean() + 1).item()
        metrics[f"{name}_median_rank"] = (torch.floor(torch.median(rankings)) + 1).item()
        for k in [1, 5, 10]:
            metrics[f'{name}_R@{k}'] = torch.mean((rankings < k).float()).item()

    return metrics
